"""Narration TTS - ONE continuous take, word timestamps, image-moment sync.

The narrator reads the whole script as a single take: perfect prosody, no
gaps between beats (concatenated per-beat files would click and pause).
Words are still timestamped individually - that is the sync source images
flip on.

Sentences starting with "sfx:" are production cues, not speech: they are
removed from the narration and kept as short lead-in images.

Outputs (in --out-dir):
  narration.mp3            the single take
  audio/beatNN.mp3         per-beat slices (easy review)
  beats_with_timing.json   beat spans inside the narration
  moments_timing.json      every image moment, gapless tiling of the audio
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import shutil
import subprocess
from pathlib import Path


def audio_duration(path: str) -> float:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise SystemExit("ffprobe not found - install ffmpeg (Kaggle/Colab images include it)")
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip())


def norm_words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def find_subseq(needle: list[str], hay: list[str]) -> int | None:
    n = len(needle)
    if n == 0 or n > len(hay):
        return None
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return i
    return None


def strip_sfx(beat_text: str) -> str:
    """'sfx: ...' sentences are cues for images/sound design, not spoken words."""
    parts = re.split(r"(?<=[.!?])\s+", beat_text.strip())
    spoken = [p for p in parts if not p.lower().startswith("sfx")]
    return " ".join(spoken).strip()


def load_beats(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    raise SystemExit("pass beats.jsonl (from scriptwriter package)")


async def synth_one_take(text: str, voice: str, rate: str, dest: Path) -> list[dict]:
    """One synthesis = natural prosody across the whole script + word timings."""
    import edge_tts

    audio = bytearray()
    words: list[dict] = []
    async for chunk in edge_tts.Communicate(text, voice, rate=rate).stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
        elif chunk["type"] == "WordBoundary":
            words.append({"w": chunk["text"], "t": chunk["offset"] / 1e7})
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(bytes(audio))
    return words


def main() -> None:
    p = argparse.ArgumentParser(description="one-take narration TTS with word timings")
    p.add_argument("--beats", required=True, help="beats.jsonl from scriptwriter package")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--voice", default="en-US-AndrewNeural")  # user-picked narrator
    p.add_argument("--rate", default="+0%", help='speaking pace, e.g. "+5%"')
    p.add_argument("--visuals", default="", help="visuals.csv (moment rows) -> moments_timing.json")
    p.add_argument("--force", action="store_true", help="re-synthesize")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    items = load_beats(args.beats)
    spoken = {int(b["beat"]): strip_sfx(b["text"]) for b in items}
    full_text = " ".join(s for s in spoken.values() if s)
    narr = out_dir / "narration.mp3"
    print(f"[tts] {len(items)} beats | one take | voice {args.voice} | rate {args.rate}", flush=True)

    words: list[dict] = []
    if not (narr.exists() and not args.force):
        words = asyncio.run(synth_one_take(full_text, args.voice, args.rate, narr))
        print(f"[tts] narration.mp3 written ({len(words)} words tracked)", flush=True)
    else:
        # reuse cached take: re-track is not possible offline; timings already saved
        prev = out_dir / "beats_with_timing.json"
        if prev.exists():
            old = json.loads(prev.read_text(encoding="utf-8"))
            words = [{"w": w["w"], "t": w["t"]} for b in old for w in b.get("words", [])]
        print("[tts] narration.mp3 cached - reusing saved word timings", flush=True)

    total = audio_duration(str(narr))

    # ---- map every beat onto the spoken word stream ------------------------
    sp_tokens = norm_words(" ".join(w["w"] for w in words))
    sp_times: list[float] = []
    for w in words:
        sp_times.extend([w["t"]] * max(1, len(norm_words(w["w"]))))
    sp_times.append(total)

    timed = []
    for i, b in enumerate(items):
        beat = int(b["beat"])
        toks = norm_words(spoken[beat])
        idx = find_subseq(toks, sp_tokens)
        if idx is not None and toks:
            start = sp_times[idx]
            end = sp_times[min(idx + len(toks), len(sp_times) - 1)]
        else:
            frac = i / max(1, len(items))
            start, end = total * frac, total * (i + 1) / max(1, len(items))
        timed.append({**b, "spoken": spoken[beat], "start": round(start, 2), "sec": round(end - start, 2),
                      "words": [w for w in words if start <= w["t"] < end]})

    (out_dir / "beats_with_timing.json").write_text(json.dumps(timed, indent=1), encoding="utf-8")
    print(f"[tts] total narration: {int(total // 60)}:{int(total % 60):02d} -> beats_with_timing.json", flush=True)

    # ---- per-beat slices for easy listening --------------------------------
    (out_dir / "audio").mkdir(parents=True, exist_ok=True)
    for b in timed:
        dest = out_dir / "audio" / f"beat{int(b['beat']):02d}.mp3"
        if not dest.exists() or args.force:
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(narr),
                            "-ss", f"{b['start']:.2f}", "-t", f"{b['sec']:.2f}",
                            "-c:a", "libmp3lame", "-q:a", "4", str(dest)], check=True)

    # ---- moments -> gapless tiling of the audio ----------------------------
    if args.visuals:
        import csv as _csv

        rows = list(_csv.DictReader(open(args.visuals, encoding="utf-8")))
        groups: dict[int, list[dict]] = {}
        for r in rows:
            groups.setdefault(int(r["slide_id"]), []).append(r)

        moments = []
        for b in timed:
            beat = int(b["beat"])
            ms = groups.get(beat, [])
            if not ms:
                continue
            # tokens + times straight from the word stream (always aligned)
            toks: list[str] = []
            ts: list[float] = []
            for w in b["words"]:
                ws = norm_words(w["w"])
                if not ws:
                    continue
                for tok in ws:
                    toks.append(tok)
                    ts.append(w["t"])
            desired: list[float] = []
            lead = 0.0
            for m in ms:
                idx = find_subseq(norm_words(m["moment_text"]), toks)
                if idx is not None:
                    desired.append(ts[idx])
                else:  # sfx cue / unmatched -> short lead-in slot
                    desired.append(b["start"] + lead)
                    lead += 1.2
            order = sorted(range(len(ms)), key=lambda j: (desired[j], j))
            # gapless tiling: each segment ends exactly where the next begins
            tiled = []
            run = b["start"]
            for pos, j in enumerate(order):
                nxt = desired[order[pos + 1]] if pos + 1 < len(order) else b["start"] + b["sec"]
                if nxt <= run:
                    nxt = run + 0.4
                tiled.append((j, run, nxt - run))
                run = nxt
            for j, st, sec in tiled:
                sid = str(ms[j]["slide_id"]).zfill(2)
                mom = str(ms[j].get("moment") or j + 1).zfill(2)
                moments.append({"key": f"beat{sid}m{mom}", "beat": beat,
                                "start": round(st, 2), "sec": round(sec, 2),
                                "text": ms[j]["moment_text"]})
        (out_dir / "moments_timing.json").write_text(json.dumps(moments, indent=1), encoding="utf-8")
        print(f"[tts] {len(moments)} image moments locked to spoken words -> moments_timing.json", flush=True)


if __name__ == "__main__":
    main()
