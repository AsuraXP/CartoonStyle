"""Narration TTS - ONE continuous take, word timestamps, image-moment sync.

The narrator reads the whole script as a single take: perfect prosody, no
gaps between beats. Words are timestamped individually - that is the sync
source images flip on.

Matching rule: moments are matched SEQUENTIALLY against the spoken word
stream, with numerals expanded the way TTS says them ("40,000" -> "forty
thousand"). Anything that cannot be matched is REPORTED, never silently
misplaced.

Outputs (in --out-dir):
  narration.mp3            the single take
  audio/beatNN.mp3         per-beat slices (easy review)
  beats_with_timing.json   beat spans (gapless cover of [0, total])
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


_ONES = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
         "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
         "seventeen", "eighteen", "nineteen"]
_TENS = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]


def num2words_en(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + (" " + _ONES[n % 10] if n % 10 else "")
    if n < 1000:
        return _ONES[n // 100] + " hundred" + (" " + num2words_en(n % 100) if n % 100 else "")
    for i, suf in ((1_000_000_000, "billion"), (1_000_000, "million"), (1000, "thousand")):
        if n >= i:
            return num2words_en(n // i) + " " + suf + ((" " + num2words_en(n % i)) if n % i else "")
    return str(n)


def norm_words(text: str) -> list[str]:
    """Tokens for matching. edge-tts WordBoundary events emit numerals as raw
    digit strings ('1980s', '1963', '264') and contractions split as separate
    words (\"couldn't\" -> [\"couldn\", \"t\"]), \"seventeen-year-old\" ->
    [\"seventeen\", \"year\", \"old\"]). Mirror that exactly so predictions match
    the actual token stream."""
    return re.findall(r"[a-z0-9]+", text.lower())


def norm_variants(text: str) -> list[list[str]]:
    """Token variants to try. edge-tts emits numerals as raw digit strings,
    contractions split at apostrophe, hyphens split - just lowercase+alnum."""
    return [norm_words(text)]


def match_first(needles: list[list[str]], hay: list[str], start: int = 0) -> tuple[int, int] | None:
    """(index, length) of the first variant found at/after start."""
    best = None
    for nd in needles:
        idx = find_subseq(nd, hay, start)
        if idx is not None and (best is None or idx < best[0]):
            best = (idx, len(nd))
    return best


def find_subseq(needle: list[str], hay: list[str], start: int = 0) -> int | None:
    n = len(needle)
    if n == 0 or n > len(hay):
        return None
    for i in range(start, len(hay) - n + 1):
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

    # edge-tts 7.x defaults to SentenceBoundary and DISABLES word events -
    # we must explicitly ask for WordBoundary. edge-tts 6.x has no such param
    # (it always sends words) -> fall back.
    try:
        comm = edge_tts.Communicate(text, voice, rate=rate, boundary="WordBoundary")
    except TypeError:
        comm = edge_tts.Communicate(text, voice, rate=rate)

    audio = bytearray()
    words: list[dict] = []
    async for chunk in comm.stream():
        if chunk["type"] == "audio":
            audio += chunk["data"]
        elif chunk["type"] == "WordBoundary":
            words.append({"w": chunk["text"], "t": chunk["offset"] / 1e7})
    if not words:
        raise SystemExit(
            "edge-tts returned 0 WordBoundary events - word timing is impossible. "
            "Check the edge_tts version (needs >=6.1.9, boundary support since 7.x)."
        )
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(bytes(audio))
    return words


def stream_tokens(words: list[dict]) -> tuple[list[str], list[float]]:
    """Word-stream tokens + per-token times (always aligned with each other).
    Numerals/digits in edge-tts WordBoundary text are expanded to words the
    same way norm_words does (e.g. "1980s" -> ["nineteen", "eighties"]) so
    they match the predicted-token side."""
    toks: list[str] = []
    ts: list[float] = []
    for w in words:
        ws = norm_words(w["w"])  # expand digits/decades consistently
        for tok in ws:
            toks.append(tok)
            ts.append(w["t"])
    return toks, ts


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
        prev = out_dir / "beats_with_timing.json"
        if prev.exists():
            old = json.loads(prev.read_text(encoding="utf-8"))
            words = [{"w": w["w"], "t": w["t"]} for b in old for w in b.get("words", [])]
        print("[tts] narration.mp3 cached - reusing saved word timings", flush=True)

    total = audio_duration(str(narr))

    # ---- map every beat onto the spoken word stream ------------------------
    sp_tokens, sp_times = stream_tokens(words)
    sp_times.append(total)

    timed = []
    for i, b in enumerate(items):
        beat = int(b["beat"])
        hit = match_first(norm_variants(spoken[beat]), sp_tokens)
        if hit is not None:
            idx, nlen = hit
            start = sp_times[idx]
            end = sp_times[min(idx + nlen, len(sp_times) - 1)]
        else:
            frac = i / max(1, len(items))
            start, end = total * frac, total * (i + 1) / max(1, len(items))
            print(f"[tts] WARNING: beat {beat} text not found in word stream (numerals?) - proportional span", flush=True)
        timed.append({**b, "spoken": spoken[beat], "start": start, "sec": end - start,
                      "words": [w for w in words if start <= w["t"] < end]})

    # force a gapless cover of [0, total]: images stream then matches audio 1:1
    for i in range(len(timed)):
        if i == 0:
            timed[i]["start"] = 0.0
        else:
            timed[i]["start"] = max(timed[i]["start"], timed[i - 1]["start"] + 0.2)
    for i in range(len(timed)):
        end = timed[i + 1]["start"] if i + 1 < len(timed) else total
        timed[i]["sec"] = end - timed[i]["start"]
        timed[i]["words"] = [w for w in words if timed[i]["start"] <= w["t"] < end]

    for b in timed:
        b["start"] = round(b["start"], 3)
        b["sec"] = round(b["sec"], 3)

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

    # ---- moments -> gapless tiling, SEQUENTIAL word-locked -----------------
    if args.visuals:
        import csv as _csv

        rows = list(_csv.DictReader(open(args.visuals, encoding="utf-8")))
        groups: dict[int, list[dict]] = {}
        for r in rows:
            groups.setdefault(int(r["slide_id"]), []).append(r)

        moments = []
        missed = 0
        for b in timed:
            beat = int(b["beat"])
            ms = groups.get(beat, [])
            if not ms:
                continue
            toks, ts = stream_tokens(b["words"])
            toks_raw = [re.findall(r"[a-z0-9]+", w["w"].lower()) for w in b["words"]]
            ts_raw = [w["t"] for w in b["words"]]
            desired: list[float] = []
            pos = 0
            lead = 0.0
            for m in ms:
                hit = match_first(norm_variants(m["moment_text"]), toks, pos)
                if hit is not None:
                    idx, nlen = hit
                    desired.append(ts[idx])
                    pos = idx + nlen
                else:
                    missed += 1
                    print(f"[tts] WARNING: moment beat{int(m['slide_id']):02d}m{int(m.get('moment') or 0):02d} "
                          f"words not in speech: {m['moment_text']!r} - lead-in slot", flush=True)
                    desired.append(b["start"] + lead)
                    lead += 1.2
            order = sorted(range(len(ms)), key=lambda j: (desired[j], j))
            tiled = []
            run = b["start"]
            for pos2, j in enumerate(order):
                nxt = desired[order[pos2 + 1]] if pos2 + 1 < len(order) else b["start"] + b["sec"]
                if nxt <= run:
                    nxt = run + 0.4
                tiled.append((j, run, nxt - run))
                run = nxt
            for j, st, sec in tiled:
                sid = str(ms[j]["slide_id"]).zfill(2)
                mom = str(ms[j].get("moment") or j + 1).zfill(2)
                moments.append({"key": f"beat{sid}m{mom}", "beat": beat,
                                "start": round(st, 3), "sec": round(sec, 3),
                                "text": ms[j]["moment_text"]})
        (out_dir / "moments_timing.json").write_text(json.dumps(moments, indent=1), encoding="utf-8")
        status = "OK" if missed == 0 else f"{missed} MISSED - CHECK WARNINGS ABOVE"
        print(f"[tts] {len(moments)} image moments locked to spoken words -> moments_timing.json [{status}]",
              flush=True)


if __name__ == "__main__":
    main()
