"""Stage 3 of the narration-first workflow: per-beat TTS narration.

Audio is the master - images follow it. This synthesizes one audio file per
beat and records EXACT durations, so the video stitches in perfect sync.

Default backend: edge-tts (Microsoft neural voices, free, no API key).
The backend is pluggable --voice is a plain string; any voice of any
future backend (Gemini, ElevenLabs, ...) plugs in the same way.

Usage:
  python -m cartoonstyle.tts --beats examples/my_video/beats.jsonl \
      --out-dir examples/my_video --voice en-US-ChristopherNeural

Output:
  audio/beatNN.mp3  +  beats_with_timing.json  (beat, text, start, sec)
"""
from __future__ import annotations

import argparse
import asyncio
import json
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


def load_beats(path: str) -> list[dict]:
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    raise SystemExit("pass beats.jsonl (from scriptwriter package)")


def norm_words(text: str) -> list[str]:
    import re
    return re.findall(r"[a-z0-9]+", text.lower())


def find_subseq(needle: list[str], hay: list[str]) -> int | None:
    n = len(needle)
    for i in range(len(hay) - n + 1):
        if hay[i:i + n] == needle:
            return i
    return None


async def synth_edge(items: list[dict], out_dir: Path, voice: str, rate: str,
                     force: bool) -> dict[int, list[dict]]:
    """Synthesize per-beat audio AND capture word timestamps (the image sync source)."""
    import edge_tts

    words_by_beat: dict[int, list[dict]] = {}
    for b in items:
        beat = int(b["beat"])
        dest = out_dir / "audio" / f"beat{beat:02d}.mp3"
        dest.parent.mkdir(parents=True, exist_ok=True)
        words: list[dict] = []
        if dest.exists() and not force:
            words_by_beat[beat] = words
            continue
        audio = bytearray()
        async for chunk in edge_tts.Communicate(b["text"], voice, rate=rate).stream():
            if chunk["type"] == "audio":
                audio += chunk["data"]
            elif chunk["type"] == "WordBoundary":
                words.append({"w": chunk["text"], "t": chunk["offset"] / 1e7})
        dest.write_bytes(bytes(audio))
        words_by_beat[beat] = words
        print(f"[tts] beat {beat:>2} -> {dest.name}  ({len(words)} words tracked)", flush=True)
    return words_by_beat


def resolve_moments(timed_beats: list[dict], words_by_beat: dict[int, list[dict]],
                    visuals_path: str, out_dir: Path) -> None:
    """Map every image moment to its spoken word span -> moments_timing.json."""
    import csv as _csv
    import re as _re

    rows = list(_csv.DictReader(open(visuals_path, encoding="utf-8")))
    groups: dict[int, list[dict]] = {}
    for r in rows:
        groups.setdefault(int(r["slide_id"]), []).append(r)

    moments = []
    for b in timed_beats:
        beat = int(b["beat"])
        ms = groups.get(beat, [])
        if not ms:
            continue
        beat_words = norm_words(b["text"])
        spoken = words_by_beat.get(beat, [])
        # spoken word tokens aligned to beat_words (edge-tts boundaries ~ match tokens)
        sp_tokens = norm_words(" ".join(w["w"] for w in spoken))
        sp_times: list[float] = []
        for w in spoken:
            sp_times.extend([w["t"]] * max(1, len(norm_words(w["w"]))))
        starts: list[float | None] = []
        for m in ms:
            idx = find_subseq(norm_words(m["moment_text"]), beat_words)
            t = None
            if idx is not None and idx < len(sp_times):
                t = sp_times[idx]
            elif idx is not None and spoken:
                frac = idx / max(1, len(beat_words))
                t = spoken[0]["t"] + frac * (b["sec"] - spoken[0]["t"])
            starts.append(t)
        for j, m in enumerate(ms):
            nxt = None
            for k in range(j + 1, len(ms)):
                if starts[k] is not None:
                    nxt = starts[k]
                    break
            st = starts[j] if starts[j] is not None else (0.0 if j == 0 else None)
            if st is None:  # fall back: equal split of remainder
                st = b["sec"] * j / len(ms)
            end = nxt if nxt is not None else b["sec"]
            sec = max(0.3, end - st)
            sid = str(m["slide_id"]).zfill(2)
            mom = str(m.get("moment") or j + 1).zfill(2)
            moments.append({"key": f"beat{sid}m{mom}", "beat": beat,
                            "start": round(b["start"] + st, 2),
                            "sec": round(sec, 2), "text": m["moment_text"]})
    (out_dir / "moments_timing.json").write_text(json.dumps(moments, indent=1), encoding="utf-8")
    print(f"[tts] {len(moments)} image moments locked to spoken words -> moments_timing.json", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="per-beat narration TTS")
    p.add_argument("--beats", required=True, help="beats.jsonl from scriptwriter package")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--voice", default="en-US-ChristopherNeural")
    p.add_argument("--rate", default="+0%", help='speaking pace, e.g. "+5%"')
    p.add_argument("--backend", choices=["edge"], default="edge")
    p.add_argument("--force", action="store_true", help="re-synthesize existing files")
    p.add_argument("--visuals", default="", help="visuals.csv (moment rows) -> moments_timing.json")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    items = load_beats(args.beats)
    print(f"[tts] {len(items)} beats | voice {args.voice} | rate {args.rate}", flush=True)

    words_by_beat: dict[int, list[dict]] = {}
    if args.backend == "edge":
        words_by_beat = asyncio.run(synth_edge(items, out_dir, args.voice, args.rate, args.force))

    t = 0.0
    timed = []
    for b in items:
        mp3 = out_dir / "audio" / f"beat{int(b['beat']):02d}.mp3"
        sec = audio_duration(str(mp3))
        timed.append({**b, "start": round(t, 2), "sec": round(sec, 2),
                      "words": words_by_beat.get(int(b["beat"]), [])})
        t += sec
    (out_dir / "beats_with_timing.json").write_text(json.dumps(timed, indent=1), encoding="utf-8")
    print(f"[tts] total narration: {int(t // 60)}:{int(t % 60):02d} -> beats_with_timing.json", flush=True)
    if args.visuals:
        resolve_moments(timed, words_by_beat, args.visuals, out_dir)


if __name__ == "__main__":
    main()
