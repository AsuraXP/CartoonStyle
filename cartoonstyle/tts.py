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


async def synth_edge(items: list[dict], out_dir: Path, voice: str, rate: str, force: bool) -> None:
    import edge_tts

    for b in items:
        dest = out_dir / "audio" / f"beat{int(b['beat']):02d}.mp3"
        if dest.exists() and not force:
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        await edge_tts.Communicate(b["text"], voice, rate=rate).save(str(dest))
        print(f"[tts] beat {b['beat']:>2} -> {dest.name}", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(description="per-beat narration TTS")
    p.add_argument("--beats", required=True, help="beats.jsonl from scriptwriter package")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--voice", default="en-US-ChristopherNeural")
    p.add_argument("--rate", default="+0%", help='speaking pace, e.g. "+5%"')
    p.add_argument("--backend", choices=["edge"], default="edge")
    p.add_argument("--force", action="store_true", help="re-synthesize existing files")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    items = load_beats(args.beats)
    print(f"[tts] {len(items)} beats | voice {args.voice} | rate {args.rate}", flush=True)

    if args.backend == "edge":
        asyncio.run(synth_edge(items, out_dir, args.voice, args.rate, args.force))

    t = 0.0
    timed = []
    for b in items:
        mp3 = out_dir / "audio" / f"beat{int(b['beat']):02d}.mp3"
        sec = audio_duration(str(mp3))
        timed.append({**b, "start": round(t, 2), "sec": round(sec, 2)})
        t += sec
    (out_dir / "beats_with_timing.json").write_text(json.dumps(timed, indent=1), encoding="utf-8")
    print(f"[tts] total narration: {int(t // 60)}:{int(t % 60):02d} -> beats_with_timing.json", flush=True)


if __name__ == "__main__":
    main()
