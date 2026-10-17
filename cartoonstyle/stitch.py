"""Final stage: ffmpeg auto-stitch - one image per beat, exact TTS timing.

Reads beats_with_timing.json (from cartoonstyle.tts) and a directory of
generated images named beatNN_vMM_sSEED.png (from cartoonstyle.generate_flux2),
picks one variant per beat, and assembles narration-driven VIDEO.mp4.

Usage:
  python -m cartoonstyle.stitch --timing examples/my_video/beats_with_timing.json \
      --images-dir /kaggle/working/cartoonstyle-out/images \
      --out /kaggle/working/VIDEO.mp4 [--pick 0] [--fallback-dir ...]

--pick N  which variant to show per beat (v00, v01, ...). Choose after review,
          or generate several videos with different picks.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def find_image(images_dir: str, beat: int, pick: int, fallback_dir: str | None) -> str | None:
    pat = f"beat{beat:02d}_v{pick:02d}_s*.png"
    hits = sorted(glob.glob(str(Path(images_dir) / pat)))
    if not hits and fallback_dir:
        hits = sorted(glob.glob(str(Path(fallback_dir) / pat)))
    return hits[0] if hits else None


def main() -> None:
    p = argparse.ArgumentParser(description="stitch images + narration into VIDEO.mp4")
    p.add_argument("--timing", required=True, help="beats_with_timing.json")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pick", type=int, default=0, help="variant number per beat")
    p.add_argument("--fallback-dir", default="")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found - install ffmpeg (Kaggle/Colab images include it)")

    beats = json.loads(Path(args.timing).read_text(encoding="utf-8"))
    work = Path(tempfile.mkdtemp(prefix="stitch_"))
    segments = []
    missing = []

    for b in beats:
        n = int(b["beat"])
        img = find_image(args.images_dir, n, args.pick, args.fallback_dir or None)
        audio = Path(args.timing).parent / "audio" / f"beat{n:02d}.mp3"
        if not img or not audio.exists():
            missing.append(n)
            continue
        seg = work / f"seg{n:02d}.mp4"
        cmd = ["ffmpeg", "-y", "-loop", "1", "-framerate", str(args.fps),
               "-t", f"{b['sec']:.3f}", "-i", img,
               "-i", str(audio),
               "-c:v", "libx264", "-tune", "stillimage",
               "-c:a", "aac", "-b:a", "192k",
               "-pix_fmt", "yuv420p", "-shortest",
               str(seg)]
        subprocess.run(cmd, check=True, capture_output=True)
        segments.append(seg)
        print(f"[stitch] beat {n:>2}  {b['sec']:>5.1f}s  {Path(img).name}", flush=True)

    if missing:
        print(f"[stitch] WARNING: no image/audio for beats {missing} - they are skipped", flush=True)
    if not segments:
        raise SystemExit("nothing to stitch")

    lst = work / "list.txt"
    lst.write_text("".join(f"file '{s}'\n" for s in segments), encoding="utf-8")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", args.out], check=True, capture_output=True)
    total = sum(Path(s).stat().st_size for s in segments) / 1e6
    print(f"[stitch] VIDEO -> {args.out}  ({len(segments)} beats, ~{total:.0f}MB source)", flush=True)


if __name__ == "__main__":
    main()
