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


def find_image(images_dir: str, key: str, pick: int, fallback_dir: str | None) -> str | None:
    pat = f"{key}_v{pick:02d}_s*.png"
    hits = sorted(glob.glob(str(Path(images_dir) / pat)))
    if not hits and fallback_dir:
        hits = sorted(glob.glob(str(Path(fallback_dir) / pat)))
    return hits[0] if hits else None


def beat_segment(cmd_base: list[str], work: Path, beat: int, entries: list[tuple[str, float]],
                 audio: Path) -> Path:
    """One beat = one audio file + one or more images (word-locked)."""
    lst = work / f"imgs{beat:02d}.txt"
    lines = []
    for img, sec in entries:
        lines.append(f"file '{img}'\nduration {sec:.3f}\n")
    lines.append(f"file '{entries[-1][0]}'\n")  # concat demuxer wants the last file repeated
    lst.write_text("".join(lines), encoding="utf-8")
    seg = work / f"seg{beat:02d}.mp4"
    subprocess.run(cmd_base + ["-f", "concat", "-safe", "0", "-i", str(lst),
                               "-i", str(audio),
                               "-c:v", "libx264", "-tune", "stillimage", "-r", "30",
                               "-c:a", "aac", "-b:a", "192k",
                               "-pix_fmt", "yuv420p", "-shortest", str(seg)],
                   check=True, capture_output=True)
    return seg


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
    mom_path = Path(args.timing).parent / "moments_timing.json"
    moments = json.loads(mom_path.read_text(encoding="utf-8")) if mom_path.exists() else []
    by_beat: dict[int, list[dict]] = {}
    for m in moments:
        by_beat.setdefault(int(m["beat"]), []).append(m)

    work = Path(tempfile.mkdtemp(prefix="stitch_"))
    segments = []
    missing = []
    cmd_base = ["ffmpeg", "-y", "-loglevel", "error"]

    for b in beats:
        n = int(b["beat"])
        audio = Path(args.timing).parent / "audio" / f"beat{n:02d}.mp3"
        entries: list[tuple[str, float]] = []
        for m in sorted(by_beat.get(n, []), key=lambda x: x["start"]):
            img = find_image(args.images_dir, m["key"], args.pick, args.fallback_dir or None)
            if img:
                entries.append((img, float(m["sec"])))
        if not entries:
            img = find_image(args.images_dir, f"beat{n:02d}", args.pick, args.fallback_dir or None)
            if img:
                entries.append((img, float(b["sec"])))
        if not entries or not audio.exists():
            missing.append(n)
            continue
        seg = beat_segment(cmd_base, work, n, entries, audio)
        segments.append(seg)
        print(f"[stitch] beat {n:>2}  {b['sec']:>5.1f}s  {len(entries)} image(s)", flush=True)

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
