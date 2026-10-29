#!/usr/bin/env python3
"""Phase 6 helper - publish assets derived from MEASURED output.

    python tools/publish.py --repo .                 # measured runtime + chapter table
    python tools/publish.py --repo . --video final.mp4   # + exact runtime of the render

Reads build/timings.csv (Phase 4 output - never estimates) and prints:
  * runtime (probed from --video when given, else the measured timeline end)
  * a paste-ready chapter list: one chapter per frame, promoted only when
    >= --min-gap seconds have passed since the last chapter start

Titles are derived from frame ids as a starting point - the assistant still
verifies every line against timings.csv and polishes titles (GATE 6).
Exits 2 if timings.csv is missing (run Phase 4 first).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_DEFAULT))

from vidsync.config import build_dir, load_config  # noqa: E402


def mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60}:{seconds % 60:02d}"


def title_from_id(frame_id: str) -> str:
    slug = re.sub(r"^\d+_", "", frame_id)
    words = re.sub(r"[_\-]+", " ", slug).strip()
    return words[:1].upper() + words[1:] if words else frame_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=str(REPO_ROOT_DEFAULT))
    ap.add_argument("--video", default=None, help="rendered file to probe for exact runtime")
    ap.add_argument("--min-gap", type=float, default=45.0, help="minimum seconds between chapter starts")
    args = ap.parse_args()

    repo_root = Path(args.repo).resolve()
    cfg = load_config(repo_root)
    csv_path = build_dir(cfg, repo_root) / "timings.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found - run Phase 4 (--dry-run) first; never invent timings.", file=sys.stderr)
        return 2

    with open(csv_path, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("ERROR: timings.csv is empty.", file=sys.stderr)
        return 2

    measured_end = max(float(r["hold_end_s"]) for r in rows)

    runtime = measured_end
    runtime_src = "measured timeline"
    if args.video:
        video = Path(args.video)
        if not video.is_absolute():
            video = Path.cwd() / video
        if not video.exists():
            print(f"ERROR: video not found: {video}", file=sys.stderr)
            return 2
        from vidsync import media

        runtime = media.probe_duration(video)
        runtime_src = f"probed from {video.name}"

    print("PUBLISH (Phase 6) - all timings from build/timings.csv")
    print(f"  runtime : {mmss(runtime)} ({runtime:.1f}s)  [{runtime_src}]")
    if args.video:
        print(f"  measured: {mmss(measured_end)} (render must be within 1 s - Gate 5)")
    print(f"  frames  : {len(rows)}")
    print()
    print("chapters (paste-ready; verify line by line against timings.csv, then polish titles):")
    last_chapter = float("-inf")
    for row in rows:
        start = float(row["hold_start_s"])
        if last_chapter != float("-inf") and start - last_chapter < args.min_gap:
            continue
        last_chapter = start
        print(f"  {mmss(start)}  {title_from_id(row['id'])}")
    print(f"  {mmss(runtime)}  end")
    print()
    print("next: thumbnail (legible at 336 px) -> title -> description; runtime quoted above.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
