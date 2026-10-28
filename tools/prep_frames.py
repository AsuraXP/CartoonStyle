#!/usr/bin/env python3
"""Phase 3 - frame preparation and contact sheet (WORKSPACE ONLY).

    python tools/prep_frames.py --repo . --sheet --strict

Reads frames.json (the only source of truth for frame count), finds each
frame's generated source image in assets/frames/source/, compresses it to a
reviewed 1280x720 JPG in assets/frames/ (the only frame artifacts that go
into Git), and optionally builds one contact sheet for end-to-end review.

--strict: every frames.json entry must have a source image and a produced
JPG, otherwise exit non-zero listing what is missing.  Never generate a
frame here - generation happens from anchors via the image tool, with visual
QC before anything is pushed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_DEFAULT))

from vidsync.anchors import AnchorError, load_frames_json  # noqa: E402
from vidsync.config import load_config  # noqa: E402

SOURCE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def find_source(source_dir: Path, frame_id: str) -> Path | None:
    for ext in SOURCE_EXTS:
        p = source_dir / f"{frame_id}{ext}"
        if p.exists():
            return p
    # Tolerate id prefixes like NN_id
    candidates = sorted(source_dir.glob(f"*{frame_id}{ext}")) if source_dir.exists() else []
    for ext in SOURCE_EXTS:
        candidates = sorted(source_dir.glob(f"*{frame_id}{ext}"))
        if candidates:
            return candidates[0]
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=str(REPO_ROOT_DEFAULT))
    ap.add_argument("--sheet", action="store_true", help="build assets/contact_sheet.jpg")
    ap.add_argument("--strict", action="store_true", help="fail if any frame is missing")
    args = ap.parse_args()

    from PIL import Image, ImageDraw, ImageFont

    repo_root = Path(args.repo).resolve()
    cfg = load_config(repo_root)
    try:
        frames = load_frames_json(repo_root / cfg["paths"]["frames_json"])
    except AnchorError as e:
        print(f"PREP ERROR: {e}", file=sys.stderr)
        return 2

    width = int(cfg["video"]["width"])
    height = int(cfg["video"]["height"])
    frames_dir = repo_root / cfg["frames"]["dir"]
    source_dir = repo_root / cfg["frames"]["source_dir"]
    frames_dir.mkdir(parents=True, exist_ok=True)

    missing_sources = []
    outputs = []
    for f in frames:
        src = find_source(source_dir, f.id)
        if src is None:
            missing_sources.append(f.id)
            print(f"MISSING source image for id '{f.id}' (expected in {source_dir})")
            continue
        out = frames_dir / f"{f.id}.jpg"
        with Image.open(src) as im:
            im = im.convert("RGB")
            if im.width / im.height != width / height:
                print(
                    f"note: {src.name} aspect {im.width}x{im.height} != {width}:{height}; "
                    "center-cropping to fit"
                )
            from PIL import ImageOps

            im = ImageOps.fit(im, (width, height), method=Image.LANCZOS)
            im.save(out, "JPEG", quality=85, optimize=True)
        kb = out.stat().st_size / 1024
        print(f"prepared {out.relative_to(repo_root)}  {kb:.0f} KB  (from {src.name})")
        outputs.append(out)

    total = sum(p.stat().st_size for p in frames_dir.glob("*.jpg"))
    print(
        f"\nframes: {len(outputs)}/{len(frames)} prepared; "
        f"total {total / (1024 * 1024):.2f} MB in {frames_dir.relative_to(repo_root)}"
    )
    if total > 5 * 1024 * 1024:
        print("WARNING: frame weight exceeds the 5 MB repo target - raise compression")

    if args.sheet and outputs:
        thumb_w, thumb_h = width // 3, height // 3
        label_h = 22
        cols = 3
        rows = (len(outputs) + cols - 1) // cols
        sheet = Image.new("RGB", (cols * thumb_w, rows * (thumb_h + label_h)), (24, 24, 24))
        draw = ImageDraw.Draw(sheet)
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        for i, out in enumerate(outputs):
            x = (i % cols) * thumb_w
            y = (i // cols) * (thumb_h + label_h)
            with Image.open(out) as im:
                sheet.paste(im.resize((thumb_w, thumb_h), Image.LANCZOS), (x, y))
            draw.text((x + 4, y + thumb_h + 4), frames[i].id, fill=(230, 230, 230), font=font)
        sheet_path = repo_root / "assets" / "contact_sheet.jpg"
        sheet_path.parent.mkdir(parents=True, exist_ok=True)
        sheet.save(sheet_path, "JPEG", quality=80, optimize=True)
        print(f"contact sheet       : {sheet_path.relative_to(repo_root)}")

    if missing_sources:
        print(f"\n{len(missing_sources)} frame(s) missing source images", file=sys.stderr)
        if args.strict:
            print("STRICT MODE: failing.", file=sys.stderr)
            return 1
    if args.strict and len(outputs) != len(frames):
        print("STRICT MODE: not every frames.json entry produced a JPG.", file=sys.stderr)
        return 1
    print("prep_frames: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
