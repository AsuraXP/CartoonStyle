"""Tile generated images into one labeled comparison sheet for quick judging."""
from __future__ import annotations

import argparse
import glob
import os

from PIL import Image, ImageDraw


def main() -> None:
    p = argparse.ArgumentParser(description="labeled contact sheet of generated images")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out", default="comparison.jpg")
    p.add_argument("--cols", type=int, default=3)
    p.add_argument("--thumb", type=int, default=640, help="thumbnail width in px")
    args = p.parse_args()

    files = sorted(
        glob.glob(os.path.join(args.images_dir, "*.png"))
        + glob.glob(os.path.join(args.images_dir, "*.jpg"))
    )
    if not files:
        raise SystemExit(f"no images in {args.images_dir}")

    cols = max(1, min(args.cols, len(files)))
    rows = (len(files) + cols - 1) // cols
    tw = args.thumb
    th = int(tw * 9 / 16)
    label_h = 26
    sheet = Image.new("RGB", (cols * tw, rows * (th + label_h)), "white")
    draw = ImageDraw.Draw(sheet)

    for i, f in enumerate(files):
        im = Image.open(f).convert("RGB")
        im.thumbnail((tw, th))
        x = (i % cols) * tw
        y = (i // cols) * (th + label_h)
        sheet.paste(im, (x, y))
        draw.text((x + 6, y + th + 5), os.path.basename(f), fill="black")

    sheet.save(args.out, quality=88)
    print(f"[sheet] {len(files)} images -> {args.out}")


if __name__ == "__main__":
    main()
