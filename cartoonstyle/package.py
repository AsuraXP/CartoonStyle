"""Package a finished video + thumbnail + metadata into a single deliverable zip."""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="path to VIDEO.mp4")
    p.add_argument("--thumbnail", required=True, help="path to thumbnail image")
    p.add_argument("--metadata", required=True, help="metadata.json with title/description/tags")
    p.add_argument("--out", required=True, help="output zip path")
    p.add_argument("--script", default="", help="optional script.md to include")
    args = p.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    meta = json.load(open(args.metadata, encoding="utf-8"))

    # Also write a human-readable youtube_upload.txt so the user can copy-paste
    readme = []
    readme.append(f"TITLE:\n{meta['title']}\n")
    readme.append(f"DESCRIPTION:\n{meta['description']}\n")
    readme.append("TAGS (comma separated):\n" + ", ".join(meta["tags"]) + "\n")

    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(args.video, "VIDEO.mp4")
        z.write(args.thumbnail, "thumbnail.jpg")
        z.write(args.metadata, "metadata.json")
        if args.script and Path(args.script).is_file():
            z.write(args.script, "script.md")
        z.writestr("youtube_upload.txt", "\n".join(readme))
    print(f"[package] wrote {out} ({out.stat().st_size / (1024*1024):.1f} MB)")
    print(f"  - VIDEO.mp4")
    print(f"  - thumbnail.jpg")
    print(f"  - metadata.json (title/description/tags)")
    print(f"  - youtube_upload.txt (copy-paste ready)")
    if args.script:
        print(f"  - script.md")


if __name__ == "__main__":
    main()
