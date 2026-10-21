"""Auto-caption a folder of style reference images with BLIP.

Writes/updates sidecar .txt files used by cartoonstyle.train_lora.
Always hand-edit the captions afterwards for a style LoRA — describe the
character, line style, palette and scene explicitly.

    python -m cartoonstyle.caption --data-dir refs --prefix "ckstyle, flat cartoon style"
"""

from __future__ import annotations

import argparse
import os

from PIL import Image

from .upscale import IMAGE_EXT


def main():
    p = argparse.ArgumentParser(description="CartoonStyle BLIP auto-captioner")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--model", default="Salesforce/blip-image-captioning-large")
    p.add_argument("--prefix", default="", help="prepended to every caption (trigger + style)")
    p.add_argument("--suffix", default="", help="appended to every caption")
    p.add_argument("--max-new-tokens", type=int, default=40)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    import torch
    from transformers import BlipForConditionalGeneration, BlipProcessor

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[caption] loading {args.model} on {device}", flush=True)
    processor = BlipProcessor.from_pretrained(args.model)
    model = BlipForConditionalGeneration.from_pretrained(args.model, torch_dtype=torch.float16).to(device)

    names = [n for n in sorted(os.listdir(args.data_dir)) if os.path.splitext(n)[1].lower() in IMAGE_EXT]
    written = 0
    for i, name in enumerate(names, 1):
        stem = os.path.splitext(name)[0]
        cap_path = os.path.join(args.data_dir, stem + ".txt")
        if os.path.exists(cap_path) and not args.overwrite:
            print(f"[{i}/{len(names)}] {name} (caption exists, skip)", flush=True)
            continue
        img = Image.open(os.path.join(args.data_dir, name)).convert("RGB")
        inputs = processor(img, return_tensors="pt").to(device, torch.float16)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
        caption = processor.decode(out[0], skip_special_tokens=True).strip()
        parts = [args.prefix.strip(), caption, args.suffix.strip()]
        full = ", ".join(x for x in parts if x)
        with open(cap_path, "w", encoding="utf-8") as f:
            f.write(full + "\n")
        written += 1
        print(f"[{i}/{len(names)}] {name}: {full}", flush=True)

    print(f"[caption] done — {written} captions written to {args.data_dir}", flush=True)
    print("[caption] REVIEW THEM by hand — a style/character LoRA is only as good as its captions.", flush=True)


if __name__ == "__main__":
    main()
