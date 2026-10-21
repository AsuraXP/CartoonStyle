"""Upscale generated images to 1920x1080 (YouTube landscape)."""

from __future__ import annotations

import argparse
import os

from PIL import Image, ImageFilter

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


def upscale_image(
    img: Image.Image,
    size: tuple[int, int] = (1920, 1080),
    method: str = "lanczos",
    sharpen: bool = True,
) -> Image.Image:
    if img.size == size:
        return img
    if method == "esrgan":
        try:
            img = _esrgan_upscale(img, size)
        except Exception as e:  # noqa: BLE001 - fallback keeps the run alive
            print(f"[upscale] esrgan unavailable ({e}); falling back to lanczos", flush=True)
            method = "lanczos"
    if method != "esrgan":
        img = img.resize(size, Image.LANCZOS)
        if sharpen:
            img = img.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))
    return img


_ESRGAN_MODEL = None


def _esrgan_upscale(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Optional 4x ESRGAN via spandrel + a local .pth (pass via CARTOONSTYLE_ESRGAN)."""
    global _ESRGAN_MODEL
    import numpy as np
    import torch

    weights = os.environ.get("CARTOONSTYLE_ESRGAN")
    if not weights or not os.path.exists(weights):
        raise RuntimeError("set CARTOONSTYLE_ESRGAN to a 4x .pth upscaler to use esrgan")
    if _ESRGAN_MODEL is None:
        from spandrel import ImageModelDescriptor, ModelLoader

        _ESRGAN_MODEL = ModelLoader().load_from_file(weights)
        assert isinstance(_ESRGAN_MODEL, ImageModelDescriptor)
        _ESRGAN_MODEL.eval().cuda()
    t = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)[None].cuda()
    with torch.no_grad():
        out = _ESRGAN_MODEL(t)[0].clamp(0, 1)
    big = Image.fromarray((out.permute(1, 2, 0).cpu().numpy() * 255).astype("uint8"))
    return big.resize(size, Image.LANCZOS)


def main() -> None:
    p = argparse.ArgumentParser(description="Upscale a folder of images to 1920x1080.")
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--width", type=int, default=1920)
    p.add_argument("--height", type=int, default=1080)
    p.add_argument("--method", default="lanczos", choices=["lanczos", "esrgan"])
    p.add_argument("--no-sharpen", action="store_true")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    files = [f for f in sorted(os.listdir(args.in_dir)) if os.path.splitext(f)[1].lower() in IMAGE_EXT]
    for i, name in enumerate(files, 1):
        img = Image.open(os.path.join(args.in_dir, name)).convert("RGB")
        img = upscale_image(img, (args.width, args.height), args.method, not args.no_sharpen)
        img.save(os.path.join(args.out_dir, name))
        print(f"[{i}/{len(files)}] {name} -> {img.size[0]}x{img.size[1]}")


if __name__ == "__main__":
    main()
