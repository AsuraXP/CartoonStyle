"""SDXL-Lightning 8-step + IP-Adapter-Plus generator for B&W XKCD-style stick figures.

Usage (CLI, single-node multi-GPU the same way as generate_flux2.py):
    python -m cartoonstyle.generate_sdxl \
        --prompts path/to/visuals.csv \
        --out /kaggle/working/sdxl-out \
        --gpus 0

Model stack:
  * stabilityai/stable-diffusion-xl-base-1.0 (fp16)
  * ByteDance/SDXL-Lightning 8-step LoRA (fused)
  * h94/IP-Adapter Plus ViT-H for SDXL  (style+character lock from reference images)
  * EulerDiscreteScheduler, trailing timestep spacing
  * guidance_scale=1.5, num_inference_steps=8
  * 1024x576 generation -> lanczos 1920x1080 delivery

Memory: ~9-10 GB VRAM per GPU at 1024x576 (fits 2 workers on 2xT4 16GB).
Speed: ~4-6 sec/frame on T4 -> 73 frames in ~4-5 minutes on 2xT4.
"""
from __future__ import annotations

import argparse
import csv
import glob as _glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from diffusers import (
    StableDiffusionXLPipeline,
    EulerDiscreteScheduler,
    AutoencoderKL,
)
from transformers import CLIPVisionModelWithProjection

# ---------------------------------------------------------------------------
# Style & guard
# ---------------------------------------------------------------------------

STYLE_PREFIX = (
    "simple XKCD-style stick figure cartoon, pure black and white, thin black pen outline on a plain solid white background, "
    "single weight black line, round circle head, two big solid black oval eyes, simple curved line mouth, "
    "2 or 3 short hair tufts on top, vertical stick body, simple straight line arms and legs, small oval feet, "
    "mitten-shaped oval hands, no clothing, no color, no shading, no grey tones, no fill colors, no watercolor, "
    "no paper texture, no cream tint, lots of white space, minimal line drawing. "
)
SCENE_NEG = (
    "color, yellow grass, blue sky, red sunset, green plants, brown rocks, grey shading, watercolor, oil painting, "
    "3d render, photorealistic, photo, gradient, cross-hatching, filled shapes, beards, facial hair, mustache, "
    "robes, tunics, togas, loincloths, hoods, hats, helmets, crowns, turbans, armor, clothing, shirts, pants, "
    "clothes on stick figures, tridents, pitchforks, staffs, scepters, extra people, extra animals, "
    "extra objects not mentioned, blurry, deformed hands, extra fingers, extra limbs, mutated, disfigured, "
    "low quality, jpeg artifacts, signature, text, watermark, signature in corner"
)
SCENE_GUARD = (
    " Draw ONLY what is described. "
    "Every person is a simple stick figure with a round head, two black oval eyes, and stick limbs - no clothes, "
    "no beards, no spiky hair, no robes. Objects are simple black line outlines. White background only."
)

# Default reference images (ship with repo: 5 pure XKCD B&W stick-figure refs from user)
DEFAULT_REFS = ",".join(
    str(p) for p in sorted((Path(__file__).resolve().parent.parent / "lora_refs" / "stick_figure").glob("*.jpg"))
)

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(visual: str) -> tuple[str, str]:
    """Build the full positive + negative prompt for a scene."""
    # Strip a leading "Scene:" if visuals.csv has it, to avoid double-prefixing
    body = visual.strip()
    if body.lower().startswith("scene"):
        body = body.split(":", 1)[1].strip()
    pos = STYLE_PREFIX + body + "." + SCENE_GUARD
    return pos, SCENE_NEG


# ---------------------------------------------------------------------------
# CSV loading (same schema as generate_flux2.py: slide_id,moment,moment_text,visual_prompt)
# ---------------------------------------------------------------------------

def load_rows(path: str) -> list[tuple[str, str, str, str]]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            sid = str(r.get("slide_id", r.get("scene", ""))).strip()
            mom = str(r.get("moment", r.get("beat", ""))).strip()
            txt = r.get("moment_text", r.get("narration", r.get("text", ""))).strip()
            vis = r.get("visual_prompt", r.get("prompt", "")).strip()
            if not sid or not mom or not vis:
                continue
            rows.append((sid, mom, txt, vis))
    return rows


def beat_key(sid: str, mom: str) -> str:
    return f"beat{sid}m{mom}"


# ---------------------------------------------------------------------------
# Pipeline builder (one per worker process)
# ---------------------------------------------------------------------------

def build_pipeline(ref_paths: list[str], device: str = "cuda") -> tuple[StableDiffusionXLPipeline, list[Image.Image]]:
    """Load SDXL + Lightning 8-step LoRA + IP-Adapter Plus, return (pipe, ref_images)."""
    print(f"[{device}] loading SDXL base...", flush=True)
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        variant="fp16",
        use_safetensors=True,
    )

    # Replace scheduler for Lightning 8-step
    pipe.scheduler = EulerDiscreteScheduler.from_config(
        pipe.scheduler.config,
        timestep_spacing="trailing",
        prediction_type="epsilon",
    )

    print(f"[{device}] loading SDXL-Lightning 8-step LoRA...", flush=True)
    pipe.load_lora_weights(
        "ByteDance/SDXL-Lightning",
        weight_name="sdxl_lightning_8step_lora.safetensors",
    )
    pipe.fuse_lora(lora_scale=1.0)

    print(f"[{device}] loading IP-Adapter Plus...", flush=True)
    pipe.load_ip_adapter(
        "h94/IP-Adapter",
        subfolder="sdxl_models",
        weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
        image_encoder_folder="models/image_encoder",
    )
    pipe.set_ip_adapter_scale(0.65)  # strong style lock but leaves composition free

    # Memory tuning for T4 (16 GB) - SDXL+IP-Adapter fp16 is ~8 GB per worker.
    # With CUDA_VISIBLE_DEVICES properly pinning one worker per GPU, this fits
    # comfortably with VAE slicing + xformers.
    try:
        pipe.vae.enable_slicing()
    except Exception:
        try:
            pipe.enable_vae_slicing()  # older diffusers
        except Exception:
            pass
    xformers_ok = False
    try:
        pipe.enable_xformers_memory_efficient_attention()
        xformers_ok = True
        print(f"[{device}] xformers enabled", flush=True)
    except Exception:
        print(f"[{device}] xformers unavailable; falling back to attention slicing", flush=True)
    if not xformers_ok:
        try:
            pipe.enable_attention_slicing(slice_size="auto")
        except Exception:
            pass

    pipe = pipe.to(device)

    # Load + preprocess reference images into a SINGLE composite style sheet.
    # IP-Adapter expects one image per loaded adapter; to condition on multiple
    # refs without loading multiple copies of the adapter we tile them into one
    # grid image (3x2 or similar).
    print(f"[{device}] loading {len(ref_paths)} reference image(s)...", flush=True)
    ref_pil_raw = []
    for rp in ref_paths:
        img = Image.open(rp).convert("RGB")
        ref_pil_raw.append(img)
        print(f"[{device}]   ref: {rp} ({img.size})", flush=True)

    def make_style_sheet(imgs: list[Image.Image], cell: int = 512) -> Image.Image:
        """Tile reference images into a square grid, resize each cell to 'cell' px."""
        n = len(imgs)
        import math
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        sheet = Image.new("RGB", (cols * cell, rows * cell), (255, 255, 255))
        for i, im in enumerate(imgs):
            r, c = divmod(i, cols)
            # Fit into cell preserving aspect ratio, centered on white
            im2 = im.copy()
            im2.thumbnail((cell, cell), Image.Resampling.LANCZOS)
            x = c * cell + (cell - im2.width) // 2
            y = r * cell + (cell - im2.height) // 2
            sheet.paste(im2, (x, y))
        return sheet

    ref_sheet = make_style_sheet(ref_pil_raw, cell=512)
    print(f"[{device}] style sheet: {ref_sheet.size} ({len(ref_pil_raw)} refs tiled)", flush=True)
    ref_images = [ref_sheet]  # single image -> matches single IP-adapter

    return pipe, ref_images


# ---------------------------------------------------------------------------
# Worker (one per GPU, spawned via fork)
# ---------------------------------------------------------------------------

def run_worker(args: argparse.Namespace) -> None:
    # CUDA_VISIBLE_DEVICES is set by parent via env before this process starts.
    k, n = (int(x) for x in args.shard.split("/"))
    torch.cuda.set_device(0)
    device = "cuda"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = out_dir / f"manifest.shard{k}.jsonl"

    # Build pipeline
    ref_paths = [p for p in args.refs.split(",") if p.strip()]
    if not ref_paths:
        print(f"[{device}] WARNING: no reference images passed - IP-Adapter will have no style anchor", flush=True)
    pipe, ref_images = build_pipeline(ref_paths, device=device)

    # Determine which beats this shard owns
    all_rows = load_rows(args.prompts)
    my_rows = [r for i, r in enumerate(all_rows) if i % n == k]
    total = len(my_rows)

    # Resume: skip keys already in manifest
    done: set[str] = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if line.strip():
                try:
                    obj = json.loads(line)
                    done.add(obj["key"])
                except Exception:
                    pass
    print(f"[{device}] shard {k}/{n}: {total} beats total, {len(done)} already done, {total - len(done)} to render", flush=True)

    out_w, out_h = 1920, 1080
    t0 = time.time()

    with manifest.open("a") as mf:
        for idx, (sid, mom, txt, vis) in enumerate(my_rows):
            key = beat_key(sid, mom)
            if key in done:
                continue
            out_path = out_dir / f"{key}.png"

            prompt, neg = build_prompt(vis)

            # Deterministic seed per beat (different per shard but stable across runs)
            seed = args.seed + int(sid) * 100 + int(mom) + k * 10_000
            gen = torch.Generator(device=device).manual_seed(seed)

            try:
                result = pipe(
                    prompt=prompt,
                    negative_prompt=neg,
                    ip_adapter_image=ref_images if ref_images else None,
                    num_inference_steps=args.steps,
                    guidance_scale=args.guidance,
                    width=args.width,
                    height=args.height,
                    generator=gen,
                )
                img = result.images[0]
            except Exception as e:
                print(f"[{device}] ERROR rendering {key}: {e}", flush=True)
                continue

            # Lanczos upscale to 1920x1080
            if img.size != (out_w, out_h):
                img = img.resize((out_w, out_h), Image.Resampling.LANCZOS)
            img.save(out_path, "PNG", optimize=True)

            # Record in manifest
            rec = {
                "key": key,
                "slide_id": sid,
                "moment": mom,
                "image": str(out_path),
                "prompt": prompt,
                "seed": seed,
                "width": out_w,
                "height": out_h,
            }
            mf.write(json.dumps(rec) + "\n")
            mf.flush()
            done.add(key)

            elapsed = time.time() - t0
            done_count = len(done) - (sum(1 for _ in range(idx)) if False else 0)
            print(f"[{device}] {key} done ({len(done)}/{total}) [{elapsed:.0f}s elapsed]", flush=True)

    print(f"[{device}] shard done in {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------------------
# Main: spawn N workers, one per GPU
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SDXL-Lightning 8-step + IP-Adapter Plus batch renderer")
    p.add_argument("--prompts", required=True, help="CSV with visual_prompt column")
    p.add_argument("--refs", default=DEFAULT_REFS, help="comma-separated list of reference image paths for IP-Adapter")
    p.add_argument("--gpus", type=int, default=0, help="GPU count; 0 = auto-detect")
    p.add_argument("--out", default="/kaggle/working/sdxl-out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance", type=float, default=1.5,
                   help="CFG scale. 0 = pure distilled; 1.0-2.0 works for 8-step Lightning with neg prompt.")
    p.add_argument("--steps", type=int, default=8, help="Lightning step count (use 4 or 8 to match LoRA)")
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--height", type=int, default=576)
    p.add_argument("--shard", default="", help="internal: k/N worker split")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.shard:
        # Worker process
        run_worker(args)
        return

    # Parent: detect GPUs, spawn workers, merge manifests
    n_gpu = torch.cuda.device_count()
    if args.gpus <= 0:
        args.gpus = max(1, n_gpu)
    print(f"[main] using {args.gpus} drawing worker(s) across {n_gpu} visible GPU(s)", flush=True)
    print(f"[main] {args.width}x{args.height} -> lanczos 1920x1080, {args.steps} steps, CFG={args.guidance}", flush=True)

    procs: list[subprocess.Popen] = []
    for k in range(args.gpus):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(k)
        env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        cmd = [
            sys.executable, "-m", "cartoonstyle.generate_sdxl",
            "--prompts", args.prompts,
            "--refs", args.refs,
            "--out", args.out,
            "--seed", str(args.seed),
            "--guidance", str(args.guidance),
            "--steps", str(args.steps),
            "--width", str(args.width),
            "--height", str(args.height),
            "--shard", f"{k}/{args.gpus}",
        ]
        print(f"[main] spawning worker {k}: bound to cuda:{k}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env))

    fail = 0
    for pr in procs:
        rc = pr.wait()
        if rc != 0:
            fail += 1

    # Merge shard manifests into one manifest.jsonl (same contract as generate_flux2.py)
    out_dir = Path(args.out)
    shards = sorted(_glob.glob(str(out_dir / "manifest.shard*.jsonl")))
    merged = out_dir / "manifest.jsonl"
    seen: set[str] = set()
    n_written = 0
    with merged.open("w") as out:
        for sp in shards:
            for line in open(sp):
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                    if obj["key"] in seen:
                        continue
                    seen.add(obj["key"])
                    out.write(line)
                    n_written += 1
                except Exception:
                    pass
    print(f"[main] merged {n_written} records -> {merged}", flush=True)
    if fail:
        print(f"[main] {fail} worker(s) failed - progress merged; re-run to resume", flush=True)
        sys.exit(1)
    print("[main] all workers done", flush=True)


if __name__ == "__main__":
    main()
