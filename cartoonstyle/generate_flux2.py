"""FLUX.2 [klein] 4B production generator — thin-pen stickman on crumpled paper.

Winning recipe (locked by user 2026-10-16):
  * multi-reference character anchors (scene-free images only — no furniture to leak)
  * scene-first prompt + style block
  * guidance 2.0, 4 steps, 1344x768 -> lanczos 1920x1080

Durability: per-shard manifests, instant local saves, background zip + Kaggle
dataset push, resume by manifest. Same contract as generate.py.
"""
from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFS = [
    str(REPO_ROOT / "lora_refs/thin_pen/style_ref_character.png"),
    str(REPO_ROOT / "lora_refs/thin_pen/ref_stock_crowd.jpg"),
    str(REPO_ROOT / "lora_refs/thin_pen/ref_stock_multitask.jpg"),
]

STYLE_BLOCK = (
    "Drawn as a minimalist hand-drawn stickman cartoon with thin clean ballpoint pen lines, "
    "no thick outlines, light flat colors, on a plain smooth solid off-white background - "
    "no paper texture, no crumpling. "
    "Every stickman has a round white head with a few short hair strands, "
    "two small oval eyes with black pupils, and a smile. "
    "Correct anatomy on every figure: exactly two arms and two legs, two hands, one head - "
    "no third arm, no extra limbs, no extra hands, no extra fingers. "
    "Arms and legs are thin single pen strokes - simple stick limbs, straight or gently bent, "
    "never muscular, never bent like animal legs, never hooves. "
    "The character is large in the frame so the simple linework reads clearly. "
    "Every character and object appears at its natural real-world size relative to the people - "
    "a lion is full-grown and huge, never a toy, miniature, or floating object. "
    "The scene is filled with the specific objects and details described, not empty or bland. "
    "Use the same character design as the reference image, but this is a completely new scene."
)
SCENE_GUARD = (
    "Only objects that belong in this scene - no toys, no miniatures, no floating objects, "
    "no third arm, no extra limbs, no extra hands, "
    "no whiteboard, no pointer stick, no extra furniture, no text or lettering unless asked."
)


def build_prompt(visual: str) -> str:
    v = visual.strip().rstrip(".,")
    v = v.replace("cartoon stickman", "stickman")
    if not v.lower().startswith("scene"):
        v = f"Scene: {v[0].lower() + v[1:] if v else v}"
    return f"{v}. {SCENE_GUARD} {STYLE_BLOCK}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLUX.2 Klein multi-ref batch generator")
    p.add_argument("--prompts", required=True, help="CSV with visual_prompt (or prompt) column")
    p.add_argument("--images", type=int, default=12, help="total images to make")
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--out", default="/kaggle/working/cartoonstyle-out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--refs", default=",".join(DEFAULT_REFS))
    p.add_argument("--offload", choices=["none", "model", "sequential"], default="model",
                   help="none = full model on GPU (RAM-light, needs ~13GB VRAM)")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--resume-from", default="")
    p.add_argument("--shard", default="", help="internal: k/N worker split")
    return p.parse_args()


def load_rows(path: str) -> list[tuple[str, str]]:
    """Returns (slide_id, full_prompt) per row - slide_id = narration beat number."""
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for i, r in enumerate(rows, 1):
        sid = str(r.get("slide_id") or i).strip().zfill(2)
        mom = str(r.get("moment") or "").strip()
        prefix = f"beat{sid}m{mom.zfill(2)}" if mom else f"beat{sid}"
        if r.get("prompt"):
            out.append((prefix, r["prompt"].strip()))
        elif r.get("visual_prompt") and r["visual_prompt"].strip():
            out.append((prefix, build_prompt(r["visual_prompt"])))
        else:
            raise SystemExit(f"row {i}: no prompt/visual_prompt text in {path}")
    return out


def run_worker(args: argparse.Namespace) -> None:
    import gc

    import torch

    from . import kaggle_io

    kaggle_io.purge_bad_torchao()
    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    prompts = load_rows(args.prompts)
    k, n = (int(x) for x in args.shard.split("/"))
    n_rows = len(prompts)
    jobs = []
    for i in range(args.images):
        prefix, prompt = prompts[i % n_rows]
        variant = i // n_rows
        seed = args.seed + i
        key = f"{prefix}_v{variant:02d}_s{seed:08x}"
        jobs.append((key, prompt, seed))
    jobs = jobs[k::n]
    print(f"[gpu {k}] {len(jobs)} jobs of {args.images} total", flush=True)

    from diffusers import Flux2KleinPipeline

    refs = []
    for p in args.refs.split(","):
        if p.strip():
            im = Image.open(p.strip()).convert("RGB")
            im.thumbnail((512, 512))
            refs.append(im)
    print(f"[gpu {k}] {len(refs)} character anchors", flush=True)

    def build(offload: str):
        pipe = Flux2KleinPipeline.from_pretrained(
            "black-forest-labs/FLUX.2-klein-4B", torch_dtype=torch.float16
        )
        if offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif offload == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")  # weights stay in VRAM -> host RAM stays free for the other worker
        return pipe

    # stagger loads so two workers never eat host RAM / bandwidth at once
    delay = k * 45
    if delay:
        print(f"[gpu {k}] staggered start: waiting {delay}s", flush=True)
        time.sleep(delay)
    try:
        pipe = build(args.offload)
    except torch.cuda.OutOfMemoryError:
        print(f"[gpu {k}] offload={args.offload} does not fit -> falling back", flush=True)
        gc.collect(); torch.cuda.empty_cache()
        order = {"none": ["model", "sequential"], "model": ["sequential"], "sequential": []}
        pipe = None
        for alt in order[args.offload]:
            try:
                pipe = build(alt)
                print(f"[gpu {k}] running with offload={alt}", flush=True)
                break
            except torch.cuda.OutOfMemoryError:
                gc.collect(); torch.cuda.empty_cache()
        if pipe is None:
            raise
        args.offload = alt  # report the mode we actually run
    print(f"[gpu {k}] flux2 klein ready ({args.offload} offload)", flush=True)

    manifest = os.path.join(args.out, f"manifest.shard{k}.jsonl")
    import glob as _glob
    all_shards = sorted(_glob.glob(os.path.join(args.out, "manifest.shard*.jsonl")))
    done = kaggle_io.read_done(all_shards + [os.path.join(args.out, "manifest.jsonl")])
    print(f"[gpu {k}] resume: {len(done)} images already done", flush=True)
    if args.resume_from:
        done.update(kaggle_io.read_done([os.path.join(args.resume_from, "manifest.jsonl")]))

    out_w, out_h = 1920, 1080  # delivery format: 1344x768 -> lanczos upscale
    pusher = None
    if not args.no_push:
        pusher = kaggle_io.Pusher(args.out, every_images=25, every_seconds=600).start()

    for key, prompt, seed in jobs:
        rel = f"images/{key}.png"
        if key in done:
            continue
        t0 = time.time()
        try:
            img = pipe(
                prompt=prompt,
                image=refs,
                width=args.width,
                height=args.height,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
        except torch.cuda.OutOfMemoryError:
            print(f"[gpu {k}] CUDA OOM -> switching to sequential offload", flush=True)
            del pipe
            gc.collect()
            torch.cuda.empty_cache()
            pipe = build("sequential")
            img = pipe(
                prompt=prompt,
                image=refs,
                width=args.width,
                height=args.height,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
        if (img.width, img.height) != (out_w, out_h):
            img = img.resize((out_w, out_h), Image.LANCZOS)
        img.save(os.path.join(args.out, rel))
        dt = time.time() - t0
        kaggle_io.append_manifest(manifest, {"id": key, "seed": seed, "prompt": prompt, "sec": round(dt, 1)})
        print(f"[gpu {k}] {key} {dt:.1f}s", flush=True)

    if pusher:
        pusher.stop()
        pusher.join(120)
    print(f"[gpu {k}] shard done", flush=True)


def main() -> None:
    args = parse_args()
    if args.shard:
        run_worker(args)
        return

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    print(f"[main] {args.images} images | seed {args.seed} | out {args.out}", flush=True)

    procs = []
    for k in range(args.gpus):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(k))
        cmd = [sys.executable, "-m", "cartoonstyle.generate_flux2",
               "--prompts", args.prompts, "--images", str(args.images),
               "--out", args.out, "--seed", str(args.seed),
               "--guidance", str(args.guidance), "--steps", str(args.steps),
               "--width", str(args.width), "--height", str(args.height),
               "--refs", args.refs, "--offload", args.offload,
               "--shard", f"{k}/{args.gpus}"]
        if args.no_push:
            cmd.append("--no-push")
        procs.append((k, subprocess.Popen(cmd, env=env)))
    fail = 0
    for k, p in procs:
        rc = p.wait()
        if rc:
            fail += 1
            print(f"[main] worker {k} exited with code {rc}", flush=True)

    import glob as _glob

    from . import kaggle_io

    shards = sorted(_glob.glob(os.path.join(args.out, "manifest.shard*.jsonl")))
    merged = kaggle_io.merge_manifests(args.out, shards)  # always, even after failures
    if not fail:
        kaggle_io.update_zip(args.out, kaggle_io.list_new_images(args.out))
        if not args.no_push:
            kaggle_io.kaggle_push(args.out, "flux2 run complete")
        print(f"[main] done: {args.images} images | manifest {merged}", flush=True)
    else:
        print(f"[main] {fail} worker(s) failed - progress merged to {merged}; re-run to resume", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
