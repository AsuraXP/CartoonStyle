"""Multi-GPU batch generation with SDXL-Lightning + style LoRA.

Designed to saturate Kaggle T4 x2:
- one worker process per GPU (true 2x throughput, no DataParallel),
- resumable deterministic job queue (crash -> re-run continues),
- every image saved the moment it exists,
- a background pusher zips new images and pushes to a Kaggle Dataset.

Usage:
    python -m cartoonstyle.generate --config config/default.yaml --images 500 --gpus 2
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import time

from . import kaggle_io
from .config import apply_overrides, load_config, save_config
from .prompts import build_jobs, job_key
from .upscale import upscale_image


def parse_args():
    p = argparse.ArgumentParser(description="CartoonStyle batch generator")
    p.add_argument("--config", default=None)
    p.add_argument("--out", default=None, help="output dir (default: output.dir from config)")
    p.add_argument("--images", type=int, default=None)
    p.add_argument("--prompts", default=None, help="override prompts.file (e.g. storyboard prompts.csv)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--gpus", type=int, default=None, help="worker processes = GPUs (default: all)")
    p.add_argument("--base", default=None, help="override model.base")
    p.add_argument("--lora", default=None, help="override model.lora (style/character LoRA)")
    p.add_argument("--lora-scale", type=float, default=None)
    p.add_argument("--steps", type=int, default=None, help="override generation.steps")
    p.add_argument("--lightning-steps", type=int, default=None, choices=[2, 4, 8])
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--resume-from", action="append", default=[], help="extra dir(s) with prior manifest/images")
    p.add_argument("--no-push", action="store_true", help="disable Kaggle Dataset auto-push")
    p.add_argument("--lowvram", action="store_true")
    p.add_argument("--shard", default=None, help="internal: i/n")
    return p.parse_args()


def load_style_lora(pipe, style_lora: str, scale: float) -> str:
    """Load a style LoRA in any known format. Returns 'std' or 'peft'."""

    def _peft_merge(peft_dir: str) -> None:
        from peft import PeftModel

        unet = PeftModel.from_pretrained(pipe.unet, peft_dir)
        if scale != 1.0:
            for module in unet.modules():
                lora_B = getattr(module, "lora_B", None)
                if lora_B is not None:
                    for k in lora_B:
                        lora_B[k].weight.data *= scale
        pipe.unet = unet.merge_and_unload()

    if os.path.isdir(style_lora):
        std = os.path.join(style_lora, "pytorch_lora_weights.safetensors")
        if os.path.exists(std):
            try:
                pipe.load_lora_weights(style_lora, adapter_name="style")
                pipe.set_adapters(["style"], adapter_weights=[scale])
                return "std"
            except Exception as e:  # noqa: BLE001
                print(f"[worker] diffusers-format load failed ({e}); trying peft format", flush=True)
        for cand in (style_lora, os.path.join(style_lora, "peft")):
            if os.path.exists(os.path.join(cand, "adapter_config.json")):
                _peft_merge(cand)
                return "peft"
        raise SystemExit(
            f"No recognizable LoRA in {style_lora} "
            "(expected pytorch_lora_weights.safetensors or peft/adapter_config.json)")
    pipe.load_lora_weights(os.path.dirname(os.path.abspath(style_lora)) or ".",
                           weight_name=os.path.basename(style_lora), adapter_name="style")
    pipe.set_adapters(["style"], adapter_weights=[scale])
    return "std"


def build_pipeline(cfg: dict, lowvram: bool = False):
    import torch
    from diffusers import EulerAncestralDiscreteScheduler, EulerDiscreteScheduler, StableDiffusionXLPipeline
    from huggingface_hub import hf_hub_download

    base = cfg["model"]["base"]
    print(f"[worker] loading base model: {base}", flush=True)
    try:
        pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16, variant="fp16")
    except Exception:
        pipe = StableDiffusionXLPipeline.from_pretrained(base, torch_dtype=torch.float16)

    n = int(cfg["model"].get("lightning_steps") or 0)
    have_lightning = n > 0
    if have_lightning:
        weight_name = f"sdxl_lightning_{n}step_lora.safetensors"
        lightning_path = hf_hub_download(cfg["model"].get("lightning_repo", "ByteDance/SDXL-Lightning"), weight_name)
        print(f"[worker] applying SDXL-Lightning {n}-step LoRA", flush=True)
        pipe.load_lora_weights(lightning_path, adapter_name="lightning")
    else:
        print("[worker] native sampling (no Lightning distill)", flush=True)

    style_lora = cfg["model"].get("lora")
    style_scale = float(cfg["model"].get("lora_scale", 0.85))
    fused = False
    if style_lora:
        if have_lightning:
            try:
                # ByteDance recipe: fuse lightning into base first, then hang the style LoRA on top.
                pipe.fuse_lora(lora_scale=1.0)
                fused = True
            except Exception as e:  # noqa: BLE001
                print(f"[worker] fuse_lora failed ({e}); using dual adapters", flush=True)
        fmt = load_style_lora(pipe, style_lora, style_scale)
        if fmt == "std" and have_lightning and not fused:
            pipe.set_adapters(["lightning", "style"], adapter_weights=[1.0, style_scale])
        print(f"[worker] style LoRA loaded ({fmt}): {style_lora} @ {style_scale}", flush=True)
    elif have_lightning:
        try:
            pipe.fuse_lora(lora_scale=1.0)
        except Exception:  # noqa: BLE001
            pass

    sched = (cfg["generation"].get("scheduler") or "euler").lower()
    if sched in ("euler_a", "euler_ancestral"):
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
    elif have_lightning:
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
    else:
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to("cuda")
    try:
        pipe.enable_vae_slicing()
    except AttributeError:  # diffusers 0.41+: moved to the VAE itself
        pipe.vae.enable_slicing()
    if lowvram:
        try:
            pipe.enable_attention_slicing("auto")
        except AttributeError:
            pass
    return pipe


def run_worker(cfg: dict, out_dir: str, shard_id: int, num_shards: int, lowvram: bool, resume_from: list[str]):
    import torch

    gen_cfg = cfg["generation"]
    out_cfg = cfg["output"]
    width, height = int(gen_cfg["width"]), int(gen_cfg["height"])
    steps = int(gen_cfg["steps"])
    ln = int(cfg["model"].get("lightning_steps") or 0)
    if ln > 0:
        steps = ln  # distill LoRAs are step-count-specific
    guidance = float(gen_cfg.get("guidance_scale", 0.0))
    batch = max(1, int(gen_cfg.get("batch", 2)))
    negative_default = gen_cfg.get("negative") or ""
    target = tuple(out_cfg.get("upscale_to") or [1920, 1080])
    upscale_method = out_cfg.get("upscale_method", "lanczos")
    model_tag = f"{cfg['model']['base']}|{cfg['model'].get('lora')}|ln{cfg['model'].get('lightning_steps', 4)}"

    total = int(cfg["run"]["images"])
    base_seed = int(cfg["run"]["seed"])
    jobs = build_jobs(cfg, total, base_seed)
    for j in jobs:
        if j.get("seed") is None:
            j["seed"] = base_seed + j["index"]
        j["key"] = job_key(
            j["index"], j["prompt"], j.get("negative") or "", j["seed"],
            width, height, steps, guidance, model_tag, cfg["model"].get("lora"),
        )

    manifest_paths = [os.path.join(out_dir, "manifest.jsonl")]
    manifest_paths += [os.path.join(out_dir, f"manifest.w{i}.jsonl") for i in range(num_shards)]
    for rdir in resume_from:
        manifest_paths.append(os.path.join(rdir, "manifest.jsonl"))
    done = kaggle_io.read_done(manifest_paths)
    shard_jobs = [j for j in jobs[shard_id::num_shards] if j["key"] not in done]
    print(f"[gpu {shard_id}] {len(shard_jobs)} pending of {len(jobs[shard_id::num_shards])} in shard", flush=True)
    if not shard_jobs:
        return

    pipe = build_pipeline(cfg, lowvram=lowvram)
    manifest_path = os.path.join(out_dir, f"manifest.w{shard_id}.jsonl")
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    t0 = time.time()
    done_count = 0
    for bi in range(0, len(shard_jobs), batch):
        chunk = shard_jobs[bi:bi + batch]
        prompts = [j["prompt"] for j in chunk]
        negatives = [j.get("negative") or negative_default for j in chunk]
        generators = [torch.Generator(device="cuda").manual_seed(int(j["seed"])) for j in chunk]
        t_gen = time.time()
        result = pipe(
            prompt=prompts,
            negative_prompt=negatives if any(negatives) else None,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=generators,
            output_type="pil",
        )
        per_image = (time.time() - t_gen) / len(chunk)
        for j, img in zip(chunk, result.images):
            img = upscale_image(img, target, method=upscale_method)
            rel = os.path.join("images", f"{j['index']:06d}_{j['key'][:8]}.png")
            img.save(os.path.join(out_dir, rel))
            kaggle_io.append_manifest(manifest_path, {
                "key": j["key"], "index": j["index"], "file": rel,
                "prompt": j["prompt"], "negative": j.get("negative") or "",
                "seed": j["seed"], "width": target[0], "height": target[1],
                "native": [width, height], "steps": steps, "guidance": guidance,
                "model": cfg["model"]["base"], "lora": cfg["model"].get("lora"),
                "seconds": round(per_image, 2), "gpu": shard_id,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            })
        done_count += len(chunk)
        elapsed = time.time() - t0
        rate = done_count / elapsed * 60
        remaining = (len(shard_jobs) - done_count) / max(rate / 60, 1e-6) / 60
        print(
            f"[gpu {shard_id}] {done_count}/{len(shard_jobs)} | {per_image:.1f}s/img | "
            f"{rate:.1f} img/min | eta {remaining:.0f}m",
            flush=True,
        )
        if bi % (batch * 10) == 0:
            gc.collect()
            torch.cuda.empty_cache()

    print(f"[gpu {shard_id}] shard complete in {(time.time() - t0) / 60:.1f}m", flush=True)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    apply_overrides(cfg, {
        "run.images": args.images,
        "run.seed": args.seed,
        "prompts.file": args.prompts,
        "model.base": args.base,
        "model.lora": args.lora,
        "model.lora_scale": args.lora_scale,
        "model.lightning_steps": args.lightning_steps,
        "generation.steps": args.steps,
        "generation.batch": args.batch,
    })
    out_dir = args.out or cfg["output"].get("dir") or "/kaggle/working/cartoonstyle-out"
    cfg["output"]["dir"] = out_dir

    if args.shard:  # worker process
        shard_id, num_shards = (int(x) for x in args.shard.split("/"))
        run_worker(cfg, out_dir, shard_id, num_shards, args.lowvram, args.resume_from)
        return

    # ---- orchestrator ----
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    save_config(cfg, os.path.join(out_dir, "run_config.yaml"))
    total = int(cfg["run"]["images"])
    base_seed = int(cfg["run"]["seed"])
    jobs = build_jobs(cfg, total, base_seed)  # validate determinism early
    print(f"[main] {total} jobs | seed {base_seed} | out {out_dir}", flush=True)

    import torch
    n_gpus = args.gpus or torch.cuda.device_count() or 1
    n_gpus = max(1, min(n_gpus, 8))

    pusher = None
    if not args.no_push:
        out_cfg = cfg["output"]
        pusher = kaggle_io.Pusher(
            out_dir,
            every_images=int(out_cfg.get("push_every_images", 50)),
            every_seconds=int(out_cfg.get("push_every_seconds", 600)),
            enable_kaggle=bool(out_cfg.get("dataset_slug") or out_cfg.get("auto_kaggle_push", True)),
        ).start()

    t0 = time.time()
    if n_gpus == 1:
        run_worker(cfg, out_dir, 0, 1, args.lowvram, args.resume_from)
    else:
        print(f"[main] launching {n_gpus} workers (one per GPU)", flush=True)
        procs = []
        for i in range(n_gpus):
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(i)
            cmd = [
                sys.executable, "-m", "cartoonstyle.generate",
                "--out", out_dir,
                "--shard", f"{i}/{n_gpus}",
            ]
            if args.config:
                cmd += ["--config", args.config]
            if args.lowvram:
                cmd.append("--lowvram")
            for rdir in args.resume_from:
                cmd += ["--resume-from", rdir]
            # pass resolved run params so workers rebuild the identical job list
            cmd += ["--images", str(total), "--seed", str(base_seed)]
            if args.base:
                cmd += ["--base", args.base]
            if args.lora:
                cmd += ["--lora", args.lora]
            if args.lora_scale is not None:
                cmd += ["--lora-scale", str(args.lora_scale)]
            if args.steps is not None:
                cmd += ["--steps", str(args.steps)]
            if args.lightning_steps is not None:
                cmd += ["--lightning-steps", str(args.lightning_steps)]
            if args.batch is not None:
                cmd += ["--batch", str(args.batch)]
            procs.append((i, subprocess.Popen(cmd, env=env)))
        failures = 0
        for i, p in procs:
            code = p.wait()
            if code != 0:
                failures += 1
                print(f"[main] worker {i} exited with code {code}", flush=True)
        if failures:
            print(f"[main] {failures} worker(s) failed — completed images are safe on disk; re-run to resume", flush=True)

    # merge worker manifests + final snapshot
    shard_files = [os.path.join(out_dir, f"manifest.w{i}.jsonl") for i in range(n_gpus)]
    merged = kaggle_io.merge_manifests(out_dir, shard_files)
    done = kaggle_io.read_done([merged])
    files = kaggle_io.list_new_images(out_dir)
    kaggle_io.update_zip(out_dir, files)
    if pusher:
        pusher.stop()
    elif not args.no_push:
        kaggle_io.kaggle_push(out_dir, "cartoonstyle-run-complete")

    mins = (time.time() - t0) / 60
    print(f"[main] done: {len(done)} images total | {mins:.1f}m | manifest {merged}", flush=True)
    print(f"[main] zip: {os.path.join(out_dir, 'snapshots', 'images.zip')}", flush=True)


if __name__ == "__main__":
    main()
