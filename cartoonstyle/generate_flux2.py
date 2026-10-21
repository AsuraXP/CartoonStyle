"""FLUX.2 [klein] 4B production generator — thin-pen stickman on crumpled paper.

Winning recipe (locked by user 2026-10-16):
  * multi-reference character anchors (scene-free images only — no furniture to leak)
  * scene-first prompt + style block
  * guidance 2.0, 6 steps (Klein 4B is step-distilled; 4-6 is the sweet spot, >8 wastes time),
    1344x768 -> lanczos 1920x1080

Durability: per-shard manifests, instant local saves, background zip + Kaggle
dataset push, resume by manifest. Same contract as generate.py.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

# Reduce CUDA fragmentation before torch inits (recommended by PyTorch error
# message; costs nothing but makes the allocator pack blocks tighter, which
# matters when we're running within a few MiB of T4 capacity).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REFS = [
    # ONLY the character sheet. Stock refs were poisoning the style: one was a
    # 6-armed multitask stick figure (limb gremlin!), the other office workers
    # in ties/glasses. Character sheet alone gave the v2 the user approved.
    str(REPO_ROOT / "lora_refs/thin_pen/style_ref_character.png"),
]

# Default style: warm cream-paper cartoon (used for the ancient-communication video).
# Per-video projects can override by placing a style.txt next to their visuals.csv;
# if found, its contents replace STYLE_BLOCK (no SCENE_GUARD is appended).
STYLE_BLOCK = (
    "warm cream textured paper background, sepia and brown ink drawing with "
    "light watercolor wash, simple bold cartoon style, hand-drawn, friendly, "
    "clean lines, muted earth-tone colors, slightly rough sketch feel. "
)

SCENE_GUARD = ""


def load_style_block(prompts_path: str) -> str:
    """If a style.txt sits next to the visuals CSV, use it; else default."""
    override = Path(prompts_path).resolve().parent / "style.txt"
    if override.is_file():
        txt = override.read_text(encoding="utf-8").strip()
        if txt:
            print(f"[style] using override from {override}", flush=True)
            return txt + " "
    return STYLE_BLOCK


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FLUX.2 Klein multi-ref batch generator")
    p.add_argument("--prompts", required=True, help="CSV with visual_prompt (or prompt) column")
    p.add_argument("--images", type=int, default=0,
                   help="total images to make (0 = one image per CSV row, i.e. full video)")
    p.add_argument("--gpus", type=int, default=0,
                   help="number of GPU workers (0 = auto-detect: use all visible GPUs)")
    p.add_argument("--out", default="/kaggle/working/cartoonstyle-out")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--qc-retries", type=int, default=2,
                   help="self-healing: judge each image vs its prompt, redraw failures up to N times (0 = off)")
    p.add_argument("--qc-model", default=None, help="default: cartoonstyle.qc.DEFAULT_MODEL")
    p.add_argument("--repair", action="store_true",
                   help="QC existing images and redraw only the ones that fail")
    p.add_argument("--guidance", type=float, default=2.0)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--width", type=int, default=1344)
    p.add_argument("--height", type=int, default=768)
    p.add_argument("--model", default="Jinstudio/FLUX.1-schnell",
                   help="HF id for the schnell base (config/CLIP/VAE/T5). "
                        "Use an ungated mirror like Jinstudio/FLUX.1-schnell to "
                        "avoid needing an HF token for the license-gated BFL repo. "
                        "Transformer weights always come from city96's public GGUF quant.")
    p.add_argument("--refs", default=",".join(DEFAULT_REFS))
    p.add_argument("--offload", choices=["none", "model", "sequential"], default="model",
                   help="none = full model on GPU (RAM-light, needs ~13GB VRAM)")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--fix-bad", default="", help="qc.jsonl from a QC pass: redraw ONLY "
                    "keys flagged BAD (judge-free - the QC model is not loaded)")
    p.add_argument("--resume-from", default="")
    p.add_argument("--shard", default="", help="internal: k/N worker split")
    return p.parse_args()


def load_rows(path: str) -> list[tuple[str, str, str, str]]:
    """Returns (key_prefix, full_prompt, raw_scene, moment_text) per row."""
    import csv
    style = load_style_block(path)

    def _build(visual: str) -> str:
        v = visual.strip().rstrip(".,")
        if v.lower().startswith("scene:"):
            v = v.split(":", 1)[1].strip()
        return f"{style}{v}."

    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out = []
    for i, r in enumerate(rows, 1):
        sid = str(r.get("slide_id") or i).strip().zfill(2)
        mom = str(r.get("moment") or "").strip()
        prefix = f"beat{sid}m{mom.zfill(2)}" if mom else f"beat{sid}"
        narr = (r.get("moment_text") or "").strip()
        if r.get("prompt"):
            out.append((prefix, _build(r["prompt"].strip()), r["prompt"].strip(), narr))
        elif r.get("visual_prompt") and r["visual_prompt"].strip():
            raw = r["visual_prompt"].strip()
            out.append((prefix, _build(raw), raw, narr))
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
        prefix, prompt, scene, narr = prompts[i % n_rows]
        variant = i // n_rows
        seed = args.seed + i
        key = f"{prefix}_v{variant:02d}_s{seed:08x}"
        jobs.append((key, prompt, seed, scene, variant, narr))
    jobs = jobs[k::n]
    print(f"[gpu {k}] {len(jobs)} jobs of {args.images} total", flush=True)
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                      text=True, stderr=subprocess.DEVNULL).strip()
        dirty = " (dirty)" if subprocess.call(
            ["git", "diff", "--quiet"], stderr=subprocess.DEVNULL) else ""
        print(f"[gpu {k}] code version: {sha}{dirty}", flush=True)
    except Exception:
        print(f"[gpu {k}] code version: unknown", flush=True)

    is_schnell = "schnell" in args.model.lower()
    if is_schnell:
        from diffusers import FluxPipeline as PipeCls
    else:
        from diffusers import Flux2KleinPipeline as PipeCls

    refs = []
    if is_schnell:
        print(f"[gpu {k}] FLUX.1-schnell: text-only generation (style locked by prompt)", flush=True)
    else:
        for p in args.refs.split(","):
            if p.strip():
                im = Image.open(p.strip()).convert("RGB")
                im.thumbnail((512, 512))
                refs.append(im)
        print(f"[gpu {k}] {len(refs)} character anchors", flush=True)

    def build(offload: str, load_in_8bit: bool = False):
        tag = "FLUX.1-schnell" if is_schnell else "FLUX.2-klein"
        print(f"[gpu {k}] loading {tag} {args.model} - one time only", flush=True)
        # ---------- FLUX.1-schnell Q4 GGUF (12B, ~6.8GB transformer, T4-fit, Apache 2.0) -----
        # Transformer weights come from city96's public Q4_0 GGUF quant
        # (ungated). T5/CLIP/VAE/config come from the base model repo specified
        # by --model (default Jinstudio/FLUX.1-schnell, an ungated mirror of
        # black-forest-labs/FLUX.1-schnell so no HF token is required). The
        # transformer itself is reconstructed from the GGUF weights and is
        # dequantized to bf16 per-block on the fly during forward.
        # model_cpu_offload is the documented recipe for 16 GB cards. No manual
        # multi-GPU splits, no T5 hooks, no VAE-on-CPU hacks.
        if is_schnell:
            from diffusers import FluxTransformer2DModel, GGUFQuantizationConfig
            from huggingface_hub import hf_hub_download
            # city96's Q4_0 GGUF (6.77 GB file, ~8 GB VRAM resident when
            # dequantizing per-block). Fits one T4 with room to spare:
            #   Q4 transformer ~8 GB + CLIP ~0.3 GB + VAE ~0.2 GB + acts ~1-2 GB
            #   = ~10-11 GB out of 14.6 GiB usable.
            # NO model_cpu_offload: per-block CPU<->GPU shuttling is what made
            # sequential/offload runs take 90+ s/step. T5 stays on CPU (only
            # called once per image) with an input-routing hook; everything
            # else lives on cuda:0 for fast denoising.
            gguf_path = hf_hub_download(
                repo_id="city96/FLUX.1-schnell-gguf",
                filename="flux1-schnell-Q4_0.gguf",
            )
            print(f"[gpu {k}] loading Q4 GGUF transformer from {gguf_path}", flush=True)
            transformer = FluxTransformer2DModel.from_single_file(
                gguf_path,
                quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
                torch_dtype=torch.bfloat16,
                config=args.model,
                subfolder="transformer",
            )
            pipe = PipeCls.from_pretrained(
                args.model,
                transformer=transformer,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
            # Place modules explicitly.
            pipe.text_encoder.to("cuda:0")
            pipe.vae.to("cuda:0")
            pipe.transformer.to("cuda:0")
            pipe.text_encoder_2.to("cpu")
            # Route T5 inputs to CPU (accelerate doesn't auto-install this hook
            # for a module that's 100% on one device).
            def _t5_inputs_to_cpu(mod, args_, kwargs=None):
                a = tuple(t.to("cpu") if isinstance(t, torch.Tensor) else t
                          for t in (args_ or ()))
                kw = {kk: (vv.to("cpu") if isinstance(vv, torch.Tensor) else vv)
                      for kk, vv in (kwargs or {}).items()}
                return a, kw
            pipe.text_encoder_2.register_forward_pre_hook(_t5_inputs_to_cpu, with_kwargs=True)
            # VAE slicing only (no tiling): tiling adds 20-30s/image of tile-
            # stitching overhead; at 1024x576 with Q4 transformer resident, a
            # single-pass decode fits in T4 VRAM. If an OOM fires at decode,
            # add tiling back.
            pipe.vae.enable_slicing()
            mode_used = "gguf-q4-gpu"
            print(f"[gpu {k}] {tag} Q4_0 GGUF ready (T5 on CPU, transformer+CLIP+VAE on cuda:0, VAE tiled+sliced)", flush=True)
            return pipe, mode_used
        # ---------- FLUX.2-klein 4B (fast, Apache 2.0, ungated) ---------
        pipe = PipeCls.from_pretrained(args.model, torch_dtype=torch.float16)
        if offload == "sequential":
            pipe.enable_sequential_cpu_offload()
        elif offload == "model":
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        return pipe, offload

    # stagger loads slightly so workers don't fight for host RAM / bandwidth
    # on startup; GGUF download is cached so the second worker waits then uses
    # the cached file.
    delay = k * 20
    if delay:
        print(f"[gpu {k}] staggered start: waiting {delay}s", flush=True)
        time.sleep(delay)
    use_8bit = "9B" in args.model
    # Auth / gate errors are NOT recoverable via offload fallbacks — fail fast
    # with a clear hint instead of noisily retrying the same gated repo.
    def _is_auth_err(e: BaseException) -> bool:
        ename = type(e).__name__
        msg = str(e)
        return (ename in ("GatedRepoError", "HTTPStatusError", "RepositoryNotFoundError",
                          "LocalEntryNotFoundError")
                or "401" in msg or "403" in msg or "gated" in msg.lower()
                or "access to model" in msg.lower())

    # Resolve build mode.
    # For schnell (GGUF Q4): only one mode (gguf-q4) — the offload arg is ignored
    # and model_cpu_offload is applied inside build().
    # For klein 4B: respect args.offload with standard fallbacks.
    if is_schnell:
        order = [("model", False)]
    else:
        order = [(args.offload, use_8bit),
                 ("model", use_8bit), ("sequential", use_8bit),
                 ("model", False), ("sequential", False)]
    pipe = None
    used_mode = None
    last_auth_err = None
    first_try = True
    for alt, bit in order:
        try:
            pipe, used_mode = build(alt, load_in_8bit=bit)
            if not first_try:
                print(f"[gpu {k}] running with mode={used_mode}{' 8-bit' if bit else ''}", flush=True)
            args.offload = used_mode
            break
        except Exception as e:
            if _is_auth_err(e):
                last_auth_err = e
                break
            if first_try:
                print(f"[gpu {k}] initial load failed ({type(e).__name__}: {e}) -> falling back", flush=True)
                import traceback; traceback.print_exc()
            gc.collect(); torch.cuda.empty_cache()
            first_try = False
    if last_auth_err is not None:
        tag = "FLUX.1-schnell" if is_schnell else "FLUX.2-klein"
        print(f"[gpu {k}] AUTH ERROR loading {tag}: {last_auth_err}", flush=True)
        print(f"[gpu {k}] -> accept license on HF + set HF_TOKEN env var", flush=True)
        raise last_auth_err
    if pipe is None:
        raise RuntimeError(f"[gpu {k}] failed to load {args.model} in any mode")
    ready_tag = "FLUX.1-schnell" if is_schnell else "FLUX.2-klein"
    print(f"[gpu {k}] {ready_tag} ready ({args.offload} mode)", flush=True)

    manifest = os.path.join(args.out, f"manifest.shard{k}.jsonl")
    import glob as _glob
    all_shards = sorted(_glob.glob(os.path.join(args.out, "manifest.shard*.jsonl")))
    done = kaggle_io.read_done(all_shards + [os.path.join(args.out, "manifest.jsonl")],
                               images_dir=os.path.join(args.out, "images"))
    print(f"[gpu {k}] resume: {len(done)} images already done", flush=True)
    if args.resume_from:
        done.update(kaggle_io.read_done([os.path.join(args.resume_from, "manifest.jsonl")]))

    out_w, out_h = 1920, 1080  # delivery format: 1344x768 -> lanczos upscale

    from .qc import judge_image, load_judge, DEFAULT_MODEL, CHECKS

    judge = None
    if args.fix_bad:
        judge = None  # phase-3 style: verdicts already exist in qc.jsonl
    elif args.qc_retries > 0 or args.repair:
        try:
            qc_name = args.qc_model or DEFAULT_MODEL
            qc_dev = "cuda:1" if torch.cuda.device_count() > 1 else "cpu"
            judge = load_judge(qc_name, device=qc_dev)
            print(f"[gpu {k}] vision QC armed ({qc_name}) on {qc_dev} - self-healing", flush=True)
        except Exception as e:
            print(f"[gpu {k}] WARNING: vision QC unavailable ({e}) - generating unchecked", flush=True)
            judge = None
    if args.repair and judge is None:
        raise SystemExit("--repair needs the vision QC model - install qwen-vl-utils / transformers")

    rejected_dir = os.path.join(args.out, "rejected")
    qc_log = os.path.join(args.out, "qc.jsonl")

    pusher = None
    if not args.no_push:
        pusher = kaggle_io.Pusher(args.out, every_images=25, every_seconds=600).start()

    def draw(aseed: int, prompt: str):
        gen = torch.Generator("cpu" if is_schnell else "cuda").manual_seed(aseed)
        # Schnell is guidance-distilled at 0.0, but Flux's embedded-guidance
        # accepts values 0-3 to tighten prompt adherence without true CFG cost.
        # We use 1.5 to enforce the strict "don't draw extras" language in
        # SCENE_GUARD/STYLE_BLOCK (negative_prompt is ignored on schnell without
        # true_cfg_scale which doubles compute, so we rely on in-prompt bans).
        call = dict(prompt=prompt, width=args.width, height=args.height,
                    guidance_scale=(0.0 if is_schnell else args.guidance),
                    num_inference_steps=args.steps, generator=gen)
        if is_schnell:
            call["max_sequence_length"] = 256
        if not is_schnell:
            call["image"] = refs
        return pipe(**call).images[0]

    # Overlap engine: GPU0 draws back-to-back while GPU1 judges the picture that
    # just finished. BAD verdicts re-queue the key with a fresh seed; GOOD
    # finalizes. Judge time hides inside the next draw.
    import collections
    from concurrent.futures import ThreadPoolExecutor

    max_tries = 1 + (args.qc_retries if judge else 0)
    flagged = 0
    retried = 0
    unchecked = 0
    done_n = 0
    total_jobs = len(jobs)
    finished = set()
    bestbad = {}   # key -> (score, img, rec)  best BAD attempt seen

    def judge_img(img, words, narr):
        nonlocal judge
        if judge is None:
            return {"score": None, "verdict": "UNCHECKED", "problem": ""}
        try:
            v = judge_image(judge[0], judge[1], img, words, narr)
            return {"score": v["score"], "verdict": v["verdict"], "problem": v["problem"]}
        except Exception as e:
            print(f"[gpu {k}] WARNING: judge failed ({e}) - images now ship UNCHECKED", flush=True)
            judge = None
            return {"score": None, "verdict": "UNCHECKED", "problem": ""}

    def finish(key, prompt, variant, dest, img, rec, t0):
        nonlocal flagged, retried, unchecked, done_n
        if rec["attempts"] > 1:
            retried += 1
        if rec["verdict"] == "UNCHECKED":
            unchecked += 1
        elif rec["verdict"] != "GOOD":
            flagged += 1
        img.save(dest)
        dt = time.time() - t0
        done_n += 1
        kaggle_io.append_manifest(manifest, {"key": key, "id": key, "prompt": prompt,
                                             "sec": round(dt, 1), **rec})
        with open(qc_log, "a", encoding="utf-8") as qf:
            qf.write(json.dumps({"key": key.rsplit("_v", 1)[0], "variant": variant,
                                 "file": os.path.basename(dest), **rec}) + "\n")
        print(f"[gpu {k}] [{done_n}/{total_jobs}] {key} {dt:.1f}s saved", flush=True)

    work = collections.deque()   # (key, prompt, scene, seed, variant, narr, t0, try_idx)
    fix_bad = set()
    if args.fix_bad:
        with open(args.fix_bad, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("verdict") and rec["verdict"] != "GOOD" and rec.get("key"):
                    fix_bad.add(rec["key"])
        print(f"[gpu {k}] fix-bad: {len(fix_bad)} flagged key(s) to redraw", flush=True)
    for ji, (key, prompt, seed, scene, variant, narr) in enumerate(jobs, 1):
        dest = os.path.join(args.out, f"images/{key}.png")
        if args.repair:
            if os.path.exists(dest):
                if judge is None:
                    continue
                try:
                    v = judge_image(judge[0], judge[1], dest, prompt, narr)
                except Exception:
                    v = {"verdict": "BAD", "score": 0, "problem": "unreadable"}
                print(f"[gpu {k}] repair: {key} {v['verdict']} ({v['score']}/{len(CHECKS)}) "
                      f"{v['problem'] if v['verdict'] != 'GOOD' else ''}", flush=True)
                if v["verdict"] == "GOOD":
                    pass
                else:
                    # original stays in the competition: a redraw replaces it ONLY
                    # if it scores higher (false alarms can never downgrade art)
                    _orig = Image.open(dest).convert("RGB")
                    bestbad[key] = (v["score"], _orig,
                                    {"seed": seed, "score": v["score"], "verdict": v["verdict"],
                                     "problem": v["problem"], "attempts": 1})
                if v["verdict"] == "GOOD":
                    kaggle_io.append_manifest(manifest, {"key": key, "id": key, "prompt": prompt,
                                                         "seed": seed, "sec": 0, "score": v["score"],
                                                         "verdict": "GOOD", "problem": "",
                                                         "attempts": 0})
                    with open(qc_log, "a", encoding="utf-8") as qf:
                        qf.write(json.dumps({"key": key.rsplit("_v", 1)[0], "variant": variant,
                                             "file": os.path.basename(dest), "seed": seed,
                                             "score": v["score"], "verdict": "GOOD",
                                             "problem": "", "attempts": 0}) + "\n")
                    continue
            # missing or BAD: redraw with a different seed than the original (try>=2)
            work.append((key, prompt, scene, seed, variant, narr, time.time(),
                         2 if os.path.exists(dest) else 1))
            continue
        if args.fix_bad:
            if key.rsplit("_v", 1)[0] in fix_bad:
                work.append((key, prompt, scene, seed, variant, narr, time.time(), 2))
            continue
        if key in done:
            continue
        work.append((key, prompt, scene, seed, variant, narr, time.time(), 1))

    jexec = ThreadPoolExecutor(max_workers=1)
    inflight = {}   # future -> (key, prompt, scene, seed, variant, narr, t0, try_idx, img)

    def harvest(block: bool):
        if not inflight:
            return
        futs = list(inflight)
        done_fs = [futs[0]] if block else [f for f in futs if f.done()]
        for f in done_fs:
            key, prompt, scene, seed, variant, narr, t0, try_idx, img = inflight.pop(f)
            v = f.result()
            rec = {"seed": seed + (try_idx - 1) * 7919, "score": v["score"],
                   "verdict": v["verdict"], "problem": v["problem"], "attempts": try_idx}
            print(f"[gpu {k}] {key} try{try_idx}/{max_tries} {v['verdict']} "
                  f"({v['score']}/{len(CHECKS)}) {v['problem'] if v['verdict'] == 'BAD' else ''}",
                  flush=True)
            dest = os.path.join(args.out, f"images/{key}.png")
            if v["verdict"] in ("GOOD", "UNCHECKED"):
                finished.add(key)
                finish(key, prompt, variant, dest, img, rec, t0)
                continue
            os.makedirs(rejected_dir, exist_ok=True)
            img.save(os.path.join(rejected_dir, f"{key}_try{try_idx}.png"))
            old = bestbad.get(key)
            if old is None or (old[2].get("score") or -1) < (rec["score"] or -1):
                bestbad[key] = (rec["score"], img, rec)
            if try_idx >= max_tries or judge is None:
                finished.add(key)
                _s, bimg, brec = bestbad.pop(key, (rec["score"], img, rec))
                finish(key, prompt, variant, dest, bimg, brec, t0)
            else:
                work.appendleft((key, prompt, scene, seed, variant, narr, t0, try_idx + 1))  # retries jump the queue

    while work or inflight:
        if work:
            key, prompt, scene, seed, variant, narr, t0, try_idx = work.popleft()
            if key in finished:
                continue
            print(f"[gpu {k}] drawing {key} try{try_idx}/{max_tries}", flush=True)
            img = draw(seed + (try_idx - 1) * 7919, prompt)
            if (img.width, img.height) != (out_w, out_h):
                img = img.resize((out_w, out_h), Image.LANCZOS)
            if judge is None:
                rec = {"seed": seed + (try_idx - 1) * 7919, "score": None,
                       "verdict": "UNCHECKED", "problem": "", "attempts": try_idx}
                finished.add(key)
                finish(key, prompt, variant, os.path.join(args.out, f"images/{key}.png"),
                       img, rec, t0)
            else:
                fut = jexec.submit(judge_img, img, scene, narr)
                inflight[fut] = (key, prompt, scene, seed, variant, narr, t0, try_idx, img)
                harvest(block=False)
        else:
            harvest(block=True)   # nothing to draw: wait on the oldest verdict
    jexec.shutdown()

    print(f"[gpu {k}] self-heal summary: {retried} image(s) needed redraws, "
          f"{flagged} still flagged after all tries, {unchecked} UNCHECKED", flush=True)
    if unchecked and judge is None and not args.fix_bad and (args.qc_retries > 0 or args.repair):
        print(f"[gpu {k}] WARNING: the robot never judged anything - install qwen-vl-utils!", flush=True)

    if pusher:
        pusher.stop()
        pusher.join(120)
    print(f"[gpu {k}] shard done", flush=True)


def main() -> None:
    args = parse_args()
    is_schnell = "schnell" in args.model
    # Auto-tune for FLUX.1-schnell on T4: native 1 MP budget fits the 24 GB
    # transformer in bf16 across 2xT4 with balanced split. 1344x768 (1.03 MP)
    # was too tight — activations OOMed even with VAE tiling. 1024x576 (0.59 MP)
    # is everyone's T4 sweet spot and still upscales cleanly to 1920x1080 with
    # lanczos. Guidance=0 is required for schnell (distilled).
    if is_schnell:
        if args.width == 1344 and args.height == 768:     # untouched defaults
            args.width, args.height = 1024, 576
        if args.guidance == 2.0:                           # untouched default
            args.guidance = 1.5
        if args.steps > 8:
            args.steps = 4
    import torch  # needed below for device_count and spawns
    if args.shard:
        run_worker(args)
        return

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    # Default --images=0 means "render every row in the CSV once" (full video).
    if args.images <= 0:
        with open(args.prompts, newline="", encoding="utf-8") as f:
            args.images = sum(1 for _ in csv.DictReader(f))
        print(f"[main] auto-detected {args.images} rows in {args.prompts}", flush=True)
    res_note = f"{args.width}x{args.height}" + (" (schnell T4 default -> lanczos to 1920x1080)" if is_schnell and args.width == 1024 else "")
    print(f"[main] {args.images} images | seed {args.seed} | {res_note} | out {args.out}", flush=True)

    # Auto-detect GPUs when --gpus=0 (default): one worker per visible GPU.
    # Each worker pins to exactly one GPU via CUDA_VISIBLE_DEVICES so workers
    # don't fight over memory — works on Kaggle 2xT4, Colab single T4/A100/L4.
    n_gpu = torch.cuda.device_count()
    if args.gpus <= 0:
        args.gpus = max(1, n_gpu)
    print(f"[main] using {args.gpus} drawing worker(s) across {n_gpu} visible GPU(s)", flush=True)

    # Pre-download the schnell GGUF in the main process so N parallel workers
    # don't race to write into the HF cache at the same time (corrupts the file).
    if is_schnell:
        from huggingface_hub import hf_hub_download
        print("[main] pre-downloading Q4 GGUF (one time)...", flush=True)
        gguf_path = hf_hub_download(
            repo_id="city96/FLUX.1-schnell-gguf",
            filename="flux1-schnell-Q4_0.gguf",
        )
        print(f"[main] GGUF cached at {gguf_path}", flush=True)

    procs = []
    for k in range(args.gpus):
        # Pin each worker to exactly one GPU: worker k -> cuda:0 from the
        # worker's own view (CUDA_VISIBLE_DEVICES rewrites the numbering).
        # The old "keep all GPUs visible for balanced split" logic is gone
        # because schnell GGUF loads one full model per GPU now.
        vis = str(k % max(1, n_gpu))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=vis)
        cmd = [sys.executable, "-m", "cartoonstyle.generate_flux2",
               "--prompts", args.prompts, "--images", str(args.images),
               "--out", args.out, "--seed", str(args.seed),
               "--guidance", str(args.guidance), "--steps", str(args.steps),
               "--width", str(args.width), "--height", str(args.height),
               "--refs", args.refs, "--model", args.model, "--offload", args.offload,
               "--qc-retries", str(args.qc_retries), "--fix-bad", args.fix_bad,
               "--shard", f"{k}/{args.gpus}"]
        if args.qc_model:
            cmd += ["--qc-model", args.qc_model]
        if args.repair:
            cmd += ["--repair"]
        if args.no_push:
            cmd += ["--no-push"]
        if args.resume_from:
            cmd += ["--resume-from", args.resume_from]
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
