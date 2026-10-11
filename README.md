# CartoonStyle

**Free, fast cartoon image factory on Kaggle's 2× T4 GPUs.**

Batch-generate hundreds of 1920×1080 (YouTube landscape) cartoon images with
**character/style consistency** via a LoRA you train in the same workflow —
for $0 on Kaggle's free tier.

- **SDXL + SDXL-Lightning** (4-step distillation) → ~2–4 s/image per GPU
- **Both T4s work in parallel** (one worker process per GPU → ~1.5–2 s/image wall-clock)
- **LoRA training** on your own style/character references (multi-GPU via `accelerate`)
- **Crash-proof output**: instant disk writes, resumable queues, periodic pushes to a
  private Kaggle Dataset — start a session and walk away

---

## The speed math

| Job | Time on 2× T4 |
|---|---|
| Per image | ~2–4 s per GPU → **~1.5–2 s** wall-clock with both GPUs |
| 500 images | ~15–25 min |
| 1000 images | ~30–50 min |
| LoRA training (one-time) | ~1–1.5 h |

Generation is the *fast* part. Training the LoRA is the one-time investment that
buys you consistency. For flat cartoon styles (solid colors, bold outlines) the
**2-step** Lightning setting often looks just as good and is ~2× faster —
use it for drafts, 4 or 8 steps for finals.

Images are generated at SDXL's native 16:9 bucket (**1344×768**) and upscaled to
**1920×1080** (Lanczos + unsharp by default; optional ESRGAN), which is both faster
and higher-quality than asking the model for 1080p directly.

---

## Consistency: the LoRA loop

This repo ships two style presets matching common YouTube looks:

| Preset | Config | Look |
|---|---|---|
| `flat_cartoon` | `config/default.yaml` | flat 2D stickman, thick outlines, solid colors (storytime-animation) |
| `paper_doodle` | `config/paper_doodle.yaml` | black-ink doodle stickman on crumpled paper |

**Workflow:**

1. **Collect 30–50 reference images** of your character/style (screenshots are fine)
   → upload them as a private Kaggle Dataset (e.g. `cartoonstyle-refs`)
2. **Caption them** (`MODE = "caption"` in the notebook) — then **hand-edit the
   `.txt` sidecars**. A style LoRA is only as good as its captions: describe the
   character, line style, palette, and scene every time. Keep your trigger token
   (`ckstyle`) in every caption.
3. **Train** (`MODE = "train"`) — both T4s, checkpoints every 250 steps so a session
   kill never loses the run. Output: `cartoonstyle-lora/final/`.
4. **Generate** (`MODE = "generate"` with `LORA_PATH` set) — every image now carries
   the character/style automatically.

For *flat, simple* art the LoRA learns quickly — 800–1200 steps is usually enough.
Re-train only when the style drifts or the character changes.

---

## Kaggle quickstart

1. **This repo must be reachable from Kaggle.** Either make it public on GitHub
   (notebook will `git clone` it), or zip it and attach the zip as a Kaggle Dataset.
2. Kaggle → **New Notebook** → Settings:
   - Accelerator: **GPU T4 x2**
   - Internet: **on**
3. **Attach datasets** (Add data):
   - your style references (`cartoonstyle-refs`) — for caption/train
   - optional: a previous outputs dataset — to `RESUME_FROM`
4. Import `notebooks/CartoonStyle-Kaggle.ipynb` (or paste its cells).
5. Set `MODE` and run all:
   - `caption` → review/edit the `.txt` files in the dataset (download, edit, re-upload,
     or edit in-place and copy to the dataset later)
   - `train` → produces `LORA_PATH`
   - `generate` → produces the images
6. Outputs land in `/kaggle/working/cartoonstyle-out/`:
   - `images/` — the PNGs (1920×1080)
   - `manifest.jsonl` — prompt/seed/params per image (for reproducibility)
   - `snapshots/images.zip` — rolling zip of everything so far
   - auto-pushed to a **private Kaggle Dataset** every 50 images / 10 min

### You will never lose images

- Every image is saved the **moment** it is generated (no end-of-run buffer).
- The job queue is deterministic and keyed — **re-running the same cell resumes**,
  skipping completed images. Session killed at hour 9? Just run again.
- A background pusher zips new images and pushes the folder to a private Kaggle
  Dataset on a timer, so even a hard crash keeps the work.
- Note: `/kaggle/working` holds ~20 GB of committed output — fine for <1k 1080p PNGs.

---

## CLI (also works locally on any CUDA machine)

```bash
pip install -r requirements.txt   # torch is preinstalled on Kaggle

# batch generate on all visible GPUs
python -m cartoonstyle.generate --config config/default.yaml \
    --images 500 --gpus 2 --lora /path/to/lora/final

# caption reference images
python -m cartoonstyle.caption --data-dir refs --prefix "ckstyle, flat cartoon style"

# train the LoRA (2 GPUs)
accelerate launch --num_processes 2 --mixed_precision fp16 \
    -m cartoonstyle.train_lora --data-dir refs --trigger ckstyle --steps 1200

# upscale an existing folder to 1080p
python -m cartoonstyle.upscale --in-dir raw --out-dir hd
```

### Useful flags (`generate`)

| Flag | Meaning |
|---|---|
| `--lightning-steps 2\|4\|8` | 2 = fastest drafts, 4 = default, 8 = max quality |
| `--images N` / `--seed N` | job count / deterministic base seed |
| `--batch N` | images per GPU batch (2 default; 1 with `--lowvram`) |
| `--lora PATH --lora-scale 0.85` | your style LoRA (dir or `.safetensors`) |
| `--resume-from DIR` | treat an older output dir as already-done |
| `--no-push` | disable Kaggle Dataset auto-push |
| `--base MODEL` | swap the SDXL base (e.g. `cagliostrolab/animagine-xl-3.1`) |

---

## Quotas & limits (Kaggle free tier, 2026)

- ~**30 GPU-hours/week**, sessions up to ~**12 h**, **20 GB** saved working output.
  T4×2 consumes quota for both GPUs — it burns 2× faster per hour but finishes
  jobs 2× faster.
- **Lightning is CFG-distilled**: keep `guidance_scale: 0` (default). Negative
  prompts then have little effect — put the style words in the *positive* prompt.
- First session downloads the base model (~7 GB) + Lightning LoRA (~400 MB) — about
  10 minutes. To skip that later, `kaggle datasets create` a `cartoonstyle-models`
  cache and load from `/kaggle/input/`.
- 1920×1080 is an **upscale** of the native 1344×768 generation. For extra crispness
  set `output.upscale_method: esrgan` and provide weights via `CARTOONSTYLE_ESRGAN`.

## Repo layout

```
config/            presets: default.yaml (flat cartoon), paper_doodle.yaml
cartoonstyle/
  generate.py      multi-GPU batch generator (resumable, auto-push)
  train_lora.py    SDXL LoRA trainer (accelerate multi-GPU)
  caption.py       BLIP auto-captioning for reference images
  upscale.py       1920x1080 upscale (lanczos / esrgan)
  kaggle_io.py     manifests, rolling zips, Kaggle Dataset pushes
  prompts.py       prompt lists (CSV/JSONL) + auto-variety templates
notebooks/CartoonStyle-Kaggle.ipynb
```
