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

**You don't download anything by hand.** The notebook's first cell `git clone`s
this repo automatically (it's public) — code, configs, storyboard tool and even
the LoRA training images in `lora_refs/` all come down by themselves.

1. Kaggle → **New Notebook** → Settings:
   - Accelerator: **GPU T4 x2**
   - Internet: **on** (needed for the clone + model download)
2. Import `notebooks/CartoonStyle-Kaggle.ipynb` (or paste its cells).
3. **Attach datasets** (Add data), only if you have them:
   - your own style references — optional, `lora_refs/flat_cartoon` already ships in the repo
   - a previous outputs dataset — to `RESUME_FROM`
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

## From narration script → slideshow images

`cartoonstyle/storyboard.py` turns a narration script into per-slide visual
prompts and an image job list — the "600 images that match what the narrator
says" workflow:

1. **Write the script with blank lines between slides** (one paragraph = one
   slide). `[SFX ...]` notes and `NARRATOR:` labels are stripped automatically;
   a short first line is the title. See `examples/test_narration_script.txt`.
2. **Write one visual scene prompt per slide** (a CSV of `slide_id,visual_prompt`,
   like `examples/test_visuals.csv`) — by hand or with any LLM. This is what
   decides image quality; skip it and the built-in `auto` mode drafts crude
   keyword prompts (fine for plumbing tests only).
3. **Build the job list** (runs anywhere, no GPU):

   ```bash
   python -m cartoonstyle.storyboard --script my_script.txt \
       --out-dir storyboard_out --images-total 600 \
       --trigger ckstyle --visuals my_visuals.csv
   ```

   Output: `prompts.csv` (feed to the generator) + `slides.jsonl` mapping each
   narration line to its image index range — use it in your editor to place
   images under the right spoken lines.
4. **Generate** (on Kaggle, both T4s):

   ```bash
   python -m cartoonstyle.generate --prompts storyboard_out/prompts.csv \
       --images 600 --gpus 2 --lora .../final
   ```

   600 images ≈ **20–30 min** on 2×T4. Each slide gets several seed-variations
   (`count` column) so you can pick the best take in editing.

In the notebook: `MODE = "storyboard"` (step 3), then `MODE = "generate"` with
`PROMPTS_CSV` set (step 4).

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

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'diffusers.loaders.lora'` at checkpoint save | fixed in latest code — `git pull`, restart training (add `--save-every 100`). Saves are now PEFT-format first (version-proof), diffusers-format as bonus. |
| `ImportError: Found an incompatible version of torchao` during `train_lora` | `!pip uninstall -y torchao` then re-run. Kaggle ships torchao 0.10.0 and peft raises at LoRA setup just because it's installed (we never use it). The notebook setup cell does this automatically. |
| CUDA out of memory | add `--batch 1 --lowvram` to `generate`; for training lower `--width/--height` to 768 |
| `accelerate: command not found` | re-run the setup cell's `pip install -r requirements.txt` |
| Session killed mid-run | new session → re-run the same cells — generation resumes, training resumes from `--resume checkpoints/step_XXXXXX` |
| Slow first cell | normal: SDXL base (~7 GB) + Lightning LoRA download once; cache it in a Kaggle Dataset to skip later |

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
