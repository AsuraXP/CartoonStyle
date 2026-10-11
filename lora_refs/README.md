# LoRA training reference sets

Ready-to-train style/character datasets for CartoonStyle. Each `.jpg` has a
sidecar `.txt` caption (trigger token `ckstyle` / `ckdoodle` + style + scene).

## flat_cartoon/ — the colored storytime style

Flat 2D cartoon stickman (round white head, thick black outlines, flat solid
colors, beige rooms, wooden plank floors). 8 varied scenes covering different
emotions, poses, props and rooms — the variation is what makes the LoRA
generalize instead of memorizing one frame.

## paper_doodle/ — the ink-on-paper style *(batch 2, in progress)*

Black-ink doodle stickman on crumpled paper texture.

## How to use

1. Zip this folder (or upload its contents) as a **private Kaggle Dataset**
   (e.g. `cartoonstyle-refs`).
2. In the notebook: `MODE = "caption"` is optional (captions are already
   written — skip or re-run with `--overwrite` to taste).
3. `MODE = "train"` → `DATA_DIR = "/kaggle/input/cartoonstyle-refs/flat_cartoon"`
   (train **one LoRA per style folder**).
4. `MODE = "generate"` with `LORA_PATH` pointing at the trained `final/` dir.

**Tip:** the more varied scenes, the better. 30–50 images is the ideal range —
top this set up with screenshots from your own videos or extra batches, and
hand-edit any caption that doesn't match its image exactly.
