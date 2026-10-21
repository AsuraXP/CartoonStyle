# Ancient Communication video — STATUS & FRESH-START GUIDE
Last updated: 2026-10-17 (after the 8-step limb test PASSED)

## Where we are
- Script: APPROVED — `examples/ancient_communication_video/script.txt` (17 beats, ~3:09)
- Moments: `visuals.csv` — 73 rows, every narration word claimed (coverage gate PASS)
- Timing: **PERFECT (user-verified)** — one-take edge-tts + WordBoundary word-locking
- Image recipe: **LOCKED (user-verified, 8-step test: clean limbs)** — full redraw NOT yet run
- Video: v1 exists but is superseded (wrong bg, anatomy bugs, old timing)

## Style (CURRENT — user-corrected 2026-10-17, supersedes the old "paper" gate)
- Thin clean ballpoint pen lines, minimalist stickman, **plain smooth off-white background
  (NOT crumpled paper — user explicitly rejected paper on 2026-10-17)**
- Light flat colors; figures LARGE in frame; scenes FULL of their named objects
- Objects at natural real-world scale (a lion is huge — never a toy/miniature/floating)
- Anatomy: exactly two arms/legs, two hands, one head — no third arm, no extra limbs,
  stick limbs (thin pen strokes), never animal legs/hooves
- Character: round white head, few hair strands, two small oval eyes with pupils, smile
- Reference: **ONLY `lora_refs/thin_pen/style_ref_character.png`** (crowd/multitask refs
  benched — they donated small figures/empty scenes)

## Hard-won facts (do not re-litigate)
- **1 GPU only.** Two workers SIGKILL each other (host RAM ~30GB < 2 model copies).
  `--offload none` does NOT fit T4 VRAM either. Stable = `--gpus 1` + model offload.
- **`--steps 8` required.** 4 steps = crooked/extra limbs. 8 ≈ 50s/image (~65 min for 73).
- Guidance scale is IGNORED on distilled FLUX.2 Klein 4B (log warns every call).
- edge-tts 7.x needs `Communicate(..., boundary="WordBoundary")` or words never arrive.
- The word matcher must expand numerals ("40,000" → "forty thousand") + year-style variants.
- Fresh Kaggle sessions WIPE /kaggle/working — always start with clone-if-missing.
- ONE cell at a time on Kaggle. Never poke a side cell while one runs.
- 1 image per word-moment (user choice). Review the VIDEO once; fix-run only bad beats.

## SELF-HEALING FACTORY (built 2026-10-18)
- `generate_flux2 --qc-retries 2` (DEFAULT): every image is judged by Qwen2-VL-2B against
  its own prompt the moment it is drawn (limbs / animals / scene). Failures redraw with a
  fresh seed up to 2 times; losers go to rejected/. qc.jsonl + manifest record verdicts.
- `--repair`: QC the existing images/ folder and redraw ONLY failures (~20 min vs full re-run).
- `stitch --qc qc.jsonl`: auto-picks the best-scored candidate per moment.
- Standalone audit: `python -m cartoonstyle.qc --images-dir I --out D [--prompts visuals.csv]`.
- Factory flow now: draw (self-healing) -> TTS -> stitch --qc -> watch the story. No manual checks.

## RESTORE after a session wipe (images live in the Dataset)
```python
import json, os
USER = os.environ.get("KAGGLE_USERNAME") or json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))["username"]
print("dataset owner:", USER)
!rm -rf /kaggle/working/restore && mkdir -p /kaggle/working/restore /kaggle/working/ancient-comm-v2
!kaggle datasets download -d {USER}/ancient-comm-v2 -p /kaggle/working/restore --unzip
!unzip -o -q $(find /kaggle/working/restore -name "images.zip" | head -1) -d /kaggle/working/ancient-comm-v2
!ls /kaggle/working/ancient-comm-v2/images | wc -l   # must print 73
```
`${KAGGLE_USERNAME}` is EMPTY in Kaggle shell cells - always resolve via the python
block above (~/.kaggle/kaggle.json holds the username).

## TOMORROW — fresh Kaggle session, 3 cells

### Cell 1 — setup + full redraw (~65 min)
```python
!test -d /kaggle/working/CartoonStyle || git clone --depth 1 -b arena/22c9cd35-cartoonstyle https://github.com/AsuraXP/CartoonStyle.git
%cd /kaggle/working/CartoonStyle
!git pull
!pip install -q qwen-vl-utils

!python -m cartoonstyle.generate_flux2 \
    --prompts examples/ancient_communication_video/visuals.csv \
    --images 73 --gpus 1 --steps 8 \
    --refs /kaggle/working/CartoonStyle/lora_refs/thin_pen/style_ref_character.png \
    --out /kaggle/working/ancient-comm-v2
```

NEVER omit `pip install -q qwen-vl-utils` - without it the vision judge cannot
load its helper lib and every image ships UNCHECKED (the summary will say so).
Wait for `[main] done: 73 images`. Images auto-push to the Kaggle Dataset as they draw.

### Cell 2 — narration + timing (~2 min) + captioned stitch (~2 min)
```python
%cd /kaggle/working/CartoonStyle
!pip install -q edge-tts
!curl -sL -o /kaggle/working/Montserrat.ttf "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf"
!python -m cartoonstyle.tts \
    --beats examples/ancient_communication_video/beats.jsonl \
    --visuals examples/ancient_communication_video/visuals.csv \
    --out-dir /kaggle/working/ancient-comm
!python -m cartoonstyle.stitch \
    --timing /kaggle/working/ancient-comm/beats_with_timing.json \
    --images-dir /kaggle/working/ancient-comm-v2/images \
    --out /kaggle/working/VIDEO.mp4 \
    --captions --font /kaggle/working/Montserrat.ttf
```
Healthy TTS report card: `narration.mp3 written (~470 words tracked)` +
`73 image moments locked to spoken words [OK]`. A `0 words tracked` means
WordBoundary is broken again; `n MISSED` prints the offending moments.

### Cell 3 — watch VIDEO.mp4, list bad moment keys (beatNNmMM) for targeted redraws

## Captions (BUILT 2026-10-18, user wants publish-ready)
- `stitch --captions --font /kaggle/working/Montserrat.ttf` burns word-locked subtitles:
  Montserrat Bold 58px, white with dark outline, bottom-center, long lines roll in time,
  sfx cues never captioned. Audio loudnormed to -14 LUFS (YouTube standard) by default.
- Font download cell: `curl -sL -o /kaggle/working/Montserrat.ttf "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf"`
  (missing font degrades gracefully to system sans).

## Remaining TODO (after publish)
- WORKFLOW.md + notebook rewrite (SDXL-era notebook is stale)
- Optional: 0.4s crossfade (user's format choice is hard cuts), int8 2-GPU speed work

## CLI cheat sheet
```
python -m cartoonstyle.scriptwriter validate --script X.txt
python -m cartoonstyle.scriptwriter cover --script X.txt --visuals V.csv
python -m cartoonstyle.scriptwriter package --script X.txt --out-dir D
python -m cartoonstyle.tts --beats D/beats.jsonl --visuals V.csv --out-dir D [--force]
python -m cartoonstyle.generate_flux2 --prompts V.csv --images N --gpus 1 --steps 8 --refs R.png --out D
python -m cartoonstyle.stitch --timing D/beats_with_timing.json --images-dir I --out VIDEO.mp4 [--pick N]
python -m cartoonstyle.contact_sheet --images-dir I --out sheet.jpg
```
Schemas: visuals.csv `slide_id, moment, moment_text, visual_prompt`;
moments_timing.json `[{key: beatNNmMM, beat, start, sec, text}]`;
images `beatNNmMM_vMM_sSEED.png`. Voice: `en-US-AndrewNeural` (user-picked).
