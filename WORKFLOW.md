# CartoonStyle — Complete Workflow

The narration drives the images. One approved script → one prompt set for the
whole video → ONE unattended Kaggle pass → images that react to the narrator.

```
 ① SCRIPT WRITER   topic -> professional, catchy, non-repetitive script (beats)
 ② HUMAN APPROVAL  you edit until it sings                       <- YOU are here
 ③ TTS NARRATION   per-beat audio + EXACT timings (audio is the master)
 ④ PROMPTS         every narration word -> visible in its beat's image
 ⑤ ONE KAGGLE PASS whole video's images, unattended, never lost
 ⑥ STITCH          ffmpeg: images under their own narration -> VIDEO.mp4
```

---

## Stage ① — Script Writer (local, no GPU)

Write the draft as **beats**: one short block per spoken moment, blank line
between blocks. Then validate it against the craft rules (hook in beat 1,
non-repetitive openings, no filler openers, 8–45 words per beat):

```bash
python -m cartoonstyle.scriptwriter validate --script examples/my_video/script.txt
```

When the draft is clean, emit the package skeleton:

```bash
python -m cartoonstyle.scriptwriter package --script examples/my_video/script.txt --out-dir examples/my_video
```

This writes:
* `script.md` — numbered beats + timecodes (150 wpm)
* `beats.jsonl` — machine-readable beats (`words_to_cover` per beat)
* `visuals.csv` — empty `visual_prompt` column + the words each image must show

## Stage ② — You approve the script

Edit `script.txt`, re-run `validate`, repeat until the report says PASS.
Nothing downstream runs before you are happy.

## Stage ③ — TTS narration (audio is the master)

Synthesize one voice file per beat and record exact durations (Kaggle/Colab
images already include ffmpeg/ffprobe):

```bash
python -m cartoonstyle.tts --beats examples/my_video/beats.jsonl \
    --out-dir examples/my_video --voice en-US-ChristopherNeural --rate "+0%"
```

Default voice = `en-US-ChristopherNeural` (Microsoft neural, free, no key).
Swap any voice with `--voice` — the stage is backend-pluggable.

## Stage ④ — Prompts that mirror the narration (TTS is the master)

Fill `visual_prompt` in `visuals.csv` so **every content word of the beat is
visible in that image** (people, objects, actions, places the narrator names).
Then run the coverage gate — this is what "images react to the words" means:

```bash
python -m cartoonstyle.scriptwriter cover --script examples/my_video/script.txt --visuals examples/my_video/visuals.csv
```

`COVERAGE PASS` = the viewer sees everything the narrator says. Fix misses and
re-run until it passes. The generator wraps each `visual_prompt` in the locked
style automatically (scene-first prompt + thin-pen paper style + character
anchors) — you only describe the SCENE.

## Stage ⑤ — ONE Kaggle pass (the whole video)

Dry run first — 12 images, and the log prints seconds/image:

```python
!test -d /kaggle/working/CartoonStyle || git clone --depth 1 -b arena/22c9cd35-cartoonstyle https://github.com/AsuraXP/CartoonStyle.git
%cd /kaggle/working/CartoonStyle
!git pull
!python -m cartoonstyle.generate_flux2 \
    --prompts examples/my_video/visuals.csv \
    --images 12 --gpus 2 \
    --out /kaggle/working/dryrun12
```

ALWAYS start sessions with the clone-if-missing line: Kaggle hands out a
brand-new empty /kaggle/working whenever the session is recycled.

Full run (600 = 18 beats x 33 variants; use 1000 if timing < 20s/image):

```python
!python -m cartoonstyle.generate_flux2 \
    --prompts examples/my_video/visuals.csv \
    --images 600 --gpus 2 \
    --out /kaggle/working/cartoonstyle-out
```

Safety (built in):
* every image saved the second it is made
* background Kaggle Dataset push every 25 images / 10 min (never blocks drawing)
* crashed? **run the same command again** — it resumes exactly where it stopped
* CUDA OOM? auto-switches to the slower safe mode and keeps going

## Stage ⑥ — Assemble the video (automatic)

```bash
python -m cartoonstyle.stitch --timing examples/my_video/beats_with_timing.json \
    --images-dir /kaggle/working/cartoonstyle-out/images \
    --out /kaggle/working/VIDEO.mp4 --pick 0
```

Outputs are named `beatNN_vMM_sSEED.png` (NN = narration beat, MM = variant).
`--pick N` chooses which variant plays under each beat — review the run, pick
the good ones, re-stitch in seconds.

---

## Speed math (T4 x2)

| seconds/image | 600 images | 1000 images |
|---|---|---|
| 15s | ~75 min | ~2 h |
| 25s | ~2 h | ~3.5 h |
| 40s | ~3.3 h | ~5.5 h |

## The locked recipe (why these settings)

* **FLUX.2 [klein] 4B** — the first base that obeys faces/scenes on free T4s
* **multi-reference anchors** — scene-free character images only (scene images
  leak their furniture into every generation — learned the hard way)
* **scene-first prompts** + guard "only objects that belong in this scene"
* **guidance 2.0, 4 steps, 1344x768 -> 1920x1080** — the tested, approved look

## Troubleshooting

| Gremlin | Cure |
|---|---|
| `torchao` / `FqnToConfig` import crash | the tools self-purge it; for raw cells run the nuclear eviction cell (RAM+disk) then Restart session |
| "tried to allocate more memory" banner | host RAM — Restart session, run ONE pipeline per kernel |
| `FileNotFoundError` on refs | the `!test -d ... || git clone ...` line is missing at the top of the cell |
| wrong scene furniture in images | a scene-bearing image is in the anchor set — keep anchors scene-free |
