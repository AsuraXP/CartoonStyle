# CartoonStyle Video Production Workflow (v5 — Apoc Day 1 proven)

End-to-end pipeline used to produce **"Day 1 of the Zombie Apocalypse (I Was Working Register)"**
(VIDEO 5, ep1). 88 frames, 6:32 runtime, B&W inkwash + red accent stick-figure animation,
AndrewNeural narration, word-locked cuts, deliverable zip. Use this as the SOP for the next
video.

---

## 0. Repo & branch

- Repo: `AsuraXP/CartoonStyle` on GitHub.
- Work on a branch named `arena/<hash>-<slug>` (Arena creates this automatically per session).
- All generated videos live under `examples/<video_slug>_video/`
  (e.g. `examples/apoc_day1_video/`). Slug is lowercase + underscores, e.g. `apoc_day1`.
- **Never delete/rename the repo root or `.git/`.** Commit+push every accepted batch to the
  working branch; never push to other branches.

---

## 1. Script & metadata (first, before any images)

Create the video folder:

```
examples/<slug>_video/
  script.md       # full narration prose, beats separated by "---", ~1200–1400 words
                  #   target runtime 6:00–7:00. Target ~5–7 words per visual moment.
  metadata.json   # {title, description, tags} — YouTube-ready.
  youtube.txt     # TITLE / DESCRIPTION / TAGS block, copy-paste ready.
  PROMPT_RULES.md # style rules / hard constraints discovered during style_tests.
  CONTINUITY.md   # (optional) set/prop/character consistency notes per beat.
  thumbnail.jpg   # final poster frame (can come later, but placeholder early).
  visuals.csv     # columns: slide_id,moment,moment_text,visual_prompt
                  # one row per frame (= one image cut). Written manually or via
                  # `python -m cartoonstyle.storyboard` then hand-edited.
```

**Voice is fixed per video from the start** — e.g. `en-US-AndrewNeural +0%`. Record it in
`PROMPT_RULES.md` so every future turn remembers it.

---

## 2. Style anchor tests (before batch generation)

Before generating story frames, produce **~10–15 style test refs** in `style_tests/`:
MC close-up, zombie horde, wide exterior, interior, dark scene, key props. Use these to:

1. Land the exact art style (palette, line weight, stylization).
2. Discover moderation-triggering words (e.g. "kick"/"slam"/"blood-covered") and their
   safe replacements ("guiding the bar down", "pressed against glass", "flat red splotch").
3. Discover composition bugs (e.g. zombies-on-outside required the glass-side rule — see §4).
4. Lock character/prop consistency baselines.

QC each test, re-roll broken ones, winners become anchors referenced in later prompts as
`images=[body_ref, location_ref, optional_approved_story_frame]`.

Commit+pwd the winners. Name them descriptively (`mc_portrait.png`, `zombie_horde_wide.png`,
`cooler_silhouette.png`, etc.).

---

## 3. Frame generation (batched, with strict QC)

**Per-turn image generation cap is 10** — never ask for more than 8 in one parallel batch.

### Prompt recipe (proven)

```
[COMPOSITION + CAMERA] Style A B&W ink-wash with red accent: [SCENE DESCRIPTION].
[STORY-STATE RULES — glass intact/cracked, door open/closed, MC inside/zombies outside].
[CHARACTER RULES — MC = red t-shirt stick figure, no pants; zombies = round white heads,
 thin black stick limbs, milky white dot eyes, ragged jagged shirt edges, small flat red
 blood splotches, NO grey skulls, NO teeth close-ups, NO pants/shoes/boots].
[REFERENCE] images=[body_ref, location_ref, prior_approved_anchor_if_any].
```

### Glass-side rule (critical for exterior shots of zombies pressing glass)

> Camera is OUTSIDE on the sidewalk, we see ZOMBIES FROM BEHIND, their BACKS to camera,
> their stick feet on OUR sidewalk, their hands pressed to the EXTERIOR of the glass,
> store interior VISIBLE THROUGH the glass behind them.

This reliably keeps zombies outside. Without this phrase, diffusion frequently puts them
inside the store.

### Batch loop (repeat until every visuals.csv row has a frame)

1. Pick up to **8 rows** from visuals.csv that aren't yet generated.
2. Generate in parallel with `generate_image` (or `offer_options=True` only when it's a
   hero frame you want to choose between).
3. **QC each frame immediately** against two standards:
   - **STORY BREAK** → re-roll. (Wrong side of glass, MC missing red shirt, zombies with
     pants/skulls/teeth, door state contradicting the beat, unreadable composition.)
   - **Compositional nitpick but story still reads** → ACCEPT. Set/prop layout, brick
     pattern, pump positions, cartoon stretch-limbs on action shots are LOW PRIORITY —
     never re-roll for these.
4. Save winners to `frames/beat{slide:02d}m{moment:02d}.png` matching visuals.csv exactly.
5. Every 8–16 winning frames, **commit + push** to the working branch with a descriptive
   message (e.g. "Beat 24–28: cooler reveal, bat closeup, EBS montage").

### Intentionally-skipped frames

If a planned frame is a composition trap (e.g. teeth close-ups that drift into monster
horror) AND the beat is already covered by another frame in the sequence, you may skip it.
Note the skip in session memory so stitch warnings are expected. For Video 5: beat15m02
(faces squished teeth against glass) was skipped; covered by beat15m01 wall-of-faces +
beat19m02 silhouettes.

### Anchor accumulation

As you QC, promote frames that lock a hard location/prop/expression to "anchor" status.
The anchor list grows as the video progresses. Major Video 5 anchors that should be
referenced for their scene types in future videos:

- `beat06m01` — aerial sea-of-stick-zombies wide
- `beat07m01` — exterior horde-at-door (zombie-backs composition)
- `beat12m03` — zombie face close-up (oval mouth, no teeth)
- `beat18m01` — dark interior with headlight beams
- `beat22m02` — radio static→clear tuning (clean UI)
- `beat23m02` — EBS minimalist 3-panel icon montage
- `beat24m02` — 90-second clock infographic
- `beat26m03` — one-point-perspective back aisle with cooler at end
- `beat27m01` — "WORLD'S OKAYEST MANAGER" bat prop close-up
- `beat28m01` — cooler silhouette reveal
- `beat30m02` — engaged metal latch w/ red indicator
- `beat30m03` — MC vs cooler standoff
- `beat35m01` — final aerial pull-back (lone red dot in black sea) — iconic end-wide

---

## 4. Commit cadence

- **Don't let uncommitted good work pile up.** Every 8–16 verified frames:
  ```
  git add examples/<slug>_video/frames/
  git commit -m "<slug>: beats X-Y through <story beat>"
  git push origin <working-branch>
  ```
- Also commit new anchors, script/metadata tweaks, and pipeline fixes.
- At the end of frame production, make a final "COMPLETE: N frames through <end card>" commit.

---

## 5. Assembly (Kaggle, not in the sandbox)

Why Kaggle: the sandbox only allows outbound to github/pypi/etc; TTS (edge-tts calls
speech.platform.bing.com) and font downloads need full internet, and video encode is
faster there. The sandbox is for frame generation + commits only.

### Step 5a — push the runner script and pipeline code

Before asking the user to run on Kaggle, make sure:

- `run_<slug>_kaggle.sh` exists in the repo root, modeled on `run_apoc_kaggle.sh`.
- `cartoonstyle/{csv_to_beats,tts,stitch}.py` are in their working state.
- `frames/` has all PNGs committed+pushed.
- `thumbnail.jpg`, `metadata.json`, `script.md`, `youtube.txt` are in the video folder.

### Step 5b — user runs on Kaggle (one cell)

```python
!rm -rf CartoonStyle img_pool
!git clone --depth 1 --branch <working-branch> https://github.com/AsuraXP/CartoonStyle.git
!bash CartoonStyle/run_<slug>_kaggle.sh
```

The runner does 5 things:

1. Installs `ffmpeg`, `edge-tts`, `pillow`, `requests`.
2. Clones/updates the repo from the working branch.
3. `python -m cartoonstyle.csv_to_beats --visuals ... --out beats.jsonl`
   — groups visuals.csv rows into beats with placeholder durations.
4. Makes a `img_pool/` directory of symlinks `beatNNmMM.png -> beatNNmMM_v00_s00000.png`
   because stitch.py expects the generate-flux variant naming convention.
5. `python -m cartoonstyle.tts` (one continuous AndrewNeural +0% take via edge-tts,
   WordBoundary word timings → `narration.mp3`, `beats_with_timing.json`,
   `moments_timing.json` — moments tiled gapless across the audio).
6. `python -m cartoonstyle.stitch` (native resolution, no resize; pre-pads every frame
   to a uniform even canvas with PIL so ffmpeg concat doesn't choke; x264 + AAC,
   -14 LUFS loudnorm, 30fps, writes VIDEO.mp4).
7. Zips `VIDEO.mp4` + `thumbnail.jpg` + `metadata.json` + `script.md` + `youtube.txt`
   into `/kaggle/working/<slug>_DELIVERABLE.zip`.

### Step 5c — hard-won stitch/ffmpeg lessons

- **Never apply `-vf pad=...` after the concat demuxer** — triggers exit status 234.
  Always normalize frame dimensions to a uniform even canvas **with PIL first**, then
  concat filter-free.
- **x264 requires even width AND height** — if your generated frames are odd-dimensioned
  (e.g. 1365x768), PIL-pad to 1366x768 with a 1px black strip on the right/bottom.
- **Don't resize frames** — keeps render fast and avoids upscaling artifacts. Just pad.
- **Default captions to OFF for the first clean mux.** If you add captions later, use
  the PIL burner in stitch.py (not libass/ASS — libass segfaults on Kaggle's ffmpeg build
  when the ASS Fontname doesn't exactly match the font's internal family name; Inter
  shipped as `Inter-Bold.otf` via apt is especially prone to this). PIL burner draws
  white text with black shadow on a black translucent bar directly onto each frame.
- **Save PIL-processed frames with `optimize=False, compress_level=6`** — Pillow 10.x has
  an `_idat` crash when saving PNGs with `optimize=True` after an RGBA-paste on RGB.
- **Concatenation demuxer quirk:** the last `file` entry must be listed twice (once with
  duration, once bare) or ffmpeg drops the final frame. The existing `_write_concat`
  helper handles this — don't hand-roll the concat list.

### Step 5d — fixing Kaggle failures

- TTS failures: usually transient Bing endpoint errors; rerun. `narration.mp3` is cached
  when present unless `--force`.
- "no image for beatXXmYY - skipped": benign if that frame was intentionally skipped
  (we only did this for beat15m02). Don't treat as failure.
- Encode failures: almost always dimension mismatch → make sure `normalize()` ran and all
  input PNGs are exactly WxH.

---

## 6. Deliverable

Final zip (produced by the runner) contains exactly:

| File | Purpose |
|---|---|
| `VIDEO.mp4` | Final video, native resolution, 30fps, -14 LUFS, AAC 192kbps |
| `thumbnail.jpg` | Poster frame for YouTube/TikTok |
| `metadata.json` | `{title, description, tags[]}` for API uploads |
| `script.md` | Full narration script (for subtitles, blogs, etc.) |
| `youtube.txt` | Copy-paste TITLE / DESCRIPTION (with chapter timestamps) / TAGS block |

If the user wants captions baked in, flip `--captions --caption-method pil` on stitch,
add the font acquisition block back into the runner, and re-encode.

---

## 7. Standing constraints (write these into PROMPT_RULES.md for each new video)

- Zombies (or whatever the "monster" is): **stick figures always** — round white heads,
  thin black stick limbs, milky white empty dot eyes, ragged jagged shirt edges, small
  stylized flat red blood blotches. NEVER grey skulls, teeth close-ups, or pants/shoes/
  boots on any figure. Re-roll any frame that breaks this.
- MC: **red t-shirt (flat red fill), thin black stick limbs, no pants.** Single red accent.
- Violence verbs ("kick", "slam", "impact", "blood-covered") trigger moderation — use
  softer phrasing ("lowering the bar", "guiding the bar down into brackets", "pressed
  against glass, impact burst lines", "flat red mark/splotch").
- **Set/prop consistency is LOW PRIORITY** — do NOT re-roll for shelf layout, brick
  pattern, pump positions. Only re-roll for STORY BREAKS or unreadable compositions.
- Cartoon stretch-limbs on motion frames are acceptable.
- **Story-state rules must be explicit in every prompt**: glass intact/cracked, door
  open/closed, MC inside / monsters outside.
- **Per-turn image cap is 10; batch ≤ 8.** Don't ask for more.
- Approved production path: **batched generation with strict QC, re-roll ~10–40%
  compositional failures, winners become anchor refs for later prompts.**

---

## 8. Handy repo commands

```bash
# Per-video runner template lives at repo root:
cp run_apoc_kaggle.sh run_<slug>_kaggle.sh   # then edit VIDEO_DIR and metadata paths

# Local-only pipeline pieces (TTS needs outbound to Bing so doesn't work in sandbox,
# but csv_to_beats and PIL pre-normalize are testable locally):
python -m cartoonstyle.csv_to_beats --visuals examples/X/visuals.csv --out examples/X/beats.jsonl

# Git (always):
git add examples/<slug>_video/frames/   # or whatever new files
git commit -m "<slug>: <what>"
git push origin <working-branch>
```

---

## 9. When starting a new video in a fresh conversation

1. Give the agent this file (WORKFLOW.md) as the first thing.
2. Specify: video slug, episode/short concept, target runtime, narrator voice.
3. Let the agent build out `examples/<slug>_video/` starting with script.md + metadata.json,
   then visuals.csv, then style_tests/, then batched frames, then the kaggle runner script,
   then hand you the single-cell Kaggle command.
4. The agent should never attempt full video assembly in the sandbox — sandbox is for
   image generation, commits, and writing code. Final encode runs on Kaggle.
