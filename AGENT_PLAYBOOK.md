# CartoonStyle Pipeline — Agent Handoff Playbook
How this repo turns a script into a CGP-Grey-style stick-figure explainer video with word-perfect image-to-narration sync.

---

## 1. Repo & Branch

- Repo: `AsuraXP/CartoonStyle` on GitHub
- Working branch pattern: `arena/<session>-<topic>` (each session gets its own branch)
- Key files in the working codebase (commit-lineage tested working):
  - `cartoonstyle/tts.py` — one-take edge-tts with WordBoundary word timings, matches images to spoken words sequentially
  - `cartoonstyle/stitch.py` — ffmpeg concat demuxer, word-locked hard cuts (no Ken Burns, no motion — user explicitly rejected that)
  - `cartoonstyle/csv_to_beats.py` — converts visuals.csv → beats.jsonl

## 2. Video Structure

- **Style**: flat-color CGP-Grey minimalist stick figures, thick black outlines, solid flat color fills, white backgrounds, 2-dot eyes, cel-shaded. NO gradients, NO photorealism, NO captions/watermarks on final video.
- **Voice**: single narrator, `en-US-AndrewNeural` edge-tts voice at `-5%` rate.
- **Pacing target**: ~5–5:30 minutes, ~100–120 moments, **avg 7–9 words per moment** (≈2–3 sec hard cuts per frame). More frames = tighter pacing. User rejected Ken Burns zoompan as a pacing fix — always use MORE frames, not motion.
- **Resolution**: use the image generator's native output resolution. Do NOT scale/crop/letterbox in stitch — ffmpeg concat-demuxer works fine as long as ALL FRAMES ARE IDENTICAL DIMENSION. (AI outputs ~1376x768 consistently, which is fine.)
- **Output**: MP4 via ffmpeg libx264, yuv420p, 30fps, aac 192k audio.

## 3. File Layout (per video)

```
examples/<video_name>/
├── script.md         # plain paragraph narration, no markdown headings beyond title
├── visuals.csv       # columns: slide_id, moment, moment_text, visual_prompt
├── metadata.json     # title + description + tags
├── youtube.txt       # clean copy-paste title/desc/tags for YouTube (user prefers this over JSON)
├── thumbnail.jpg     # YouTube thumbnail
├── STATUS.md         # Kaggle run instructions + research sources
└── frames/           # PNGs named beatNNmMm_v00_s00000000.png
```

## 4. Writing visuals.csv

- **slide_id**: beat number (paragraph/section), 1-indexed
- **moment**: sequential image number within the beat, 1-indexed
- **moment_text**: the exact words spoken while this image is on screen (used for TTS sync)
- **visual_prompt**: image generation prompt for that frame

Rules:
- Split narration into moments at natural phrase breaks. Aim for **7–9 words per moment**.
- The full text of moment_text across moments in a beat, concatenated with spaces, must equal the script paragraph verbatim (no extra words, no missing words, identical punctuation) — this is what the token-matcher aligns against.
- Each moment = exactly one frame. Hard cuts. No transitions.
- Example: "You have a picture of Napoleon in your head right now." splits into:
  - moment 1: "You have a picture of Napoleon" (think bubble forming)
  - moment 2: "in your head right now." (silhouette appears in bubble)

## 5. Image Generation (the part that takes 10+ turns)

**Hard limit**: `generate_image` tool returns **10 images per turn MAX**. Hit 10, commit+push, tell user "continue", next turn do another 10. There is no way around this per-turn cap.

**Process:**
1. **Anchor frame first**: generate beat01m01 (first frame) using the previous video's first frame as a style reference image (`images=[...]` arg). This locks in the art style. For subsequent frames, always pass beat01m01 as the `images` reference so style stays consistent across all 120 frames.
2. **Batch size**: exactly 10 parallel `generate_image` calls per turn.
3. **Naming**: `frames/beat{slide_id:02d}m{moment:02d}_v00_s00000000.png`. stitch.py looks for this exact `_v{pick:02d}_s*.png` pattern.
4. **Prompt style**: every prompt must begin with "Follow reference EXACT art style: thick bold black outlines, SOLID FLAT COLOR FILLS, cel-shaded." Then describe the scene simply and concretely. End with "16:9." (or "16:9 widescreen.")
5. **Offer options**: always pass `offer_options=false` — user wants production speed, not choice paralysis.
6. After each successful batch of 10: `git add frames/ && git commit -m "<topic> batch N: description (N/120=X%)" && git push`.
7. If an image fails (model error, "no images" response, etc.), just retry it next turn with a slightly reworded shorter prompt. Don't stop the whole pipeline.
8. After all frames generated: write `STATUS.md` with Kaggle instructions + research sources. Write `youtube.txt` with clean title/description/tags.

**Important prompt gotchas**:
- Do NOT prompt for specific resolutions like "1920x1080" — the model doesn't honor that exactly. It consistently outputs ~1376x768, which is close enough (0.8% off 16:9 — invisible).
- Stick figures: "round head, two black dot eyes" produces the CGP Grey look. Avoid "cartoon" alone — specify "stick figure".
- Avoid complex multi-scene prompts. One clear subject per frame beats busy compositions.

## 6. TTS Sync (the "magic" part)

`cartoonstyle/tts.py` does this:
1. Feeds the ENTIRE script as one continuous string to edge-tts (AndrewNeural) — natural prosody, no beat-to-beat gluing.
2. Listens to edge-tts WordBoundary events which give exact start/end times for each spoken word.
3. Matches moment_text sequences SEQUENTIALLY against the spoken word stream: for each moment, advance through the word stream until all of moment_text's tokens have been consumed; that span becomes the frame's on-screen duration.
4. Writes `moments_timing.json` with per-frame start/sec and prints a status line.

**CRITICAL TOKEN RULE (this was the biggest bug we hit):**
edge-tts emits numerals as RAW DIGIT STRINGS, not as expanded words:
- "1980s" → WordBoundary text is "1980s" (NOT "nineteen eighties")
- "1769" → "1769" (NOT "seventeen sixty nine")
- "5'6" → "5'6"
- "220" → "220"

So the tokenizer in tts.py must use a SIMPLE regex like `re.findall(r"[a-z0-9]+", text.lower())` — NO number-to-word expansion. Expanding numbers to words BREAKS matching (commit `f105be8` was wrong; fixed in `5a9878d`). If you see X/120 moments missed, 9 times out of 10 it's because a token in the script isn't matching what edge-tts actually emits.

**Safety check**: after TTS, look for a line like:
`[tts] 120 image moments locked to spoken words -> moments_timing.json [OK]`
If you don't see `[OK]` with all moments accounted for, stop and inspect the debug JSONL dump for the first missed moment.

**CLI flags (these tripped us up twice — use exact args):**
```bash
python -m cartoonstyle.csv_to_beats --visuals visuals.csv --out beats.jsonl

python -m cartoonstyle.tts \
  --beats beats.jsonl \
  --out-dir . \
  --voice en-US-AndrewNeural \
  --rate=-5% \
  --visuals visuals.csv
```
Note: tts takes `--out-dir` (a directory) not an mp3 path. It writes `narration.mp3` and `beats_with_timing.json` and `moments_timing.json` into that dir.

## 7. Stitching

```bash
python -m cartoonstyle.stitch \
  --timing beats_with_timing.json \
  --images-dir frames/ \
  --out VIDEO.mp4 \
  --pick 0
```

- `--pick 0` selects the `_v00_` variant (we only generate v00, never alternate variants).
- stitch reads `moments_timing.json` automatically from the same dir as `--timing`.
- It builds an ffmpeg concat-demuxer list where each frame appears for its spoken-word duration, then encodes with libx264 + aac audio.
- **DO NOT add any -vf scale/crop/pad filter** to stitch.py. The user wants native resolution with zero distortion. ffmpeg will fail if frames are different sizes, so verify all frames are same dimensions first (Python PIL one-liner in the runner handles this).
- No captions (user said no captions/watermarks).
- No motion/zoompan/Ken Burns (explicitly rejected by user). Hard cuts only.

## 8. The One-Shot Kaggle Runner Pattern

Provide a `run_<topic>.sh` script so user only needs one cell:

```bash
!rm -rf CartoonStyle && \
  git clone -b <branch> https://github.com/AsuraXP/CartoonStyle.git && \
  cd CartoonStyle && \
  bash run_<topic>.sh

from IPython.display import FileLink
FileLink('CartoonStyle/<topic>_DELIVERABLE.zip')
```

The bash script must:
1. `set -e` (stop on first error)
2. `pip install -q edge-tts`
3. `apt-get install -y ffmpeg` (imagemagick NOT needed — no resize step)
4. csv_to_beats (correct CLI flags!)
5. tts (correct CLI flags!) — then `grep "moments locked.*\[OK\]"` on the log and exit 1 with a clear error if not found
6. stitch (correct CLI flags!)
7. zip deliverable (VIDEO.mp4 + thumbnail.jpg + metadata.json + script.md)
8. Print ✅ DONE with ls -lh of the zip

Test the grep for [OK] against actual tts.py output format — tts prints `[OK]` at the END of the line: `"...moments_timing.json [OK]"`, not `"[OK] 120/120 moments locked"` (that was our false-alarm bug).

## 9. Research & Sources

- User asked for research sources to be tracked for video description.
- Use `web_search` (depth 2 or 3) to verify facts before writing the script.
- Save source URLs + one-line takeaways either in STATUS.md or directly in youtube.txt description under a "Sources:" section.

## 10. Common Mistakes We Made (don't repeat)

| Mistake | Fix |
|---|---|
| Expanding decades/numbers to words in norm_words | Use simple `[a-z0-9]+` regex, leave digits alone |
| Wrong CLI flags (`csv_to_beats.py` takes --visuals/--out, not positional) | Read argparse in each .py before writing the runner |
| Grep check for `[OK]` matched wrong string | Grep for `moments locked.*\[OK\]` (OK is at end of line) |
| Added `-vf scale=1920:1080` thinking it was needed | DON'T — native res only, user rejected scaling |
| Tried Ken Burns / ffmpeg zoompan to fix pacing | User said NO — use more frames instead |
| Tried programmatic character animation (eye blinks, rigs, AnimateDiff) | User explicitly declined — "would take too much time and won't be that good" |
| `mogrify` not installed on Kaggle (imagemagick missing) | Don't rely on imagemagick at all; native res, no resize step |
| Offer options slowing things down with image choices | Always pass `offer_options=false` in generate_image |
| 10-image-per-turn cap not respected | Exactly 10 generate_image calls per turn, then commit+push+ask "continue" |
| Forgot `FileLink` at end of Kaggle cell | User needs the clickable download link |

## 11. Git Workflow

- Always work on the session's assigned branch (e.g. `arena/<hash>-<topic>`).
- Commit after every batch of 10 frames with descriptive message `"<topic> batch N: description (K/TOTAL=P%)"`.
- Push after every commit: `git push origin <branch>`.
- Final commit message when all frames done: `"<topic>: ALL N FRAMES COMPLETE + STATUS.md + youtube.txt ready for Kaggle pipeline"`.
- Tag working known-good commits in session memory so you can reset if something breaks.

## 12. What a "done" video looks like

- 120 PNG frames in frames/ (one per moment in visuals.csv)
- beats.jsonl, beats_with_timing.json, moments_timing.json, narration.mp3 produced by TTS with `[OK]` status
- VIDEO.mp4 stitched, same resolution as source frames, plays start-to-finish with word-perfect image cuts
- Deliverable zip contains: VIDEO.mp4, thumbnail.jpg, metadata.json, script.md (and optionally youtube.txt)
- TTS runtime: ~30 sec edge-tts
- Stitch runtime: 3–5 min ffmpeg encode on Kaggle CPU
- Final video: ~5:00–5:30, ~30–60 MB MP4
