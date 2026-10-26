#!/bin/bash
# =============================================================================
# VIDEO 5 — Day 1 of the Zombie Apocalypse (I Was Working Register)
# One-shot Kaggle build: pulls repo -> TTS (AndrewNeural) -> word-locked
# captions in Inter Bold (Apple-style) -> MP4 -> deliverable zip.
#
# USAGE (Kaggle notebook cell):
#   !bash run_apoc_kaggle.sh
# Output ends up in /kaggle/working/apoc_day1_DELIVERABLE.zip
# =============================================================================
set -e

WORK=/kaggle/working
REPO=$WORK/CartoonStyle
VIDEO_DIR=$REPO/examples/apoc_day1_video
FRAMES=$VIDEO_DIR/frames

echo "=== [1/7] Grabbing ffmpeg + pip deps ==="
if ! which ffmpeg >/dev/null 2>&1; then
  apt-get -qq update
  apt-get -qq install -y ffmpeg
fi
pip install -q edge-tts pillow requests

echo "=== [2/7] Pulling CartoonStyle repo (frames + script + metadata) ==="
cd $WORK
if [ ! -d "$REPO" ]; then
  git clone --depth 1 --branch arena/22c9cd35-cartoonstyle https://github.com/AsuraXP/CartoonStyle.git
fi
cd $REPO
git pull --ff-only origin arena/22c9cd35-cartoonstyle || true

echo "=== [3/7] Acquiring caption font (Apple-style bold sans) ==="
FONT_PATH=$WORK/CaptionFont.ttf
get_font_size() { stat -c%s "$1" 2>/dev/null || echo 0; }
if [ ! -f "$FONT_PATH" ] || [ "$(get_font_size "$FONT_PATH")" -lt 50000 ]; then
  # 1) Try apt fonts-inter package (Ubuntu/Kaggle/Debian)
  apt-get -qq install -y fonts-inter fonts-dejavu-core 2>/dev/null || true
  fc-cache -f 2>/dev/null || true
  # Search known install locations for Inter Bold, then any Inter bold-ish variant
  FOUND=""
  for cand in \
    /usr/share/fonts/truetype/inter/Inter-Bold.ttf \
    /usr/share/fonts/truetype/Inter/Inter-Bold.ttf \
    /usr/share/fonts/truetype/Inter-Bold.ttf \
    /usr/share/fonts/opentype/inter/Inter-Bold.otf \
    /usr/share/fonts/truetype/inter/Inter-Bold.ttf \
    /usr/share/fonts/Inter-Bold.ttf ; do
    if [ -f "$cand" ]; then FOUND="$cand"; break; fi
  done
  # Glob fallback — find any Inter Bold file
  if [ -z "$FOUND" ]; then
    FOUND=$(find /usr/share/fonts -iname "*inter*bold*" 2>/dev/null | head -1)
  fi
  # 2) Try downloading Inter Bold if apt didn't deliver it
  if [ -z "$FOUND" ] || [ "$(get_font_size "$FOUND")" -lt 50000 ]; then
    TMP_FONT=$WORK/_font_dl.ttf
    for url in \
      "https://raw.githubusercontent.com/rsms/inter/master/docs/font-files/Inter-Bold.ttf" \
      "https://raw.githubusercontent.com/google/fonts/main/ofl/inter/static/Inter-Bold.ttf" ; do
      if curl -fsSL "$url" -o "$TMP_FONT" && \
         python3 -c "from PIL import ImageFont; ImageFont.truetype('$TMP_FONT',48)" 2>/dev/null && \
         [ "$(get_font_size "$TMP_FONT")" -gt 50000 ]; then
        FOUND="$TMP_FONT"; break
      fi
      rm -f "$TMP_FONT"
    done
  fi
  # 3) Fallback: any bold sans on the system (DejaVu, Liberation, Noto, etc.)
  if [ -z "$FOUND" ] || [ "$(get_font_size "$FOUND")" -lt 50000 ]; then
    for cand in \
      /usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf \
      /usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf \
      /usr/share/fonts/truetype/noto/NotoSans-Bold.ttf \
      /usr/share/fonts/truetype/freefont/FreeSans-Bold.ttf ; do
      if [ -f "$cand" ]; then FOUND="$cand"; break; fi
    done
    if [ -z "$FOUND" ]; then
      FOUND=$(find /usr/share/fonts -iname "*sans*bold*" -name "*.ttf" 2>/dev/null | head -1)
    fi
  fi
  if [ -n "$FOUND" ] && [ -f "$FOUND" ]; then
    cp "$FOUND" "$FONT_PATH"
  fi
fi
if [ ! -f "$FONT_PATH" ] || [ "$(get_font_size "$FONT_PATH")" -lt 50000 ]; then
  echo "   !! WARNING: no bold sans font found; captions will use ffmpeg default"
  FONT_PATH=""
else
  echo "   caption font -> $FONT_PATH ($(get_font_size $FONT_PATH) bytes)"
fi

echo "=== [4/7] CSV -> beats.jsonl ==="
python -m cartoonstyle.csv_to_beats \
    --visuals "$VIDEO_DIR/visuals.csv" \
    --out "$VIDEO_DIR/beats.jsonl"

# Stitch looks for filenames of the form beatNNmMM_v00_sSEED.png — our frames
# are named beatNNmMM.png, so we alias each frame to v00_s00000 via a sibling
# directory of symlinks. This keeps the original frames untouched.
echo "=== [5/7] Prepping v00 aliases for stitch ==="
IMG_POOL=$WORK/img_pool
rm -rf "$IMG_POOL"; mkdir -p "$IMG_POOL"
cd "$IMG_POOL"
for f in "$FRAMES"/beat*.png; do
  base=$(basename "$f" .png)
  ln -sf "$f" "${base}_v00_s00000.png"
done
echo "   $(ls | wc -l) frame aliases ready in $IMG_POOL"

echo "=== [6/7] TTS (AndrewNeural +0%) with word-boundary timings ==="
cd "$REPO"
python -m cartoonstyle.tts \
    --beats "$VIDEO_DIR/beats.jsonl" \
    --out-dir "$VIDEO_DIR" \
    --voice en-US-AndrewNeural \
    --rate=+0% \
    --visuals "$VIDEO_DIR/visuals.csv" 2>&1 | tee /tmp/tts_log.txt

if ! grep -qE "moments locked.*\[OK\]|moments locked.*\[.*MISSED" /tmp/tts_log.txt; then
  echo ""
  echo "!!! TTS DID NOT LOCK ALL MOMENTS — check /tmp/tts_log.txt !!!"
  exit 1
fi

echo "=== [7/7] Stitching VIDEO.mp4 with Apple-style Inter Bold captions ==="
python -m cartoonstyle.stitch \
    --timing "$VIDEO_DIR/beats_with_timing.json" \
    --images-dir "$IMG_POOL" \
    --out "$VIDEO_DIR/VIDEO.mp4" \
    --pick 0 \
    --captions \
    --font "$FONT_PATH"

echo ""
echo "=== Packaging deliverable zip ==="
cd $WORK
rm -f apoc_day1_DELIVERABLE.zip
zip -j apoc_day1_DELIVERABLE.zip \
    "$VIDEO_DIR/VIDEO.mp4" \
    "$VIDEO_DIR/thumbnail.jpg" \
    "$VIDEO_DIR/metadata.json" \
    "$VIDEO_DIR/script.md" \
    "$VIDEO_DIR/youtube.txt"

echo ""
echo "==========================================="
ls -lh apoc_day1_DELIVERABLE.zip
echo ""
echo "Contents:"
unzip -l apoc_day1_DELIVERABLE.zip
echo "==========================================="
