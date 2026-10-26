#!/bin/bash
# =============================================================================
# VIDEO 5 — Day 1 of the Zombie Apocalypse (I Was Working Register)
# One-shot Kaggle build: pulls repo -> TTS (AndrewNeural) -> word-locked stitch
# (native resolution, no captions) -> deliverable zip.
#
# USAGE (Kaggle notebook cell):
#   !bash CartoonStyle/run_apoc_kaggle.sh
# Output: /kaggle/working/apoc_day1_DELIVERABLE.zip
# =============================================================================
set -e

WORK=/kaggle/working
REPO=$WORK/CartoonStyle
VIDEO_DIR=$REPO/examples/apoc_day1_video
FRAMES=$VIDEO_DIR/frames

echo "=== [1/5] Grabbing ffmpeg + pip deps ==="
if ! which ffmpeg >/dev/null 2>&1; then
  apt-get -qq update
  apt-get -qq install -y ffmpeg
fi
pip install -q edge-tts pillow requests

echo "=== [2/5] Pulling CartoonStyle repo (frames + script + metadata) ==="
cd $WORK
if [ ! -d "$REPO" ]; then
  git clone --depth 1 --branch arena/22c9cd35-cartoonstyle https://github.com/AsuraXP/CartoonStyle.git
fi
cd $REPO
git pull --ff-only origin arena/22c9cd35-cartoonstyle || true

echo "=== [3/5] CSV -> beats.jsonl ==="
python -m cartoonstyle.csv_to_beats \
    --visuals "$VIDEO_DIR/visuals.csv" \
    --out "$VIDEO_DIR/beats.jsonl"

# Stitch expects filenames of the form beatNNmMM_v00_sSEED.png — our frames
# are beatNNmMM.png, so alias each via symlinks in a sibling directory.
echo "=== [4/5] Prepping v00 symlinks for stitch ==="
IMG_POOL=$WORK/img_pool
rm -rf "$IMG_POOL"; mkdir -p "$IMG_POOL"
for f in "$FRAMES"/beat*.png; do
  base=$(basename "$f" .png)
  ln -sf "$f" "$IMG_POOL/${base}_v00_s00000.png"
done
echo "   $(ls "$IMG_POOL" | wc -l) frame aliases ready in $IMG_POOL"

echo "=== [5/5] TTS + stitch (AndrewNeural +0%, native resolution, no captions) ==="
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

python -m cartoonstyle.stitch \
    --timing "$VIDEO_DIR/beats_with_timing.json" \
    --images-dir "$IMG_POOL" \
    --out "$VIDEO_DIR/VIDEO.mp4" \
    --pick 0

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
