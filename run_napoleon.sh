#!/bin/bash
# One-shot runner for Napoleon Complex video.
# Usage: bash run_napoleon.sh
# Stops on first error and prints what went wrong.
set -e

VIDEO_DIR="examples/napoleon_complex_video"
FRAMES="$VIDEO_DIR/frames"

echo "=== [1/6] Installing dependencies ==="
pip install -q edge-tts
which ffmpeg >/dev/null || (apt-get -qq update && apt-get -qq install -y ffmpeg imagemagick)

echo "=== [2/6] Resizing all frames to 1920x1080 ==="
mogrify -resize 1920x1080! "$FRAMES"/*.png

echo "=== [3/6] CSV -> beats.jsonl ==="
python -m cartoonstyle.csv_to_beats "$VIDEO_DIR/visuals.csv" "$VIDEO_DIR/beats.jsonl"

echo "=== [4/6] TTS (AndrewNeural -5%) ==="
python -m cartoonstyle.tts \
    "$VIDEO_DIR/beats.jsonl" \
    "$VIDEO_DIR/narration.mp3" \
    --voice en-US-AndrewNeural \
    --rate=-5% \
    --fill="$FRAMES" 2>&1 | tee /tmp/tts_log.txt

# Check for the magic [OK] line before proceeding
if ! grep -q "\[OK\] 120/120 moments locked" /tmp/tts_log.txt; then
    echo ""
    echo "!!! TTS DID NOT LOCK ALL 120 MOMENTS — stopping here. !!!"
    echo "!!! Paste the output above back to the assistant.     !!!"
    exit 1
fi

echo "=== [5/6] Stitching VIDEO.mp4 ==="
python -m cartoonstyle.stitch \
    "$VIDEO_DIR/beats.jsonl" \
    "$FRAMES" \
    "$VIDEO_DIR/narration.mp3" \
    "$VIDEO_DIR/VIDEO.mp4"

echo "=== [6/6] Packaging deliverable zip ==="
rm -f napoleon_complex_DELIVERABLE.zip
zip -j napoleon_complex_DELIVERABLE.zip \
    "$VIDEO_DIR/VIDEO.mp4" \
    "$VIDEO_DIR/thumbnail.jpg" \
    "$VIDEO_DIR/metadata.json" \
    "$VIDEO_DIR/script.md"

echo ""
echo "========================================="
echo "✅ DONE. Deliverable:"
ls -lh napoleon_complex_DELIVERABLE.zip
echo "========================================="
