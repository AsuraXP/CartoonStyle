#!/bin/bash
# One-shot runner for Napoleon Complex video.
# Usage: bash run_napoleon.sh
# Native frame resolution (1376x768), zero stretching/cropping/letterboxing.
# Stops on first error and prints what went wrong.
set -e

VIDEO_DIR="examples/napoleon_complex_video"
FRAMES="$VIDEO_DIR/frames"

echo "=== [1/5] Installing dependencies ==="
pip install -q edge-tts
which ffmpeg >/dev/null || (apt-get -qq update && apt-get -qq install -y ffmpeg)

echo "=== [2/5] CSV -> beats.jsonl ==="
python -m cartoonstyle.csv_to_beats --visuals "$VIDEO_DIR/visuals.csv" --out "$VIDEO_DIR/beats.jsonl"

echo "=== [3/5] TTS (AndrewNeural -5%) ==="
python -m cartoonstyle.tts \
    --beats "$VIDEO_DIR/beats.jsonl" \
    --out-dir "$VIDEO_DIR" \
    --voice en-US-AndrewNeural \
    --rate=-5% \
    --visuals "$VIDEO_DIR/visuals.csv" 2>&1 | tee /tmp/tts_log.txt

# Safety check
if ! grep -q "moments locked.*\[OK\]" /tmp/tts_log.txt; then
    echo ""
    echo "!!! TTS DID NOT LOCK ALL MOMENTS — stopping here. !!!"
    echo "!!! Paste the output above back to the assistant. !!!"
    exit 1
fi

echo "=== [4/5] Stitching VIDEO.mp4 (native res, no scaling) ==="
python -m cartoonstyle.stitch \
    --timing "$VIDEO_DIR/beats_with_timing.json" \
    --images-dir "$FRAMES" \
    --out "$VIDEO_DIR/VIDEO.mp4" \
    --pick 0

echo "=== [5/5] Packaging deliverable zip ==="
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
