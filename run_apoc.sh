#!/bin/bash
# Runner for Day 1 of the Zombie Apocalypse (series ep1).
# Native frame resolution, zero stretching/cropping/letterboxing.
set -e

VIDEO_DIR="examples/apoc_day1_video"
FRAMES="$VIDEO_DIR/frames"

echo "=== [1/5] Installing dependencies ==="
pip install -q edge-tts
which ffmpeg >/dev/null || (apt-get -qq update && apt-get -qq install -y ffmpeg)

echo "=== [2/5] CSV -> beats.jsonl ==="
python -m cartoonstyle.csv_to_beats --visuals "$VIDEO_DIR/visuals.csv" --out "$VIDEO_DIR/beats.jsonl"

echo "=== [3/5] TTS (AndrewNeural +0%) ==="
python -m cartoonstyle.tts \
    --beats "$VIDEO_DIR/beats.jsonl" \
    --out-dir "$VIDEO_DIR" \
    --voice en-US-AndrewNeural \
    --rate=+0% \
    --visuals "$VIDEO_DIR/visuals.csv" 2>&1 | tee /tmp/tts_log.txt

if ! grep -q "moments locked.*\[OK\]" /tmp/tts_log.txt; then
    echo ""
    echo "!!! TTS DID NOT LOCK ALL MOMENTS — stopping here. !!!"
    exit 1
fi

echo "=== [4/5] Stitching VIDEO.mp4 ==="
python -m cartoonstyle.stitch \
    --timing "$VIDEO_DIR/beats_with_timing.json" \
    --images-dir "$FRAMES" \
    --out "$VIDEO_DIR/VIDEO.mp4" \
    --pick 0

echo "=== [5/5] Packaging deliverable zip ==="
rm -f apoc_day1_DELIVERABLE.zip
zip -j apoc_day1_DELIVERABLE.zip \
    "$VIDEO_DIR/VIDEO.mp4" \
    "$VIDEO_DIR/thumbnail.jpg" \
    "$VIDEO_DIR/metadata.json" \
    "$VIDEO_DIR/script.md" \
    "$VIDEO_DIR/youtube.txt"

echo ""
echo "========================================="
echo "DONE. Deliverable:"
ls -lh apoc_day1_DELIVERABLE.zip
echo "========================================="
