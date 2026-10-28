#!/usr/bin/env bash
#
# run_bedroom_kaggle.sh
# Builds "Night 1: Trapped in My Bedroom" (Video 6) from beats.jsonl.
# Designed to run on Kaggle (Linux, ffmpeg preinstalled).
# Usage: bash CartoonStyle/examples/bedroom_night1_video/run_bedroom_kaggle.sh
#
# Steps:
#   1. pip install edge-tts (Kaggle blocks TTS downloads but edge-tts is free & offline-free)
#   2. For each beat in beats.jsonl, generate TTS mp3 (en-US-AndrewNeural, +10% rate)
#   3. Extend every frame to its narration duration as a silent-padded mp4 (ffmpeg)
#   4. Concat all clips to a single mp4
#   5. Output: bedroom_night1.mp4 in the working directory
set -euo pipefail

VIDEO_DIR="CartoonStyle/examples/bedroom_night1_video"
BEATS="$VIDEO_DIR/beats.jsonl"
OUT="bedroom_night1.mp4"
WORK="img_pool/bedroom_build"
VOICE="en-US-AndrewNeural"
RATE="+10%"

echo "==> Installing edge-tts"
pip install -q edge-tts

echo "==> Preparing work dir"
rm -rf "$WORK"
mkdir -p "$WORK/clips"

echo "==> Generating TTS and per-frame clips"
i=0
while IFS= read -r line; do
  i=$((i+1))
  frame=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['frame'])" "$line")
  text=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['narration'])" "$line")
  dur=$(python3 -c "import json,sys; print(json.loads(sys.argv[1])['duration'])" "$line")
  idx=$(printf "%03d" $i)
  mp3="$WORK/clips/${idx}.mp3"
  mp4="$WORK/clips/${idx}.mp4"
  img="CartoonStyle/$frame"

  # TTS
  edge-tts --voice "$VOICE" --rate "$RATE" --text "$text" --write-media "$mp3" >/dev/null 2>&1

  # Build clip: show image for the beat duration, mix TTS audio.
  # If TTS runs longer than the scripted duration, use the TTS duration as truth (safety pad).
  ffmpeg -y -loglevel error -loop 1 -i "$img" -i "$mp3" \
    -c:v libx264 -tune stillimage -pix_fmt yuv420p -r 24 \
    -c:a aac -b:a 128k -shortest -t "$dur" "$mp4"
  echo "  [$idx] $frame  (scripted ${dur}s)"
done < "$BEATS"

echo "==> Concatenating clips"
# Build concat list
> "$WORK/concat.txt"
for f in "$WORK"/clips/*.mp4; do
  echo "file '$PWD/$f'" >> "$WORK/concat.txt"
done
ffmpeg -y -loglevel error -f concat -safe 0 -i "$WORK/concat.txt" \
  -c:v libx264 -pix_fmt yuv420p -c:a aac -b:a 160k "$OUT"

echo "==> Done: $OUT"
ls -lh "$OUT"
ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$OUT" | awk '{printf "Final runtime: %dm%.0fs\n", int($1/60), $1%60}'
