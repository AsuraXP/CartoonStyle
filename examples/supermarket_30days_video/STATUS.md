# Trapped in a Supermarket for 30 Days — STATUS

**Status**: FRAMES IN PROGRESS — 1/209 done (anchor only). Needs ~21 more batches of 10.

## Stats
- **Beats**: 27 (narrative day-by-day)
- **Moments**: 209
- **Words**: ~1955
- **Estimated runtime**: ~12:40 (AndrewNeural at 0% = normal pace, ~155 wpm)
- **Avg words/moment**: 9.4 (~3.6 sec/cut)
- **Tone**: deadpan dark comedy first-person survival narrative
- **Ending**: dark twist — signs taped backward on inside of glass, entire "HELP" was invisible

## Kaggle Pipeline (same as napoleon, voice +0% normal rate)
1. Upload repo + frames to Kaggle
2. `python -m cartoonstyle.csv_to_beats --visuals examples/supermarket_30days_video/visuals.csv --out examples/supermarket_30days_video/beats.jsonl`
3. `python -m cartoonstyle.tts --beats examples/supermarket_30days_video/beats.jsonl --out-dir examples/supermarket_30days_video --voice en-US-AndrewNeural --rate=+0% --visuals examples/supermarket_30days_video/visuals.csv`
   - Expect `[OK] 209/209 moments locked`
4. `python -m cartoonstyle.stitch --timing examples/supermarket_30days_video/beats_with_timing.json --images-dir examples/supermarket_30days_video/frames --out examples/supermarket_30days_video/VIDEO.mp4 --pick 0`
5. `zip -j supermarket_30days_DELIVERABLE.zip examples/supermarket_30days_video/VIDEO.mp4 examples/supermarket_30days_video/thumbnail.jpg examples/supermarket_30days_video/metadata.json examples/supermarket_30days_video/script.md examples/supermarket_30days_video/youtube.txt`

## One-shot Kaggle cell
```
!rm -rf CartoonStyle && git clone -b arena/22c9cd35-cartoonstyle https://github.com/AsuraXP/CartoonStyle.git && cd CartoonStyle && bash run_supermarket.sh
from IPython.display import FileLink
FileLink('CartoonStyle/supermarket_30days_DELIVERABLE.zip')
```
