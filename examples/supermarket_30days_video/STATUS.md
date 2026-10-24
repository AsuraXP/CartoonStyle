# Trapped in a Supermarket for 30 Days — STATUS

**Status**: ✅ ALL FRAMES DONE (196/196 native 1376×768 PNGs). Ready for final TTS+stitch on Kaggle.

## Stats
- **Beats**: 27 (narrative day-by-day)
- **Moments**: 196
- **Words**: ~1943
- **Estimated runtime**: ~12:30 (AndrewNeural at 0% = normal pace, ~155 wpm)
- **Avg words/moment**: ~9.9 (~3.8 sec/cut)
- **Tone**: deadpan dark comedy first-person survival narrative
- **Ending**: dark twist — signs taped backward on inside of glass, entire "HELP" was invisible; ends with insomnia callback
- **Frame resolution**: 1376×768 native for ALL 196 frames, zero scaling/cropping/letterboxing

## One-shot Kaggle cell
```
!rm -rf CartoonStyle && git clone -b arena/22c9cd35-cartoonstyle https://github.com/AsuraXP/CartoonStyle.git && cd CartoonStyle && bash run_supermarket.sh
from IPython.display import FileLink
FileLink('CartoonStyle/supermarket_30days_DELIVERABLE.zip')
```

The runner does: deps → csv_to_beats → TTS (Andrew +0%) → stitch → zip VIDEO.mp4 + thumbnail + metadata + script.md + youtube.txt.
