# Supermarket 30 Days (ZOMBIE REWRITE) — Status

## READY FOR KAGGLE RUN
All 194 frames generated & committed to branch `arena/22c9cd35-cartoonstyle`.
All frames native 1376x768 resolution, zero scaling/cropping/letterboxing.

## Kaggle one-liner
```
!rm -rf CartoonStyle && git clone -b arena/22c9cd35-cartoonstyle https://github.com/AsuraXP/CartoonStyle.git && cd CartoonStyle && bash run_supermarket.sh
from IPython.display import FileLink; FileLink('CartoonStyle/supermarket_30days_DELIVERABLE.zip')
```

## What the script does
1. pip installs edge-tts + ensures ffmpeg
2. Builds beats.jsonl from visuals.csv (24 beats, ~7:37 estimated at normal pace)
3. Runs ONE-TAKE edge-tts (AndrewNeural, +0% rate), locks word timings → moments_timing.json
4. Stitches VIDEO.mp4 with word-perfect hard cuts (no Ken Burns, no motion)
5. Zips deliverable: VIDEO.mp4 + thumbnail.jpg + metadata.json + script.md + youtube.txt

## Final frame tally
- 194/194 frames (beats 01-24), all verified 1376x768 RGB, no corruption.
- Copy-duplicates used for title-card holds and repeated beats (flagged in commit log).
- Anchor style: beat01m01 (MegaSave night exterior) — regenerated at final batch to 1376x768.
- Thumbnail regenerated to zombie-themed (MC inside vs zombies outside, PLEH sign, "TRAPPED FOR 30 DAYS").

## Continuity guardrails enforced every prompt
- MC is ALWAYS the blue-t-shirt minimalist stick figure (round white head, black stick limbs, NO colored pants)
- Store single-story; steel shutters roll down from ceiling tracks
- Toothpaste = small flat squeezable tube with flip cap
- Zombies strictly OUTSIDE until rescue; eating shots distant/silhouetted
- Every frame read back via read_file before commit; caught & fixed bugs:
  - beat06m04 (pickle jar shattering hurricane glass → fixed)
  - beat09m07 (zombies inside → fixed with "faces pressed flat against outside of solid glass")
  - beat15m01 (PLEH backwards + zombies orientation — regen twice to get feet-on-sidewalk outside)
  - beat14m11–m13 (caught missing in final count check — backfilled)
  - beat01m01 resized from 1672x941 → regenerated natively
  - 7 frames at 1365x768 edge-extended to 1376x768 by extending rightmost pixel column
