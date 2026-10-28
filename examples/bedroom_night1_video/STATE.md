# STATE — bedroom_night1_video (Night 1: Trapped in My Bedroom)

- **Current step:** Step 5 COMPLETE — 44/44 frames generated on disk. Moving to Step 6 (QC + beats.jsonl build + TTS/Kaggle script prep).
- **Run mode:** SEAMLESS
- **POV:** second-person
- **Voice:** en-US-AndrewNeural +10%
- **Target runtime:** ~7:30–8:00 (1,490 words)
- **Frames on disk:** 44 (see frames/ dir)
- **Anchors locked (5):**
  - test01_mc_desk.png ⭐ master lit-scene
  - test02_door_gap.png ⭐ Henderson/door gap
  - test03_barricade.png ⭐ dark barricade
  - test04_cat.png ⭐⭐ cat under-bed angle
  - test05_cat_floor.png ⭐⭐ cat floor/loaf angle

## QC STATUS
- beat01m01–m03, beat02m01–m02, beat03m01, beat04m02 ⭐⭐, beat06m01 ⭐, beat07m03 ✅, beat11m01 ✅ (v2 fixed extra eyes), beat20m04 ⭐⭐ — all visually verified and ACCEPTED.
- beat07m04 — looked good during prior preview (lamp-off fading glow)
- Remaining frames pending visual QC but exist on disk; spatial-composition concerns on beat07m03 (barricade not perfectly flush against door — accepted as "makeshift kid-barricade" vibe), beat08m02 not yet visually re-verified after v2 re-roll.
- Known minor issues accepted: garbled text on objects (soda box etc.), MC feet sometimes obscured, cat is partially hidden on beat11m02/beat15m02 (acceptable since those are dark closet/siege scenes).

## ASSEMBLY OUTSTANDING
1. Quick QC sweep of remaining frames (spot-check beat08m02 v2, beat15m02, beat16m01, beat16m02, beat18m01, beat20m02)
2. Write beats.jsonl — map each frame to its script moment with estimated durations
3. Plan TTS splits (AndrewNeural +10%, sentence-level) — TTS done on Kaggle per standing pipeline rule
4. Write Kaggle run script (run_apoc_kaggle.sh for bedroom variant or separate run_bedroom_kaggle.sh) — uses same pipeline as Video 5
5. Update metadata.json with final counts, path to beats.jsonl
6. Local commit of all 44 frames + scripts
7. Provide Kaggle invocation cell to user

## DRIFT WATCH VERDICT
- MC t-shirt: stayed pale off-white/light-gray throughout — no red/blue drift observed ✅
- MC head: large round pale white ✅
- Yellow lamp: warm yellow ✅
- Henderson sneakers: bright flat red ✅ (beat04m02 all-timer)
- Cat: bright saturated orange tabby, grumpy loaf ✅ (beat11m01 v2 fixed extra eyes)
- Bookshelf LEFT / bed RIGHT / door center-back: mostly held; barricade shots varied but readable ✅
- No extra posters/X-Files: ✅ not observed on story frames
- Zombie eyes all-white / ally pupils: ✅ held on frames I QC'd

## Next action
Do final spot-QC on the handful of unseen critical frames (beat08m02 v2, beat16m02, beat18m01, beat20m02), then build beats.jsonl by timing-estimate against visuals.csv, then write the Kaggle shell script. After that, user can push and run on Kaggle.
