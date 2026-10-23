# Napoleon Complex Video — STATUS

**Status**: FRAMES COMPLETE (120/120), ready for Kaggle TTS + stitch pipeline.

## Stats
- **Beats**: 13
- **Moments**: 120
- **Words**: ~856
- **Estimated runtime**: ~5:10 (AndrewNeural at -5%)
- **Avg words/moment**: 7.1 (~2.3 sec/cut — tighter than Why We Sleep)
- **Frames**: 120 PNGs in frames/ (need resize to 1920x1080 on Kaggle)

## Kaggle Pipeline (same as Why We Sleep)
1. Upload repo + frames to Kaggle
2. Resize all frames: `mogrify -resize 1920x1080! frames/*.png` (force exact dimensions)
3. `python -m cartoonstyle.csv_to_beats examples/napoleon_complex_video/visuals.csv examples/napoleon_complex_video/beats.jsonl`
4. `python -m cartoonstyle.tts examples/napoleon_complex_video/beats.jsonl examples/napoleon_complex_video/narration.mp3 --voice en-US-AndrewNeural --rate=-5% --fill=examples/napoleon_complex_video/frames`
   - Verify output shows `[OK] 120/120 moments locked`
   - If any moments miss, check the debug JSONL and the token dump
5. `python -m cartoonstyle.stitch examples/napoleon_complex_video/beats.jsonl examples/napoleon_complex_video/frames examples/napoleon_complex_video/narration.mp3 examples/napoleon_complex_video/VIDEO.mp4`
6. Zip deliverable: `zip -r napoleon_complex_DELIVERABLE.zip examples/napoleon_complex_video/VIDEO.mp4 examples/napoleon_complex_video/thumbnail.jpg examples/napoleon_complex_video/metadata.json`

## Script Summary
Why you think Napoleon was short — you're wrong. James Gillray, a British political cartoonist paid by the UK government during the Napoleonic Wars, created "Little Boney" in 1803 as wartime propaganda. The myth compounded via:
1. French-inch to English-inch unit conversion error (5'2" French = ~5'6" modern)
2. The Imperial Old Guard being exceptionally tall, making Napoleon look short by comparison in paintings
3. "Le Petit Caporal" mistranslation (affectionate nickname referring to rank, not height)
4. 1920s psychologists (Alfred Adler's inferiority complex theory) cemented "Napoleon Complex" as pop psychology

## Research Sources (for video description)
- History.com — Napoleon height sources (valet Constant, Gourgaud, Antommarchi autopsy); Gillray "did more than all the armies of Europe" quote
- The Public Domain Review (Peter W. Walker essay) — Gillray creating "Little Boney" as minuscule megalomaniac; father of political cartoon
- National Post (Tim Clayton, historian) — tiny-Napoleon trope did not exist before 1803 "Maniac Ravings" cartoon
- National Geographic History — Gillray never saw Napoleon in person; Adler's inferiority complex 1920s origin
- History Collection — British PM Pitt government funded anti-Napoleon satire; "Le Petit Caporal" mistranslation; Old Guard height threshold
- Wikipedia "Napoleon complex" — 5'2" French = 1.67m/5'6" imperial; Gillray popularized myth
- Wikipedia "James Gillray" / Britannica / The Conversation — Gillray as father of political cartoon; Bonaparte and the British exhibition at British Museum
