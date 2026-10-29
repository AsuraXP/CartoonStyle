# Preflight calibration passage

This passage calibrates the text to speech engine before a single word of
episode narration is written. It is synthesized in one request on Kaggle and
never changes without re-running calibration.

<!-- TTS:START -->
Rivers shape the land slowly, grain by grain. A mountain stream carries sand downstream and drops it in the valley. Over many seasons the plain grows wider. Farmers trust the soil because they watched it form. Rain falls, water moves, and the quiet work of erosion continues without hurry. Every flood leaves a thin new layer behind.
<!-- TTS:END -->

The section above the start marker is not synthesized. The end marker closes
the spoken passage. Calibration results are recorded in PRODUCTION.md and
kaggle/config.yaml.
