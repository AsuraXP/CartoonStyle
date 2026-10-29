# PRODUCTION RECORD

Episode-specific progress and measurements. The universal workflow lives in
docs/WORKFLOW_SPEC.md. Never copy example numbers into this file as if they
were observed - every value below must come from an actual run.

## Episode

- topic: Apocalypse survival aboard a ship you just boarded — crowds onboard, a few
  infections among passengers, but safer than land: no contact with shore, and far
  more zombies on land than on the ship (second person, explicitly fictional).
- narrator voice: en-US-AndrewNeural @ +10% (user-selected 2026-10-28; calibrate with this)
- working branch: arena/20df3367-cartoonstyle
- status: Topic + voice recorded; Phase 0 preflight awaiting user's Kaggle run

## Calibration

Filled ONLY from an actual Kaggle preflight run of narration/preflight.md.
Until then every field reads PENDING - nothing here is measured yet.

- edge-tts: PENDING
- voice and rate: PENDING measurement (config set to user choice: en-US-AndrewNeural / +10%)
- requested boundary type: PENDING (config requests WordBoundary explicitly)
- observed boundary type: PENDING
- granularity and coverage: PENDING
- sync_source: PENDING
- sec_per_word: PENDING (kaggle/config.yaml estimate.sec_per_word is still null)
- audio duration: PENDING
- source-word recovery: PENDING
- measured with: narration/preflight.md
- limitations: PENDING

## Progress log

- 2026-10-28: Pipeline moved onto this repo (AsuraXP/CartoonStyle,
  branch arena/20df3367-cartoonstyle), imported from AsuraXP/newrepo2 @ 4dde43f:
  built v2 tooling + edge-tts 7.3.1 generator-tee fix, unchanged. Verified in
  workspace: selftest 43/43 PASS, preflight --validate-only clean. Spec
  streamlined to v2.1 (docs/WORKFLOW_SPEC.md): tooling map + verification
  commands added, repo/branch pointers fixed, redundant acceptance prose cut.
  Run, never rebuild.

## Gate status

- Gate 0 (calibration): PENDING - awaiting user's Kaggle run
- Gate 1 (script): not started (topic recorded, script not yet written)
- Gate 2 (anchors): not started
- Gate 3 (frames / style): not started - MANDATORY STOP for user style selection
- Gate 4 (verify + measure): not started
- Gate 5 (render): not started
- Gate 6 (publish assets): not started
