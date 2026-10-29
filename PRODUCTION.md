# PRODUCTION RECORD

Episode-specific progress and measurements. The universal workflow lives in
docs/WORKFLOW_SPEC.md. Never copy example numbers into this file as if they
were observed - every value below must come from an actual run.

## Episode

- topic: Apocalypse survival aboard a ship you just boarded — crowds onboard, a few
  infections among passengers, but safer than land: no contact with shore, and far
  more zombies on land than on the ship (second person, explicitly fictional).
  Carried over from the user's 2026-10-28 selection; re-confirm before Gate 1.
- narrator voice: en-US-AndrewNeural @ +10% (user-selected 2026-10-28; calibrate with this)
- working branch: arena/943137f3-cartoonstyle
- status: Tooling imported and verified on this branch; Gate 0 preflight awaiting user's Kaggle run

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

- 2026-10-29: Session branch arena/943137f3-cartoonstyle created from main.
  Frozen v2.1 tooling (docs/, kaggle/, tools/, vidsync/, narration/preflight.md,
  requirements.txt, .gitignore) brought in unchanged from
  arena/20df3367-cartoonstyle @ 2e6834a — nothing rewritten, nothing extended.
  Only repo/branch pointers updated (README.md, docs/WORKFLOW_SPEC.md) plus this
  record. Verified on this branch: `kaggle/preflight.py --validate-only` clean
  (57-word passage, markers/config OK, no synthesis) and `tools/selftest.py`
  43 passed / 0 failed. Episode topic and voice carried over from the 2026-10-28
  selection. Gate 0 blocks all authoring: no script until the preflight is measured.

- 2026-10-28: Pipeline moved onto this repo (AsuraXP/CartoonStyle),
  imported from AsuraXP/newrepo2 @ 4dde43f: built v2 tooling + edge-tts 7.3.1
  generator-tee fix, unchanged. Verified in workspace: selftest 43/43 PASS,
  preflight --validate-only clean. Spec streamlined to v2.1
  (docs/WORKFLOW_SPEC.md). Run, never rebuild.

## Gate status

- Gate 0 (calibration): PENDING - awaiting user's Kaggle preflight run on branch arena/943137f3-cartoonstyle
- Gate 1 (script): blocked by Gate 0 (topic recorded, script not yet written)
- Gate 2 (anchors): not started
- Gate 3 (frames / style): not started - MANDATORY STOP for user style selection
- Gate 4 (verify + measure): not started
- Gate 5 (render): not started
- Gate 6 (publish assets): not started
