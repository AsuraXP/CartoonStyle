"""vidsync - shared pipeline library for the narration-first explainer workflow.

Used by kaggle/preflight.py, kaggle/run.py, tools/prep_frames.py and
tools/selftest.py.  See docs/WORKFLOW_SPEC.md for the governing specification.

Boundaries enforced by this package:
  * Real TTS synthesis (tts.RealEdgeTTS) refuses to run outside Kaggle.
  * Nothing in this package authors narration, edits prompts or generates frames.
"""

__version__ = "1.0.0"
