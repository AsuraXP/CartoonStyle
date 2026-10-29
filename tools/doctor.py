#!/usr/bin/env python3
"""One-shot offline gate runner - run this instead of the individual checks.

    python tools/doctor.py

Runs every workspace-safe check in order and prints one PASS/FAIL/SKIP table:

  1. kaggle/preflight.py --validate-only   (Phase 0 offline checks)
  2. kaggle/run.py   --estimate --repo .    (Gate 2 - only if narration + frames.json exist)
  3. tools/prep_frames.py --sheet --strict  (Gate 3 - only if frames.json exists)
  4. tools/selftest.py                      (Gate 4 proof)

SKIP means the inputs for that phase do not exist yet (work not started) -
it is not a failure. FAIL means a check that should pass does not; the table
ends with the tail of its output. Exit code: 0 = nothing failed, 1 = >=1 FAIL.

No network, no TTS, no new logic: this only orchestrates the existing tools.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run(argv: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        argv,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
    )
    out = (proc.stdout or "") + (proc.stderr or "")
    return proc.returncode, out.strip()


def main() -> int:
    has_narration = (REPO_ROOT / "narration" / "narration.md").exists()
    has_frames = (REPO_ROOT / "frames.json").exists()

    checks: list[tuple[str, list[str], bool]] = [
        ("preflight --validate-only", [sys.executable, "kaggle/preflight.py", "--validate-only"], True),
        ("run --estimate (Gate 2)", [sys.executable, "kaggle/run.py", "--estimate", "--repo", "."], has_narration and has_frames),
        ("prep_frames --sheet --strict (Gate 3)", [sys.executable, "tools/prep_frames.py", "--repo", ".", "--sheet", "--strict"], has_frames),
        ("selftest (Gate 4)", [sys.executable, "tools/selftest.py"], True),
    ]

    results: list[tuple[str, str, float, str]] = []
    any_fail = False
    for name, argv, runnable in checks:
        if not runnable:
            results.append((name, "SKIP", 0.0, "inputs not present yet (phase not started)"))
            continue
        t0 = time.monotonic()
        rc, out = _run(argv)
        dt = time.monotonic() - t0
        status = "PASS" if rc == 0 else "FAIL"
        tail = "" if rc == 0 else "\n".join(out.splitlines()[-8:])
        if rc != 0:
            any_fail = True
        results.append((name, status, dt, tail))

    print("=" * 72)
    print("DOCTOR - offline gates (no network, no TTS)")
    print("=" * 72)
    width = max(len(name) for name, *_ in results)
    for name, status, dt, tail in results:
        print(f"  {name:<{width}}  {status}  ({dt:.1f}s)")
        if tail:
            for line in tail.splitlines():
                print(f"      | {line}")
    print("=" * 72)
    if any_fail:
        print("RESULT: FAIL - fix the failing check above (input data, not tooling).")
        return 1
    print("RESULT: OK - every runnable check passed; remaining phases are SKIPped until their inputs exist.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
