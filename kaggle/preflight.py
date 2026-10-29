#!/usr/bin/env python3
"""Phase 0 - preflight TTS calibration (DESIGNATED REMOTE RUNNERS ONLY).

Entry points (internet enabled; workspace never synthesizes):

    # Kaggle notebook:
    python kaggle/preflight.py
    # GitHub Actions (workflow `.github/workflows/preflight.yml`):
    gh workflow run preflight.yml --ref <branch>   # report lands in the run log

It resolves the repository from its own location, synthesizes ONLY the
marked passage of narration/preflight.md in one continuous request, with
WordBoundary explicitly requested (never an engine default), and prints a
full measurement report: engine version, voice/rate, requested vs returned
boundary types, event counts, words-per-event, source-word recovery and
coverage, timestamp order/correspondence validation, non-zero audio
duration, suggested sec_per_word, and sync_source (measured vs interpolated).

Offline syntax/logic checking WITHOUT synthesis (safe in the workspace):

    python kaggle/preflight.py --validate-only

This script NEVER synthesizes in the local workspace - only on Kaggle or
the GitHub Actions preflight workflow.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# --- resolve repository from this file's location -------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vidsync import media  # noqa: E402
from vidsync.config import load_config  # noqa: E402
from vidsync.textnorm import count_words, tokenize  # noqa: E402
from vidsync.timing import (  # noqa: E402
    SYNC_ESTIMATE,
    SYNC_MEASURED_SENTENCES,
    SYNC_MEASURED_WORDS,
    recover_word_times,
)
from vidsync.tts import RealEdgeTTS, TTSError, is_remote_runner  # noqa: E402

PREFLIGHT_MD = REPO_ROOT / "narration" / "preflight.md"
BUILD_DIR = REPO_ROOT / "build" / "preflight"

MARKER_START = "<!-- TTS:START -->"
MARKER_END = "<!-- TTS:END -->"


class PreflightError(Exception):
    pass


def extract_marked_passage(md_path: Path) -> str:
    """Extract exactly one marked passage from the markdown file."""
    if not md_path.exists():
        raise PreflightError(f"missing {md_path}")
    text = md_path.read_text(encoding="utf-8")
    if text.count(MARKER_START) != 1 or text.count(MARKER_END) != 1:
        raise PreflightError(
            f"{md_path.name} must contain exactly one {MARKER_START} and one {MARKER_END}"
        )
    start = text.index(MARKER_START) + len(MARKER_START)
    end = text.index(MARKER_END)
    if end <= start:
        raise PreflightError("TTS:END marker must come after TTS:START")
    passage = text[start:end].strip()
    if not passage:
        raise PreflightError("marked passage is empty")
    if re.search(r"[#*`\[\]]", passage):
        raise PreflightError(
            "marked passage must be plain prose - no markdown characters that TTS "
            "would read aloud"
        )
    return passage


def fmt_secs(x: float) -> str:
    return f"{x:.3f}"


def report_lines(passage: str, result, recovery, word_times) -> list:
    """Build the measurement report as printable lines."""
    eng = result.engine
    lines = []
    lines.append("=" * 72)
    lines.append("PREFLIGHT CALIBRATION REPORT (actual measurement)")
    lines.append("=" * 72)
    lines.append(f"engine              : {eng.get('name')} {eng.get('version')}")
    lines.append(f"voice and rate      : {eng.get('voice')} / {eng.get('rate')}")
    lines.append(
        f"                    : volume {eng.get('volume')} pitch {eng.get('pitch')}"
    )
    lines.append(f"requested boundary  : {result.requested_boundary}")
    observed = ", ".join(result.returned_boundary_types) or "(none)"
    lines.append(f"observed boundary   : {observed}")
    lines.append(f"boundary events     : {recovery.event_count}")
    lines.append(
        "words per event     : "
        f"min {recovery.words_per_event_min:.0f} "
        f"max {recovery.words_per_event_max:.0f} "
        f"mean {recovery.words_per_event_mean:.2f} "
        f"({recovery.events_with_one_word}/{recovery.event_count} events carried exactly 1 word)"
    )
    lines.append(f"granularity         : {granularity_label(recovery)}")
    lines.append(f"source-word recovery: {recovery.summary()}")
    if recovery.unrecovered_list:
        lines.append(f"unrecovered tokens  : {recovery.unrecovered_list}")
    lines.append(
        "timestamp order     : "
        + ("monotonically non-decreasing - OK" if recovery.timestamps_monotonic else "VIOLATIONS FOUND")
    )
    for v in recovery.order_violations:
        lines.append(f"    order violation : {v}")
    corr = (
        "event text corresponds to source words in order - OK"
        if recovery.correspondence_ok
        else "MISMATCH between event text and source words (see recovery line)"
    )
    lines.append(f"word correspondence : {corr}")
    lines.append(f"audio duration (s)  : {result.audio_duration_s:.3f} [measured from file]")
    if result.audio_duration_s <= 0:
        lines.append("    ERROR: audio duration must be non-zero")

    words = count_words(passage)
    spw_audio = result.audio_duration_s / words if words else 0.0
    spoken = [w for w in word_times if w.t_start is not None and w.source == "measured"]
    if spoken:
        spw_events = (spoken[-1].t_end or spoken[-1].t_start) / max(len(spoken), 1)
    else:
        spw_events = 0.0
    lines.append(f"passage words       : {words}")
    lines.append(f"sec_per_word (audio): {spw_audio:.4f}  = audio_duration / words")
    lines.append(f"sec_per_word (evts) : {spw_events:.4f}  = last measured event end / words")
    lines.append(f"suggested estimate.sec_per_word: {spw_audio:.4f}")
    lines.append("sync_source         : " + sync_source_for(recovery, result))
    n_meas = sum(1 for w in word_times if w.source == "measured")
    n_interp = sum(1 for w in word_times if w.source == "interpolated")
    lines.append(
        "timestamp provenance: "
        f"{n_meas} words backed by measured event timestamps; "
        f"{n_interp} words interpolated (approximate - never word-accurate)"
    )
    for note in result.notes:
        lines.append(f"note                : {note}")
    lines.append("=" * 72)
    return lines


def granularity_label(recovery) -> str:
    if recovery.event_count == 0:
        return "NONE - no boundary events returned"
    if (
        recovery.words_per_event_max <= 1.0
        and recovery.recovered_words == recovery.expected_words
    ):
        return "WORD-LEVEL - every covered word has a measured timestamp"
    if recovery.words_per_event_max > 1.0:
        return (
            "SENTENCE/COARSE-LEVEL - events span multiple words; only event edges "
            "are measured"
        )
    return "INCOMPLETE WORD COVERAGE - some words lack measured timestamps"


def sync_source_for(recovery, result) -> str:
    observed = set(result.returned_boundary_types)
    if (
        observed == {"WordBoundary"}
        and recovery.words_per_event_max <= 1.0
        and recovery.recovered_words == recovery.expected_words
    ):
        return SYNC_MEASURED_WORDS + " (verified word correspondence)"
    if recovery.event_count > 0:
        return SYNC_MEASURED_SENTENCES + " - NOT word-accurate; reduced precision"
    return "none: no boundary metadata returned"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--validate-only",
        action="store_true",
        help="offline checks only (markers, words, config); NEVER synthesizes",
    )
    args = ap.parse_args()

    try:
        passage = extract_marked_passage(PREFLIGHT_MD)
        cfg = load_config(REPO_ROOT)
    except Exception as e:
        print(f"PREFLIGHT ERROR: {e}", file=sys.stderr)
        return 2

    words = count_words(passage)
    print(f"repository          : {REPO_ROOT}")
    print(f"passage             : {PREFLIGHT_MD.relative_to(REPO_ROOT)}")
    print(f"passage words       : {words} (target: roughly 60)")
    print(f"voice / rate        : {cfg['tts']['voice']} / {cfg['tts']['rate']}")
    print(f"boundary request    : {cfg['tts']['boundary']} (explicit, never engine default)")

    if args.validate_only:
        print("mode                : --validate-only (NO synthesis - workspace safe)")
        print("validate            : markers OK, passage OK, config OK")
        print("sync_source         : n/a until measured on a remote runner")
        return 0

    # --- synthesis only on designated remote runners (Kaggle / Actions) ----
    if not is_remote_runner():
        print(
            "\nPREFLIGHT REFUSING TO SYNTHESIZE IN THE LOCAL WORKSPACE.\n"
            "The workspace must never run TTS (spec: three locations).\n"
            "Run it on a remote runner: a Kaggle notebook, or trigger the\n"
            "`preflight` GitHub Actions workflow (gh workflow run preflight.yml).\n"
            "Use `python kaggle/preflight.py --validate-only` for offline checks here.",
            file=sys.stderr,
        )
        return 3

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    audio_path = BUILD_DIR / "preflight.mp3"
    events_path = BUILD_DIR / "preflight_events.json"

    tts = RealEdgeTTS(
        voice=cfg["tts"]["voice"],
        rate=cfg["tts"]["rate"],
        volume=cfg["tts"]["volume"],
        pitch=cfg["tts"]["pitch"],
    )
    print("mode                : full calibration (ONE synthesis request)")
    try:
        result = tts.synthesize(passage, audio_path, boundary=cfg["tts"]["boundary"])
    except TTSError as e:
        print(f"PREFLIGHT SYNTHESIS FAILED: {e}", file=sys.stderr)
        return 4

    events_path.write_text(result.to_json(), encoding="utf-8")

    word_times, recovery = recover_word_times(passage, result.events)
    for line in report_lines(passage, result, recovery, word_times):
        print(line)
    print(f"artifacts           : {audio_path} , {events_path}")
    print("record these numbers in PRODUCTION.md and kaggle/config.yaml (Gate 0).")

    # Gate 0 hard checks -> non-zero exit if unmet.
    problems = []
    if result.audio_duration_s <= 0:
        problems.append("audio duration is zero")
    if not result.events:
        problems.append("no boundary events returned")
    if not recovery.timestamps_monotonic:
        problems.append("timestamps are not monotonically ordered")
    if recovery.recovered_words != recovery.expected_words:
        problems.append(
            f"source-word recovery incomplete: {recovery.recovered_words}/{recovery.expected_words}"
        )
    if problems:
        print("\nGATE 0 PROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 5
    print("\nGate 0 measurement checks: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
