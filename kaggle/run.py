#!/usr/bin/env python3
"""Phases 2/4/5 - offline anchor check, measured timing, render (KAGGLE).

Modes (exactly one per invocation):

  Offline diagnostics in the workspace (NO network, NO synthesis):
      python kaggle/run.py --estimate --repo .

  Kaggle - synthesize the full narration in ONE request, measure, verify,
  and preserve audio + timing artifacts for the render (no video yet):
      python kaggle/run.py --dry-run --repo /kaggle/working/video-repo

  Kaggle - render the video, REUSING Phase 4 artifacts (never re-synthesizes):
      python kaggle/run.py --out final.mp4 --repo /kaggle/working/video-repo

Synthesis refuses to run outside Kaggle (spec: the workspace never runs TTS).
Frame count is always the number of entries in frames.json - frames are never
merged.  Durations are measured between anchor timestamps - never hand-set.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT_DEFAULT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT_DEFAULT))

from vidsync.anchors import (  # noqa: E402
    AnchorError,
    load_frames_json,
    match_anchors,
)
from vidsync.config import (  # noqa: E402
    build_dir,
    frames_json_path,
    load_config,
    narration_path,
    sec_per_word_or_fallback,
)
from vidsync.manifest import manifest_payload  # noqa: E402
from vidsync.textnorm import tokenize  # noqa: E402
from vidsync.timing import (  # noqa: E402
    SYNC_ESTIMATE,
    build_timeline,
    recover_word_times,
    timeline_to_json,
    write_timings_csv,
)
from vidsync.tts import RealEdgeTTS, TTSError, is_kaggle, load_events_json  # noqa: E402

MARKER_START = "<!-- TTS:START -->"
MARKER_END = "<!-- TTS:END -->"
MD_MARKDOWN_CHARS = re.compile(r"[#*`|\[\]<>]")
MD_URL = re.compile(r"\bhttps?://|\bwww\.")


class RunError(Exception):
    pass


def extract_narration(md_path: Path) -> str:
    """Speakable text between the markers.  '## Sources' after END is stripped."""
    if not md_path.exists():
        raise RunError(
            f"narration file not found: {md_path} (Phase 1 writes narration/narration.md)"
        )
    text = md_path.read_text(encoding="utf-8")
    if text.count(MARKER_START) != 1 or text.count(MARKER_END) != 1:
        raise RunError(f"{md_path} must contain exactly one {MARKER_START}/{MARKER_END} pair")
    start = text.index(MARKER_START) + len(MARKER_START)
    end = text.index(MARKER_END)
    passage = text[start:end].strip()
    if not passage:
        raise RunError("marked narration passage is empty")
    return passage


def check_speakable(passage: str) -> List[str]:
    """Gate 1: no markdown characters TTS would read aloud."""
    problems = []
    for i, line in enumerate(passage.splitlines(), 1):
        if MD_MARKDOWN_CHARS.search(line):
            problems.append(
                f"line {i}: markdown character in speakable text (TTS would read it): {line.strip()!r}"
            )
        if MD_URL.search(line):
            problems.append(f"line {i}: URL in speakable text: {line.strip()!r}")
        if "(" in line or ")" in line:
            problems.append(
                f"line {i}: parentheses in speakable text (no stage directions): {line.strip()!r}"
            )
    return problems


def current_engine_version() -> str:
    try:
        import edge_tts

        return getattr(edge_tts, "__version__", "unknown")
    except Exception:
        return "unavailable"


def frame_image_paths(cfg, repo_root: Path, frame_ids: List[str]) -> List[Path]:
    frames_dir = Path(repo_root) / cfg["frames"]["dir"]
    return [frames_dir / f"{fid}.jpg" for fid in frame_ids]


# --------------------------------------------------------------------------
# Offline estimator (Gate 2)
# --------------------------------------------------------------------------

def run_estimate(repo_root: Path) -> int:
    cfg = load_config(repo_root)
    try:
        passage = extract_narration(narration_path(cfg, repo_root))
    except RunError as e:
        print(f"ESTIMATE ERROR: {e}", file=sys.stderr)
        return 2

    speakable = check_speakable(passage)
    for p in speakable:
        print(f"GATE 1 CHECK: {p}")
    if speakable:
        print("ESTIMATE ERROR: speakable text must contain no markdown/stage directions", file=sys.stderr)
        return 2

    try:
        frames = load_frames_json(frames_json_path(cfg, repo_root))
        matches = match_anchors(frames, passage)
    except (RunError, AnchorError) as e:
        print(f"ESTIMATE ERROR: {e}", file=sys.stderr)
        return 2

    spw, calibrated, spw_note = sec_per_word_or_fallback(cfg)
    words = [w for w in tokenize(passage) if w.norm]
    n_words = len(words)

    print("=" * 72)
    print("OFFLINE ESTIMATE (no synthesis, no network - durations are NOT measured)")
    print("=" * 72)
    print(f"repo                : {repo_root}")
    print(f"narration words     : {n_words}")
    print(f"frames.json entries : {len(frames)}")
    print(f"matched anchors     : {len(matches)} (0 merged; frame count == frames.json)")
    print(f"sec_per_word        : {spw:.4f} [{spw_note}]")
    print(f"sync_source         : {SYNC_ESTIMATE}")
    print()
    print(
        f"{'id':<22}{'occ':>4} {'word@':>6} {'est_start':>10} {'est_hold':>9}  matched_text"
    )
    prev_word = 0
    total_est = 0.0
    rows = []
    for i, m in enumerate(matches):
        first_word = 0 if i == 0 else m.match_word_start
        next_word = matches[i + 1].match_word_start if i + 1 < len(matches) else n_words
        words_covered = next_word - first_word
        est_hold = words_covered * spw
        est_start = total_est
        total_est += est_hold
        rows.append((m, est_start, est_hold, words_covered))
        occ = "" if m.frame.occurrence is None else str(m.frame.occurrence)
        flags = []
        if est_hold < 1.5:
            flags.append("SHORT-HOLD-REVIEW")
        if est_hold > 12.0:
            flags.append("LONG-HOLD-REVIEW")
        print(
            f"{m.frame.id:<22}{occ:>4} {m.match_word_start:>6} {est_start:>10.2f} "
            f"{est_hold:>9.2f}  {m.matched_text[:44]!r} {' '.join(flags)}"
        )
    print()
    est_runtime = n_words * spw
    print(f"estimated runtime   : {est_runtime:.1f} s  ({est_runtime / 60:.2f} min)")
    print(f"estimation basis    : {n_words} narration words x {spw:.4f} s/word ({spw_note})")
    print("NOTE: runtime is an OUTPUT of the script, not a target. Do not pad or trim to a length.")
    print("sync_source         : " + SYNC_ESTIMATE + " - measured timing happens in Phase 4")
    if not calibrated:
        print(
            "\nWARNING: estimate uses the UNCALIBRATED fallback sec_per_word. "
            "Run Phase 0 (kaggle/preflight.py on Kaggle) and record the measured value "
            "in kaggle/config.yaml before reading this estimate as more than a rough shape."
        )
    print("GATE 2 offline check: PASS (all anchors matched; no ambiguity; order OK)")
    return 0


# --------------------------------------------------------------------------
# Kaggle dry-run: measure once, verify, preserve artifacts (Phase 4)
# --------------------------------------------------------------------------

def run_dry(repo_root: Path) -> int:
    if not is_kaggle():
        print(
            "DRY-RUN REFUSING TO SYNTHESIZE OUTSIDE KAGGLE.\n"
            "Run this in a Kaggle notebook (internet enabled). "
            "Use --estimate in the workspace for offline checks.",
            file=sys.stderr,
        )
        return 3

    cfg = load_config(repo_root)
    try:
        passage = extract_narration(narration_path(cfg, repo_root))
    except RunError as e:
        print(f"DRY-RUN ERROR: {e}", file=sys.stderr)
        return 2

    speakable = check_speakable(passage)
    if speakable:
        for p in speakable:
            print(f"GATE 1 CHECK: {p}")
        print("DRY-RUN ERROR: speakable text must contain no markdown/stage directions", file=sys.stderr)
        return 2

    try:
        frames = load_frames_json(frames_json_path(cfg, repo_root))
        matches = match_anchors(frames, passage)
    except (RunError, AnchorError) as e:
        print(f"DRY-RUN ERROR: {e}", file=sys.stderr)
        return 2

    bdir = build_dir(cfg, repo_root)
    bdir.mkdir(parents=True, exist_ok=True)
    audio_path = bdir / "narration.mp3"
    events_path = bdir / "word_events.json"

    tts = RealEdgeTTS(
        voice=cfg["tts"]["voice"],
        rate=cfg["tts"]["rate"],
        volume=cfg["tts"]["volume"],
        pitch=cfg["tts"]["pitch"],
    )
    print("Synthesizing FULL narration in ONE continuous request (never chunked)...")
    try:
        result = tts.synthesize(passage, audio_path, boundary=cfg["tts"]["boundary"])
    except TTSError as e:
        print(f"SYNTHESIS FAILED: {e}", file=sys.stderr)
        return 4
    events_path.write_text(result.to_json(), encoding="utf-8")

    # --- engine verification block (Gate 4) ---
    eng = result.engine
    print("=" * 72)
    print("ENGINE VERIFICATION (full narration, actual measurement)")
    print("=" * 72)
    print(f"edge-tts version    : {eng.get('version')}")
    print(f"voice and rate      : {eng.get('voice')} / {eng.get('rate')}")
    print(f"requested boundary  : {result.requested_boundary}")
    print(f"returned boundary   : {', '.join(result.returned_boundary_types) or '(none)'}")
    print(f"audio duration (s)  : {result.audio_duration_s:.3f} [measured from file, non-zero]")
    for note in result.notes:
        print(f"note                : {note}")

    word_times, recovery = recover_word_times(passage, result.events)
    print(f"boundary events     : {recovery.event_count}")
    print(
        f"words per event     : min {recovery.words_per_event_min:.0f} max "
        f"{recovery.words_per_event_max:.0f} mean {recovery.words_per_event_mean:.2f}"
    )
    print(f"source-word recovery: {recovery.summary()}")
    print(
        "timestamp order     : "
        + ("OK" if recovery.timestamps_monotonic else "VIOLATIONS: " + "; ".join(recovery.order_violations))
    )

    from vidsync.timing import SYNC_MEASURED_SENTENCES, SYNC_MEASURED_WORDS

    if (
        set(result.returned_boundary_types) == {"WordBoundary"}
        and recovery.words_per_event_max <= 1.0
        and recovery.recovered_words == recovery.expected_words
    ):
        sync_source = SYNC_MEASURED_WORDS + " (verified word correspondence)"
    elif recovery.event_count > 0:
        sync_source = SYNC_MEASURED_SENTENCES
    else:
        sync_source = "none: no boundary metadata"
    print(f"sync_source         : {sync_source}")
    n_meas = sum(1 for w in word_times if w.source == "measured")
    n_interp = sum(1 for w in word_times if w.source == "interpolated")
    print(
        f"timestamp provenance: {n_meas} measured / {n_interp} interpolated (of "
        f"{recovery.expected_words} spoken words)"
    )

    spw, calibrated, spw_note = sec_per_word_or_fallback(cfg)
    if not calibrated:
        spw_meas = result.audio_duration_s / max(recovery.expected_words, 1)
        print(
            f"\nCALIBRATION UPDATE: measured sec_per_word = {spw_meas:.4f} "
            f"(= {result.audio_duration_s:.3f}s / {recovery.expected_words} words)"
        )
        print(
            "Record this in kaggle/config.yaml -> estimate.sec_per_word and in "
            "PRODUCTION.md, then re-run --dry-run so est_hold_s uses the measured value."
        )
        spw_used = spw_meas
    else:
        spw_used = spw

    timeline = build_timeline(
        matches,
        passage,
        word_times,
        recovery,
        result.audio_duration_s,
        sec_per_word=spw_used,
        # spw_used is always measured at this point: either the calibrated
        # config value or the value measured from this very synthesis run.
        sec_per_word_calibrated=True,
        transition_s=float(cfg["transitions"]["duration_s"]),
        default_transition=cfg["transitions"]["default"],
        sync_source=sync_source,
    )

    csv_path = bdir / "timings.csv"
    write_timings_csv(timeline, csv_path)
    (bdir / "timing.json").write_text(timeline_to_json(timeline, recovery), encoding="utf-8")

    engine_version = str(eng.get("version"))
    (bdir / "manifest.json").write_text(
        json.dumps(
            {
                "manifest": manifest_payload(passage, cfg, engine_version),
                "engine_name": eng.get("name"),
                "edge_tts_version": engine_version,
                "voice": cfg["tts"]["voice"],
                "rate": cfg["tts"]["rate"],
                "boundary_requested": result.requested_boundary,
                "boundary_returned": result.returned_boundary_types,
                "audio_duration_s": result.audio_duration_s,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print_timings_table(timeline)
    print(f"\ntimings.csv         : {csv_path}")
    print(f"timing artifacts    : {bdir}/narration.mp3 , word_events.json , timing.json , manifest.json")
    print(
        "RENDER COMMAND (reuses these artifacts, will NOT synthesize again):\n"
        f"  python {Path(__file__).as_posix()} --out final.mp4 --repo {repo_root}"
    )

    if timeline.reduced_precision:
        print(
            "\n*** REDUCED PRECISION *** Some anchors are NOT backed by verified word-level "
            "timestamps (see anchor_precision column). Interpolation is not word-accurate. "
            "Rendering will require --allow-reduced-precision as explicit informed approval."
        )

    problems = []
    if recovery.recovered_words != recovery.expected_words:
        problems.append(f"word recovery incomplete: {recovery.summary()}")
    if not recovery.timestamps_monotonic:
        problems.append("timestamps not monotonic")
    if result.audio_duration_s <= 0:
        problems.append("audio duration zero")
    if timeline.frame_count != len(frames):
        problems.append("frame count mismatch vs frames.json")
    if problems:
        print("\nGATE 4 PROBLEMS:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        return 5
    print("\nGATE 4 measured-timing checks: PASS (share the outputs above for review)")
    return 0


def print_timings_table(timeline) -> None:
    print(
        f"{'id':<22}{'anchor@':>8}{'hold_s':>8}{'est_s':>8}{'prec':>12}{'src':>13}  status"
    )
    for f in timeline.frames:
        print(
            f"{f.frame_id:<22}{f.anchor_start_s:>8.2f}{f.hold_s:>8.2f}{f.est_hold_s:>8.2f}"
            f"{f.anchor_precision:>12}{f.timestamp_source:>13}  {f.status}"
        )
        if f.notes:
            for n in f.notes:
                print(f"{'':>22}note: {n}")
    if timeline.transition_conflicts:
        print("TRANSITION CONFLICTS:")
        for c in timeline.transition_conflicts:
            print(f"  - {c}")


# --------------------------------------------------------------------------
# Render (Phase 5) - reuses Phase 4 artifacts
# --------------------------------------------------------------------------

def run_out(repo_root: Path, out_file: str, allow_overlap: bool, allow_reduced: bool, synthesize: bool) -> int:
    cfg = load_config(repo_root)
    try:
        passage = extract_narration(narration_path(cfg, repo_root))
        frames = load_frames_json(frames_json_path(cfg, repo_root))
        matches = match_anchors(frames, passage)
    except (RunError, AnchorError) as e:
        print(f"RENDER ERROR: {e}", file=sys.stderr)
        return 2

    bdir = build_dir(cfg, repo_root)
    audio_path = bdir / "narration.mp3"
    events_path = bdir / "word_events.json"
    manifest_path = bdir / "manifest.json"

    if not (audio_path.exists() and events_path.exists() and manifest_path.exists()):
        if not synthesize:
            print(
                "RENDER REFUSING: Phase 4 artifacts not found in "
                f"{bdir}.\nRun first:\n"
                f"  python {Path(__file__).as_posix()} --dry-run --repo {repo_root}\n"
                "(or pass --synthesize to allow measuring AND rendering in one go - "
                "prefer the dry-run flow so the measurement is reviewed before render)",
                file=sys.stderr,
            )
            return 3
        if not is_kaggle():
            print("RENDER REFUSING TO SYNTHESIZE OUTSIDE KAGGLE.", file=sys.stderr)
            return 3
        print("No Phase 4 artifacts - --synthesize explicitly allows measuring now.")
        rc = run_dry(repo_root)
        if rc != 0:
            return rc
    else:
        # Verify artifacts are current before reuse.
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        want = manifest_payload(passage, cfg, str(man.get("edge_tts_version", "")))
        if man.get("manifest") != want:
            print(
                "RENDER REFUSING: Phase 4 artifacts are STALE (narration, voice, rate, "
                "pitch, boundary or engine version changed since measurement).\n"
                "Re-run --dry-run to re-measure, then render.",
                file=sys.stderr,
            )
            return 3
        eng_ver_now = current_engine_version()
        measured_engine = str(man.get("engine_name", "edge-tts"))
        if measured_engine.startswith("edge-tts") and eng_ver_now not in (
            "unavailable",
            str(man.get("edge_tts_version", "")),
        ):
            print(
                f"RENDER REFUSING: installed edge-tts {eng_ver_now} != measured "
                f"{man.get('edge_tts_version')}. Re-calibrate (preflight) and re-run --dry-run.",
                file=sys.stderr,
            )
            return 3
        print(
            f"REUSING Phase 4 artifacts (verified manifest): {bdir}/narration.mp3 , word_events.json"
        )

    result = load_events_json(events_path)
    # Re-verify the preserved audio itself is intact and non-zero.
    from vidsync import media

    measured_dur = media.probe_duration(audio_path)
    if measured_dur <= 0:
        print("RENDER ERROR: preserved audio has zero duration", file=sys.stderr)
        return 5
    print(f"audio duration      : {measured_dur:.3f}s [re-measured from preserved file]")
    print(f"edge-tts (measured) : {result.engine.get('version')} / {result.engine.get('voice')} / {result.engine.get('rate')}")
    print(f"returned boundary   : {', '.join(result.returned_boundary_types) or '(none)'}")

    word_times, recovery = recover_word_times(passage, result.events)

    from vidsync.timing import SYNC_MEASURED_SENTENCES, SYNC_MEASURED_WORDS

    if (
        set(result.returned_boundary_types) == {"WordBoundary"}
        and recovery.words_per_event_max <= 1.0
        and recovery.recovered_words == recovery.expected_words
    ):
        sync_source = SYNC_MEASURED_WORDS + " (verified word correspondence)"
    elif recovery.event_count > 0:
        sync_source = SYNC_MEASURED_SENTENCES
    else:
        sync_source = "none: no boundary metadata"

    spw, calibrated, spw_note = sec_per_word_or_fallback(cfg)
    if not calibrated:
        spw = measured_dur / max(recovery.expected_words, 1)

    timeline = build_timeline(
        matches,
        passage,
        word_times,
        recovery,
        measured_dur,
        sec_per_word=spw,
        sec_per_word_calibrated=True,
        transition_s=float(cfg["transitions"]["duration_s"]),
        default_transition=cfg["transitions"]["default"],
        sync_source=sync_source,
    )

    if timeline.reduced_precision and not allow_reduced:
        print(
            "RENDER REFUSING: anchor precision is REDUCED (not word-accurate). "
            "Review timings.csv/timing.json and, if you accept the reduced precision, "
            "re-run with --allow-reduced-precision (explicit informed approval).",
            file=sys.stderr,
        )
        return 6

    # Frame images: reviewed compressed JPGs (prepared in the workspace).
    img_paths = frame_image_paths(cfg, repo_root, [f.frame_id for f in timeline.frames])
    missing = [p for p in img_paths if not p.exists()]
    if missing:
        print("RENDER ERROR: missing frame images (run tools/prep_frames.py --strict):", file=sys.stderr)
        for p in missing:
            print(f"  - {p}", file=sys.stderr)
        return 2

    if timeline.frame_count != len(frames):
        print("RENDER ERROR: timeline frame count != frames.json entries", file=sys.stderr)
        return 5

    from vidsync.stitch import ClipInput, StitchError, render

    clips = [
        ClipInput(
            image_path=img,
            hold_start_s=f.hold_start_s,
            hold_end_s=f.hold_end_s,
            transition=f.transition,
        )
        for img, f in zip(img_paths, timeline.frames)
    ]
    out_path = Path(out_file)
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    try:
        render(
            clips,
            audio_path,
            out_path,
            width=int(cfg["video"]["width"]),
            height=int(cfg["video"]["height"]),
            fps=int(cfg["video"]["fps"]),
            transition_s=float(cfg["transitions"]["duration_s"]),
            total_s=measured_dur,
            allow_transition_overlap=allow_overlap,
        )
    except StitchError as e:
        print(f"RENDER FAILED: {e}", file=sys.stderr)
        return 7

    from vidsync import media as m2

    out_dur = m2.probe_duration(out_path)
    streams = m2.probe_streams(out_path)
    print(f"rendered            : {out_path}")
    print(f"rendered duration   : {out_dur:.3f}s (narration {measured_dur:.3f}s, "
          f"delta {abs(out_dur - measured_dur):.3f}s)")
    print(f"streams             : video={streams.video_streams} audio={streams.audio_streams}")
    print(f"rendered frames     : {timeline.frame_count} (== frames.json entries, 0 merged)")
    print(f"transitions         : {cfg['transitions']['default']} @ {cfg['transitions']['duration_s']}s "
          "(camera locked off; every cut is a visible blend)")
    print_timings_table(timeline)
    print("\nGATE 5 candidate: REVIEW THE ACTUAL VIDEO END TO END before accepting.")
    return 0


# --------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--estimate", action="store_true", help="offline anchor/estimate check (no network)")
    mode.add_argument("--dry-run", action="store_true", help="Kaggle: measure + verify + preserve artifacts")
    mode.add_argument("--out", metavar="FILE", help="Kaggle: render video, reusing Phase 4 artifacts")
    ap.add_argument("--repo", default=str(REPO_ROOT_DEFAULT), help="repository root")
    ap.add_argument("--allow-transition-overlap", action="store_true",
                    help="explicitly accept overlapping transitions (must be a reviewed decision)")
    ap.add_argument("--allow-reduced-precision", action="store_true",
                    help="explicit informed approval to render with non word-accurate anchors")
    ap.add_argument("--synthesize", action="store_true",
                    help="with --out: allow measuring in the same run when artifacts are missing")
    args = ap.parse_args()

    repo_root = Path(args.repo).resolve()
    if not repo_root.exists():
        print(f"repo root not found: {repo_root}", file=sys.stderr)
        return 2

    try:
        if args.estimate:
            return run_estimate(repo_root)
        if args.dry_run:
            return run_dry(repo_root)
        return run_out(
            repo_root,
            args.out,
            allow_overlap=args.allow_transition_overlap,
            allow_reduced=args.allow_reduced_precision,
            synthesize=args.synthesize,
        )
    except RunError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
