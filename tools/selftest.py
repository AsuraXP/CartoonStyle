#!/usr/bin/env python3
"""Phase 4 - offline pipeline selftest (NO network, NO real TTS, NO Kaggle).

    python tools/selftest.py

Uses a stand-in TTS backend and synthetic frames.  It asserts, from an
actually rendered video wherever pixels matter:

  * sync at every anchor (correct picture is fully visible at each anchor)
  * multiple distinct anchors inside one sentence
  * uneven, timestamp-driven holds (never fixed-duration slots)
  * a genuinely static picture inside a hold
  * transitions that actually blend (not snaps)
  * correct duration and audio muxed
  * missing / ambiguous / out-of-order anchors fail loudly
  * sentence-only or incomplete metadata is labelled reduced precision,
    and word-accurate timing is only claimed for verified word events
  * frame count == frames.json entries, zero merges
  * transition minimum (0.8 s) enforced
  * reduced-precision renders demand explicit approval

SYNTHETIC TESTS VERIFY PIPELINE LOGIC, NOT ACTUAL ENGINE BEHAVIOR.  The
engine itself is measured only by kaggle/preflight.py and kaggle/run.py
on Kaggle (Phase 0 / Phase 4).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vidsync import media  # noqa: E402
from vidsync.anchors import AnchorError, FrameSpec, load_frames_json, match_anchors  # noqa: E402
from vidsync.config import load_config  # noqa: E402
from vidsync.stitch import ClipInput, StitchError, render  # noqa: E402
from vidsync.textnorm import tokenize  # noqa: E402
from vidsync.timing import build_timeline, recover_word_times  # noqa: E402
from vidsync.tts import StandInTTS  # noqa: E402

NARRATION = (
    "The red fox jumps over the grey log and lands softly on the moss. "
    "Water runs downhill all night long. Stones wait in the grass."
)
# anchors: A "red fox jumps over" (words 1..4) and B "grey log and lands"
# (words 6..9) sit in the SAME sentence; C is in the last sentence.
ANCHORS = [
    ("frame_a", "red fox jumps over", 1),
    ("frame_b", "grey log and lands", 6),
    ("frame_c", "Stones wait in the", 20),
]
COLORS = {
    "frame_a": (200, 30, 30),
    "frame_b": (30, 170, 40),
    "frame_c": (40, 60, 200),
}
T = 1.0  # transition duration used in tests (>= 0.8 minimum)

PASSED: list = []
FAILED: list = []


def check(name: str, cond: bool, detail: str = "") -> bool:
    if cond:
        PASSED.append(name)
        print(f"  PASS  {name}" + (f"  [{detail}]" if detail else ""))
    else:
        FAILED.append(name)
        print(f"  FAIL  {name}  {detail}")
    return cond


def approx(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol


def make_frame_png(path: Path, rgb) -> None:
    from PIL import Image

    Image.new("RGB", (1280, 720), rgb).save(path)


def make_frames_json(path: Path, entries) -> None:
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")


def frame_entry(fid: str, phrase: str, occurrence=None) -> dict:
    e = {
        "id": fid,
        "anchor_phrase": phrase,
        "image_prompt": f"test subject {fid}, flat 2D",
        "negative_prompt": "blurry, text, watermark",
    }
    if occurrence is not None:
        e["occurrence"] = occurrence
    return e


def to_specs(entries) -> list:
    """Convert raw frame dicts to FrameSpec objects (as load_frames_json does)."""
    return [
        FrameSpec(
            id=e["id"],
            anchor_phrase=e["anchor_phrase"],
            image_prompt=e["image_prompt"],
            negative_prompt=e["negative_prompt"],
            occurrence=e.get("occurrence"),
            transition=e.get("transition"),
            raw=e,
        )
        for e in entries
    ]


def build_synthetic(tmp: Path, mode: str = "WordBoundary"):
    """Full synthetic measurement -> timeline, audio, word times."""
    events_audio = tmp / "narration.wav"
    standin = StandInTTS(sec_per_word=0.30, jitter=True, mode=mode)
    result = standin.synthesize(NARRATION, events_audio, boundary="WordBoundary")
    word_times, recovery = recover_word_times(NARRATION, result.events)
    return result, word_times, recovery


def main() -> int:
    print("=" * 72)
    print("SELFTEST (stand-in TTS + synthetic frames; pipeline logic only)")
    print(f"ffmpeg: {media.find_ffmpeg()}")
    print("=" * 72)

    tmp = Path(tempfile.mkdtemp(prefix="vidsync_selftest_"))
    try:
        _run_tests(tmp)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("=" * 72)
    print(f"SELFTEST RESULT: {len(PASSED)} passed, {len(FAILED)} failed")
    for f in FAILED:
        print(f"  failed: {f}")
    if not FAILED:
        print("SELFTEST: PASS")
    return 1 if FAILED else 0


def _run_tests(tmp: Path) -> None:
    cfg = load_config(REPO_ROOT)

    # ------------------------------------------------------------------
    print("\n[1] anchor matching: multiple distinct anchors within one sentence")
    result, word_times, recovery = build_synthetic(tmp)
    frames = [frame_entry(fid, phrase) for fid, phrase, _ in ANCHORS]
    matches = match_anchors(to_specs(frames), NARRATION)
    check("both same-sentence anchors matched",
          len(matches) == 3 and matches[0].match_word_start == 1 and matches[1].match_word_start == 6)
    check("same-sentence anchors have distinct increasing starts",
          matches[0].match_word_start < matches[1].match_word_start)
    check("matched_text is verbatim source", matches[1].matched_text == "grey log and lands")

    print("\n[2] word-time recovery + precision classification")
    check("word-level recovery complete (word mode)",
          recovery.recovered_words == recovery.expected_words and recovery.words_per_event_max <= 1.0,
          recovery.summary())
    timeline = build_timeline(
        matches, NARRATION, word_times, recovery, result.audio_duration_s,
        sec_per_word=0.30, sec_per_word_calibrated=True, transition_s=T,
        default_transition="fade", sync_source="measured",
    )
    check("word-accurate anchors classified exact",
          all(f.anchor_precision == "exact" for f in timeline.frames))
    check("reduced_precision False for word-accurate timing", timeline.reduced_precision is False)

    sent_result, sent_times, sent_recovery = build_synthetic(tmp, mode="SentenceBoundary")
    sent_timeline = build_timeline(
        match_anchors(to_specs(frames), NARRATION), NARRATION, sent_times, sent_recovery,
        sent_result.audio_duration_s, sec_per_word=0.30, sec_per_word_calibrated=True,
        transition_s=T, default_transition="fade", sync_source="approximate",
    )
    check("sentence-level metadata: words-per-event > 1 detected",
          sent_recovery.words_per_event_max > 1.0)
    check("sentence-level metadata labelled reduced precision",
          sent_timeline.reduced_precision is True
          and all(f.anchor_precision == "approximate" for f in sent_timeline.frames))
    check("word vs sentence metadata clearly distinguished",
          timeline.reduced_precision != sent_timeline.reduced_precision)

    print("\n[3] uneven, timestamp-driven holds")
    holds = [f.hold_s for f in timeline.frames]
    check("holds are uneven (not fixed-duration slots)", max(holds) - min(holds) > 0.2,
          f"holds={[round(h, 2) for h in holds]}")
    ok_delta = True
    for i in range(len(timeline.frames) - 1):
        expected = timeline.frames[i + 1].anchor_start_s - (
            timeline.frames[i].anchor_start_s if i > 0 else 0.0
        )
        ok_delta &= approx(timeline.frames[i].hold_s, expected, 1e-6)
    ok_delta &= approx(timeline.frames[-1].hold_s,
                       result.audio_duration_s - timeline.frames[-1].anchor_start_s, 1e-6)
    check("each hold equals the measured anchor delta", ok_delta)
    check("picture covers from time zero", timeline.frames[0].hold_start_s == 0.0)
    check("final hold ends at measured narration end",
          approx(timeline.frames[-1].hold_end_s, result.audio_duration_s, 1e-6))
    check("frame count == frames.json entries, zero merges",
          timeline.frame_count == len(frames) == 3)

    print("\n[4] render + audio mux + duration")
    img_dir = tmp / "frames"
    img_dir.mkdir()
    for fid, phrase, _ in ANCHORS:
        make_frame_png(img_dir / f"{fid}.jpg", COLORS[fid])
    clips = [
        ClipInput(img_dir / f"{f.frame_id}.jpg", f.hold_start_s, f.hold_end_s, "fade")
        for f in timeline.frames
    ]
    out = tmp / "selftest.mp4"
    render(clips, result.audio_path, out, transition_s=T, total_s=result.audio_duration_s)
    streams = media.probe_streams(out)
    check("audio muxed into the render", streams.has_audio and streams.has_video,
          f"video={streams.video_streams} audio={streams.audio_streams}")
    out_dur = media.probe_duration(out)
    check("rendered duration matches measured narration within 1 s",
          abs(out_dur - result.audio_duration_s) <= 1.0,
          f"video={out_dur:.3f}s audio={result.audio_duration_s:.3f}s")

    print("\n[5] sync at every anchor (pixel evidence)")
    for i, f in enumerate(timeline.frames):
        probe = tmp / f"sync_{f.frame_id}.png"
        media.extract_frame(out, f.hold_start_s + 0.12, probe)  # transition already over
        got = media.mean_rgb(probe)
        want = COLORS[f.frame_id]
        check(f"frame '{f.frame_id}' fully visible at its anchor",
              all(abs(g - w) <= 12 for g, w in zip(got, want)),
              f"got {tuple(round(g) for g in got)} want {want}")
    # the picture before the first transition is frame A
    probe = tmp / "sync_pre.png"
    media.extract_frame(out, timeline.frames[1].hold_start_s - T - 0.12, probe)
    got = media.mean_rgb(probe)
    check("frame A fully visible before its outgoing transition begins",
          all(abs(g - w) <= 12 for g, w in zip(got, COLORS["frame_a"])),
          f"got {tuple(round(g) for g in got)}")

    print("\n[6] genuinely static picture inside a long hold")
    f_b = timeline.frames[1]
    mid_a = f_b.hold_start_s + (f_b.hold_s - T) * 0.5 - 0.30
    p1, p2 = tmp / "static1.png", tmp / "static2.png"
    media.extract_frame(out, mid_a, p1)
    media.extract_frame(out, mid_a + 0.35, p2)
    c1, c2 = media.mean_rgb(p1), media.mean_rgb(p2)
    check("two frames inside a hold are the same picture",
          all(abs(a - b) <= 2.5 for a, b in zip(c1, c2)),
          f"delta {tuple(round(abs(a - b), 1) for a, b in zip(c1, c2))}")

    print("\n[7] transitions actually blend")
    b_start = timeline.frames[1].hold_start_s
    mid_t = b_start - T / 2.0
    probe = tmp / "blend.png"
    media.extract_frame(out, mid_t, probe)
    got = media.mean_rgb(probe)
    want = tuple((COLORS["frame_a"][i] + COLORS["frame_b"][i]) / 2 for i in range(3))
    check("mid-transition pixel is a blend of both frames",
          all(abs(g - w) <= 18 for g, w in zip(got, want)),
          f"got {tuple(round(g) for g in got)} want midpoint {tuple(round(w) for w in want)}")
    probe = tmp / "blend_before.png"
    media.extract_frame(out, b_start - T - 0.12, probe)
    check("no snap: picture is pure before the transition window starts",
          all(abs(g - w) <= 12 for g, w in zip(media.mean_rgb(probe), COLORS["frame_a"])))

    print("\n[8] single-frame render (edge case, no xfade)")
    one_clip = [ClipInput(img_dir / "frame_a.jpg", 0.0, result.audio_duration_s, "fade")]
    out1 = tmp / "single.mp4"
    render(one_clip, result.audio_path, out1, transition_s=T, total_s=result.audio_duration_s)
    check("single-frame video renders with audio",
          media.probe_streams(out1).has_audio and media.probe_duration(out1) > 0)

    print("\n[9] transition minimum enforced (spec rule 9)")
    try:
        render(clips, result.audio_path, tmp / "bad.mp4", transition_s=0.5,
               total_s=result.audio_duration_s)
        check("stitch refuses transitions < 0.8 s", False)
    except StitchError as e:
        check("stitch refuses transitions < 0.8 s", "0.8" in str(e))

    print("\n[10] transition conflicts require explicit resolution")
    # Shrink frame A's hold below T by moving frame B's anchor earlier.
    tight_matches = match_anchors(to_specs(frames), NARRATION)
    tight_result, tight_times, tight_recovery = build_synthetic(tmp)
    tight_timeline = build_timeline(
        tight_matches, NARRATION, tight_times, tight_recovery, tight_result.audio_duration_s,
        sec_per_word=0.30, sec_per_word_calibrated=True, transition_s=1.0,
        default_transition="fade", sync_source="measured",
    )
    # Force a conflict: pretend transition is 4 s (longer than holds 0 and 1).
    conflict_timeline = build_timeline(
        tight_matches, NARRATION, tight_times, tight_recovery, tight_result.audio_duration_s,
        sec_per_word=0.30, sec_per_word_calibrated=True, transition_s=4.0,
        default_transition="fade", sync_source="measured",
    )
    check("conflicting transition flagged in timeline",
          len(conflict_timeline.transition_conflicts) > 0)
    tight_clips = [
        ClipInput(img_dir / f"{f.frame_id}.jpg", f.hold_start_s, f.hold_end_s, "fade")
        for f in tight_timeline.frames
    ]
    try:
        render(tight_clips, tight_result.audio_path, tmp / "conflict.mp4",
               transition_s=4.0, total_s=tight_result.audio_duration_s)
        check("stitch refuses overlapping transitions by default", False)
    except StitchError as e:
        check("stitch refuses overlapping transitions by default", "CONFLICT" in str(e))
    render(tight_clips, tight_result.audio_path, tmp / "conflict_ok.mp4",
           transition_s=4.0, total_s=tight_result.audio_duration_s,
           allow_transition_overlap=True)
    check("explicit --allow-transition-overlap resolves conflicts", (tmp / "conflict_ok.mp4").exists())

    print("\n[11] missing / ambiguous / out-of-order anchors fail loudly")
    repo = _make_repo_fixture(tmp)
    _write_narration(repo, NARRATION)

    make_frames_json(repo / "frames.json", [frame_entry("f1", "completely absent phrase here")])
    rc, outp = _run_estimate(repo)
    check("missing anchor -> non-zero exit + loud error",
          rc != 0 and "MISSING" in outp, f"rc={rc}")

    make_frames_json(repo / "frames.json", [
        frame_entry("f1", "the grey log and"), frame_entry("f2", "red fox jumps over")])
    rc, outp = _run_estimate(repo)
    check("out-of-order anchors -> non-zero exit + loud error",
          rc != 0 and "ORDER" in outp, f"rc={rc}")

    # A phrase that genuinely repeats in the narration needs an explicit
    # occurrence - never a silent first match.
    _write_narration(repo, "The grey log and the grey log and the grey log.")
    make_frames_json(repo / "frames.json", [frame_entry("f1", "the grey log and")])
    rc, outp = _run_estimate(repo)
    check("repeated phrase without occurrence -> non-zero exit + AMBIGUOUS",
          rc != 0 and "AMBIGUOUS" in outp, f"rc={rc}")
    make_frames_json(
        repo / "frames.json",
        [frame_entry("f1", "the grey log and", occurrence=1),
         frame_entry("f2", "the grey log and", occurrence=2)],
    )
    rc, outp = _run_estimate(repo)
    check("explicit occurrence selects the right appearance", rc == 0, f"rc={rc}")

    _write_narration(repo, NARRATION)
    good = [frame_entry(fid, phrase) for fid, phrase, _ in ANCHORS]
    make_frames_json(repo / "frames.json", good)
    rc, outp = _run_estimate(repo)
    check("valid anchors -> estimate passes offline", rc == 0, f"rc={rc}")
    check("estimate is labelled uncalibrated/estimate, never measured",
          "estimate" in outp.lower() and "sec_per_word" in outp)

    print("\n[12] hand-set durations rejected (spec rule 3)")
    bad = [frame_entry("f1", "red fox jumps over")]
    bad[0]["duration"] = 5.0
    make_frames_json(repo / "frames.json", bad)
    rc, outp = _run_estimate(repo)
    check("frame entry with hand-set duration -> rejected loudly",
          rc != 0 and "duration" in outp.lower(), f"rc={rc}")

    print("\n[13] Phase 5 render path reuses artifacts + demands reduced-precision approval")
    # Build a valid measured-artifact set (stand-in), then run the REAL run.py --out.
    _write_narration(repo, NARRATION)
    make_frames_json(repo / "frames.json", good)
    frames_dir = repo / "assets" / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for fid, _, _ in ANCHORS:
        make_frame_png(frames_dir / f"{fid}.jpg", COLORS[fid])

    def write_artifacts(mode: str):
        standin = StandInTTS(sec_per_word=0.30, jitter=True, mode=mode)
        res = standin.synthesize(NARRATION, repo / "build" / "narration.wav", boundary="WordBoundary")
        (repo / "build").mkdir(exist_ok=True)
        (repo / "build" / "word_events.json").write_text(res.to_json(), encoding="utf-8")
        cfg2 = load_config(repo)
        from vidsync.manifest import manifest_payload

        (repo / "build" / "manifest.json").write_text(
            json.dumps(
                {
                    "manifest": manifest_payload(NARRATION, cfg2, str(res.engine["version"])),
                    "engine_name": res.engine["name"],
                    "edge_tts_version": str(res.engine["version"]),
                }
            ),
            encoding="utf-8",
        )
        # run.py expects narration.mp3 name; point it at our wav copy.
        shutil.copy(res.audio_path, repo / "build" / "narration.mp3")

    write_artifacts("WordBoundary")
    rc, outp = _run_out(repo, tmp / "reuse.mp4", extra=[])
    check("run.py --out reuses Phase 4 artifacts and renders", rc == 0, f"rc={rc}")
    check("render output produced", (tmp / "reuse.mp4").exists())
    check("no synthesis happened during reuse", "REUSING" in outp)

    # Stale artifacts (narration changed) must refuse.
    _write_narration(repo, NARRATION + " Extra words appear now.")
    rc, outp = _run_out(repo, tmp / "stale.mp4", extra=[])
    check("stale artifacts -> refused loudly", rc != 0 and "STALE" in outp, f"rc={rc}")
    _write_narration(repo, NARRATION)

    # Reduced precision -> approval gate.
    write_artifacts("SentenceBoundary")
    rc, outp = _run_out(repo, tmp / "reduced.mp4", extra=[])
    check("reduced precision without approval -> refused", rc != 0 and "REDUCED" in outp, f"rc={rc}")
    rc, outp = _run_out(repo, tmp / "reduced.mp4", extra=["--allow-reduced-precision"])
    check("explicit approval renders with reduced precision", rc == 0, f"rc={rc}")

    print("\n[14] narration extraction: sources stripped, markdown rejected")
    _write_narration(repo, NARRATION, sources="## Sources\n- https://example.com\n")
    rc, outp = _run_estimate(repo)
    check("## Sources after END marker is stripped before synthesis", rc == 0, f"rc={rc}")
    _write_narration(repo, "The **bold** fox jumps over the log and lands.")
    rc, outp = _run_estimate(repo)
    check("markdown in speakable text rejected (Gate 1)", rc != 0 and "markdown" in outp.lower(),
          f"rc={rc}")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _make_repo_fixture(tmp: Path) -> Path:
    repo = tmp / "repo"
    (repo / "kaggle").mkdir(parents=True, exist_ok=True)
    (repo / "narration").mkdir(exist_ok=True)
    shutil.copy(REPO_ROOT / "kaggle" / "config.yaml", repo / "kaggle" / "config.yaml")
    return repo


def _write_narration(repo: Path, text: str, sources: str = "") -> None:
    (repo / "narration" / "narration.md").write_text(
        "# Test narration\n\n<!-- TTS:START -->\n"
        + text
        + "\n<!-- TTS:END -->\n\n"
        + sources,
        encoding="utf-8",
    )


def _run_estimate(repo: Path):
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "kaggle" / "run.py"), "--estimate", "--repo", str(repo)],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=120,
    )
    return proc.returncode, proc.stdout.decode("utf-8", "replace")


def _run_out(repo: Path, out_file: Path, extra):
    proc = subprocess.run(
        [sys.executable, str(REPO_ROOT / "kaggle" / "run.py"), "--out", str(out_file),
         "--repo", str(repo)] + list(extra),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600,
    )
    return proc.returncode, proc.stdout.decode("utf-8", "replace")


if __name__ == "__main__":
    sys.exit(main())
