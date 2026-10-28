"""Measured timing: word-time recovery, precision classification, timeline.

Core principle (spec, section 0): narration is the master timeline.
  * Frame duration = measured time between anchor timestamps.  Derived,
    never hand-set.
  * Holds come from successive measured anchor timestamps.  The final hold
    ends at the measured narration end.  Picture covers from time zero.
  * Measured timestamps and interpolated positions are always labelled
    differently (sync_source / timestamp_source).  Interpolation is never
    called word-accurate.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .anchors import AnchorMatch
from .textnorm import Word, norm_word, tokenize
from .tts import BoundaryEvent

SYNC_MEASURED_WORDS = "measured: word-boundary event timestamps"
SYNC_MEASURED_SENTENCES = "approximate: sentence-boundary events + interpolation"
SYNC_ESTIMATE = "estimate: sec_per_word (no synthesis; offline diagnostics only)"


@dataclass
class WordTime:
    index: int  # index into narration words
    t_start: Optional[float]  # None if not recovered
    t_end: Optional[float]
    source: str  # "measured" | "interpolated" | "unrecovered"
    event_text: str = ""


@dataclass
class RecoveryReport:
    expected_words: int
    recovered_words: int
    interpolated_words: int
    unrecovered_words: int
    coverage: float
    words_per_event_min: float
    words_per_event_max: float
    words_per_event_mean: float
    events_with_one_word: int
    event_count: int
    timestamps_monotonic: bool
    order_violations: List[str]
    correspondence_ok: bool
    unrecovered_list: List[str]

    def summary(self) -> str:
        return (
            f"source-word recovery: {self.recovered_words}/{self.expected_words} "
            f"({self.coverage * 100:.1f}% coverage), interpolated: {self.interpolated_words}, "
            f"unrecovered: {self.unrecovered_words}"
        )


def recover_word_times(
    source_text: str, events: List[BoundaryEvent]
) -> Tuple[List[WordTime], RecoveryReport]:
    """In-order greedy correspondence between source words and boundary events.

    Every event's text is matched (normalized) against the *next* unmatched
    source word(s).  A match counts as recovery only when the event text
    corresponds to the source word; event counts alone prove nothing (spec,
    Phase 0 step 5).  Source words without a recovered event receive
    interpolated timestamps between neighbours, explicitly labelled.
    """
    words = tokenize(source_text)
    sig = [w for w in words if w.norm]
    expected = len(sig)

    times: Dict[int, WordTime] = {
        w.index: WordTime(index=w.index, t_start=None, t_end=None, source="unrecovered")
        for w in words
    }

    # Validate event order first.
    order_violations: List[str] = []
    prev_start = -1.0
    for i, ev in enumerate(events):
        if ev.offset_s < 0 or ev.duration_s < 0:
            order_violations.append(f"event #{i}: negative offset/duration")
        if ev.offset_s < prev_start:
            order_violations.append(
                f"event #{i}: offset {ev.offset_s:.4f} goes backwards (prev {prev_start:.4f})"
            )
        prev_start = max(prev_start, ev.offset_s)

    # Greedy in-order recovery.
    cursor = 0  # index into sig
    events_with_one_word = 0
    words_per_event: List[int] = []
    unrecovered_event_words: List[str] = []
    for ev in events:
        ev_words = [t for t in ev.text.split() if t.strip(".,;:!?\"'()[]{}")]
        n = len([t for t in ev_words if norm_word(t)])
        words_per_event.append(max(n, 1))
        if n == 1:
            events_with_one_word += 1

        # Try to consume as many source words as the event claims, in order.
        consumed = 0
        ev_sig = [norm_word(t) for t in ev_words if norm_word(t)]
        n = max(len(ev_sig), 1)
        for j, ntok in enumerate(ev_sig):
            if cursor < expected and sig[cursor].norm == ntok:
                w = sig[cursor]
                if len(ev_sig) == 1:
                    times[w.index] = WordTime(
                        index=w.index,
                        t_start=ev.offset_s,
                        t_end=ev.end_s,
                        source="measured",
                        event_text=ev.text,
                    )
                else:
                    # Sentence/coarse event: only the event edges are measured.
                    # Word positions INSIDE the span are interpolated - real
                    # timestamps, but never called word-accurate.
                    frac0 = j / n
                    frac1 = (j + 1) / n
                    times[w.index] = WordTime(
                        index=w.index,
                        t_start=ev.offset_s + (ev.end_s - ev.offset_s) * frac0,
                        t_end=ev.offset_s + (ev.end_s - ev.offset_s) * frac1,
                        source="interpolated",
                        event_text=ev.text,
                    )
                cursor += 1
                consumed += 1
            else:
                unrecovered_event_words.append(ntok)
        if consumed == 0 and ev.text:
            unrecovered_event_words.append(f"[event unmatched: {ev.text!r}]")

    recovered = sum(1 for w in sig if times[w.index].source == "measured")
    coarse = sum(1 for w in sig if times[w.index].source == "interpolated")

    # Interpolate any remaining unrecovered significant words between known
    # neighbours so every anchor has SOME timestamp, explicitly labelled.
    sig_idx = [w.index for w in sig]
    known = [i for i in sig_idx if times[i].source in ("measured", "interpolated")]
    for k, wi in enumerate(sig_idx):
        if times[wi].source != "unrecovered":
            continue
        before = [i for i in known if i < wi]
        after = [i for i in known if i > wi]
        t_b = times[before[-1]].t_end if before else None
        t_a = times[after[0]].t_start if after else None
        if t_b is not None and t_a is not None and t_a > t_b:
            frac = 1.0 / (after[0] - before[-1])
            t = t_b + (t_a - t_b) * frac
        elif t_b is not None:
            t = t_b
        elif t_a is not None:
            t = t_a
        else:
            t = 0.0
        times[wi] = WordTime(
            index=wi, t_start=t, t_end=t, source="interpolated", event_text=""
        )
    interpolated = sum(1 for w in sig if times[w.index].source == "interpolated")
    unrecovered = sum(1 for w in sig if times[w.index].source == "unrecovered")

    monotonic = len(order_violations) == 0
    correspondence_ok = recovered == expected and not unrecovered_event_words

    if words_per_event:
        wp_min = float(min(words_per_event))
        wp_max = float(max(words_per_event))
        wp_mean = sum(words_per_event) / len(words_per_event)
    else:
        wp_min = wp_max = wp_mean = 0.0

    report = RecoveryReport(
        expected_words=expected,
        recovered_words=recovered,
        interpolated_words=interpolated,
        unrecovered_words=unrecovered,
        coverage=(recovered / expected) if expected else 0.0,
        words_per_event_min=wp_min,
        words_per_event_max=wp_max,
        words_per_event_mean=wp_mean,
        events_with_one_word=events_with_one_word,
        event_count=len(events),
        timestamps_monotonic=monotonic,
        order_violations=order_violations,
        correspondence_ok=correspondence_ok,
        unrecovered_list=unrecovered_event_words[:20],
    )
    return [times[w.index] for w in words], report


@dataclass
class TimelineFrame:
    index: int
    frame_id: str
    anchor_phrase: str
    occurrence: Optional[int]
    matched_text: str
    anchor_word_start: int
    anchor_start_s: float
    timestamp_source: str  # "measured" | "interpolated"
    hold_start_s: float
    hold_end_s: float
    hold_s: float
    est_hold_s: float
    words_covered: int
    anchor_precision: str  # "exact" | "approximate"
    transition: str
    transition_s: float
    status: str
    notes: List[str] = field(default_factory=list)


@dataclass
class Timeline:
    frames: List[TimelineFrame]
    narration_words: List[Word]
    audio_duration_s: float
    sync_source: str
    sec_per_word: float
    sec_per_word_calibrated: bool
    transition_s: float
    transition_conflicts: List[str]
    reduced_precision: bool
    frame_count: int  # == len(frames.json entries), never merged

    def est_runtime_s(self) -> float:
        return len([w for w in self.narration_words if w.norm]) * self.sec_per_word


def classify_anchor_precision(word_time: WordTime) -> str:
    """exact only for verified word-level measured timestamps.

    recover_word_times labels a word 'measured' only when a single-word
    boundary event corresponded to it; anything interpolated (including word
    positions inside coarse/sentence events) is approximate.
    """
    if word_time.source == "measured":
        return "exact"
    return "approximate"


def build_timeline(
    matches: List[AnchorMatch],
    narration_text: str,
    word_times: List[WordTime],
    recovery: RecoveryReport,
    audio_duration_s: float,
    *,
    sec_per_word: float,
    sec_per_word_calibrated: bool,
    transition_s: float,
    default_transition: str,
    sync_source: str,
) -> Timeline:
    """Derive per-frame holds from measured anchor timestamps.

    Window rules:
      * frame 0 covers from time zero (picture coverage from 0).
      * frame i>0 starts at its anchor timestamp.
      * every hold ends where the next begins; the last ends at the measured
        narration end (audio duration).
    """
    words = tokenize(narration_text)
    if not (audio_duration_s > 0):
        raise ValueError("audio_duration_s must be non-zero (measured)")

    # Locate each anchor's first word time + precision.
    frames: List[TimelineFrame] = []
    anchor_starts: List[float] = []
    for i, m in enumerate(matches):
        wt = word_times[m.match_word_start]
        anchor_t = wt.t_start if wt.t_start is not None else 0.0
        anchor_starts.append(anchor_t)
    for i in range(1, len(anchor_starts)):
        if anchor_starts[i] <= anchor_starts[i - 1]:
            raise ValueError(
                "anchor timestamps are not strictly increasing ("
                f"#{i} {anchor_starts[i]:.3f}s <= #{i - 1} {anchor_starts[i - 1]:.3f}s) - "
                "a metadata limitation must not collapse distinct visual beats; "
                "obtain better alignment or rework the anchors explicitly"
            )

    conflicts: List[str] = []
    reduced = False
    for i, m in enumerate(matches):
        wt = word_times[m.match_word_start]
        anchor_t = anchor_starts[i]
        precision = classify_anchor_precision(wt)
        if precision != "exact":
            reduced = True
        ts_source = wt.source if wt.source in ("measured", "interpolated") else "interpolated"
        if ts_source == "interpolated":
            precision = "approximate"
            reduced = True

        hold_start = 0.0 if i == 0 else anchor_starts[i]
        hold_end = anchor_starts[i + 1] if i + 1 < len(matches) else audio_duration_s
        hold_s = max(0.0, hold_end - hold_start)

        # Estimated hold: words covered in this window x sec_per_word.
        next_word_start = (
            matches[i + 1].match_word_start if i + 1 < len(matches) else len(words)
        )
        first_word = 0 if i == 0 else m.match_word_start
        words_covered = sum(
            1 for w in words[first_word:next_word_start] if w.norm
        )
        est_hold_s = words_covered * sec_per_word

        notes: List[str] = []
        if i == 0 and anchor_starts[0] > 0:
            notes.append(
                f"picture starts at 0.0; lead-in of {anchor_starts[0]:.2f}s before first anchor"
            )
        if hold_s < 1.5:
            notes.append(f"SHORT hold {hold_s:.2f}s - review readability (diagnostic, not a rule)")
        if hold_s > 12.0:
            notes.append(f"LONG hold {hold_s:.2f}s - review for staleness (diagnostic, not a rule)")
        # A transition conflict exists when this frame's hold is shorter than
        # the transition that LEAVES it (frames after the first) or, for frame
        # 0, shorter than the first transition.  The final frame has no
        # outgoing transition: a short final hold is a readability review
        # item, not a transition conflict.
        leaves_now = i < len(matches) - 1
        if hold_s < transition_s and leaves_now:
            conflicts.append(
                f"id '{m.frame.id}': hold {hold_s:.2f}s < transition {transition_s:.2f}s - "
                "transitions would overlap and this beat would never be fully "
                "visible; resolve explicitly (adjust anchors/transition or pass "
                "--allow-transition-overlap)"
            )
            notes.append("TRANSITION CONFLICT - see warnings")
        if not sec_per_word_calibrated:
            notes.append("est_hold_s uses UNCALIBRATED sec_per_word")

        status = "ok"
        if any("TRANSITION CONFLICT" in n for n in notes):
            status = "warning-transition-conflict"
        elif any(n.startswith("SHORT") for n in notes):
            status = "warning-short-hold"
        elif any(n.startswith("LONG") for n in notes):
            status = "warning-long-hold"

        trans = m.frame.transition or default_transition
        frames.append(
            TimelineFrame(
                index=i,
                frame_id=m.frame.id,
                anchor_phrase=m.frame.anchor_phrase,
                occurrence=m.frame.occurrence,
                matched_text=m.matched_text,
                anchor_word_start=m.match_word_start,
                anchor_start_s=anchor_t,
                timestamp_source=ts_source,
                hold_start_s=hold_start,
                hold_end_s=hold_end,
                hold_s=hold_s,
                est_hold_s=est_hold_s,
                words_covered=words_covered,
                anchor_precision=precision,
                transition=trans,
                transition_s=transition_s,
                status=status,
                notes=notes,
            )
        )

    return Timeline(
        frames=frames,
        narration_words=words,
        audio_duration_s=audio_duration_s,
        sync_source=sync_source,
        sec_per_word=sec_per_word,
        sec_per_word_calibrated=sec_per_word_calibrated,
        transition_s=transition_s,
        transition_conflicts=conflicts,
        reduced_precision=reduced,
        frame_count=len(frames),
    )


CSV_COLUMNS = [
    "id",
    "anchor_phrase",
    "occurrence",
    "matched_text",
    "anchor_word_start",
    "anchor_start_s",
    "hold_start_s",
    "hold_end_s",
    "hold_s",
    "est_hold_s",
    "words_covered",
    "anchor_precision",
    "timestamp_source",
    "transition",
    "transition_s",
    "status",
    "notes",
]


def write_timings_csv(timeline: Timeline, path: Path) -> None:
    """Write timings.csv.  Row count == frames.json entry count; no merged_* rows."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for f in timeline.frames:
            writer.writerow(
                {
                    "id": f.frame_id,
                    "anchor_phrase": f.anchor_phrase,
                    "occurrence": "" if f.occurrence is None else f.occurrence,
                    "matched_text": f.matched_text,
                    "anchor_word_start": f.anchor_word_start,
                    "anchor_start_s": f"{f.anchor_start_s:.3f}",
                    "hold_start_s": f"{f.hold_start_s:.3f}",
                    "hold_end_s": f"{f.hold_end_s:.3f}",
                    "hold_s": f"{f.hold_s:.3f}",
                    "est_hold_s": f"{f.est_hold_s:.3f}",
                    "words_covered": f.words_covered,
                    "anchor_precision": f.anchor_precision,
                    "timestamp_source": f.timestamp_source,
                    "transition": f.transition,
                    "transition_s": f"{f.transition_s:.2f}",
                    "status": f.status,
                    "notes": "; ".join(f.notes),
                }
            )


def timeline_to_json(timeline: Timeline, recovery: RecoveryReport) -> str:
    return json.dumps(
        {
            "sync_source": timeline.sync_source,
            "audio_duration_s": timeline.audio_duration_s,
            "sec_per_word": timeline.sec_per_word,
            "sec_per_word_calibrated": timeline.sec_per_word_calibrated,
            "transition_s": timeline.transition_s,
            "transition_conflicts": timeline.transition_conflicts,
            "reduced_precision": timeline.reduced_precision,
            "frame_count": timeline.frame_count,
            "est_runtime_s": timeline.est_runtime_s(),
            "recovery": {
                "expected_words": recovery.expected_words,
                "recovered_words": recovery.recovered_words,
                "interpolated_words": recovery.interpolated_words,
                "unrecovered_words": recovery.unrecovered_words,
                "coverage": recovery.coverage,
                "words_per_event_min": recovery.words_per_event_min,
                "words_per_event_max": recovery.words_per_event_max,
                "words_per_event_mean": recovery.words_per_event_mean,
                "events_with_one_word": recovery.events_with_one_word,
                "event_count": recovery.event_count,
                "timestamps_monotonic": recovery.timestamps_monotonic,
                "order_violations": recovery.order_violations,
                "correspondence_ok": recovery.correspondence_ok,
                "unrecovered_list": recovery.unrecovered_list,
            },
            "frames": [
                {
                    "index": f.index,
                    "id": f.frame_id,
                    "anchor_phrase": f.anchor_phrase,
                    "occurrence": f.occurrence,
                    "matched_text": f.matched_text,
                    "anchor_word_start": f.anchor_word_start,
                    "anchor_start_s": f.anchor_start_s,
                    "timestamp_source": f.timestamp_source,
                    "hold_start_s": f.hold_start_s,
                    "hold_end_s": f.hold_end_s,
                    "hold_s": f.hold_s,
                    "est_hold_s": f.est_hold_s,
                    "words_covered": f.words_covered,
                    "anchor_precision": f.anchor_precision,
                    "transition": f.transition,
                    "transition_s": f.transition_s,
                    "status": f.status,
                    "notes": f.notes,
                }
                for f in timeline.frames
            ],
        },
        indent=2,
    )
