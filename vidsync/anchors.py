"""Anchor matching: frames.json entries against narration text.

Rules enforced (spec, Phase 2):
  * anchor_phrase is copied VERBATIM from the narration - 3 to 8 words.
  * occurrence selects which appearance of a repeated phrase (1-based;
    negative counts from the end).  A repeated phrase without an explicit
    occurrence is an AMBIGUITY error, never a silent first-match.
  * Anchor start positions must be distinct and strictly increasing.
  * A missing phrase is a MISSING error - never silently dropped.
  * Hand-set durations in frame entries are rejected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .textnorm import Word, find_phrase_occurrences, slice_text, tokenize

REQUIRED_KEYS = ("id", "anchor_phrase", "image_prompt", "negative_prompt")
OPTIONAL_KEYS = ("occurrence", "transition")
FORBIDDEN_KEYS = ("duration", "hold_s", "hold", "duration_s", "seconds")


class AnchorError(Exception):
    """Raised when frames.json or anchor matching fails validation."""


@dataclass
class FrameSpec:
    id: str
    anchor_phrase: str
    image_prompt: str
    negative_prompt: str
    occurrence: Optional[int] = None
    transition: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass
class AnchorMatch:
    frame: FrameSpec
    match_word_start: int  # index into narration words (first significant word)
    match_word_end: int  # exclusive, into narration words
    matched_text: str  # verbatim slice of the narration
    note: str = ""


def load_frames_json(path: Path) -> List[FrameSpec]:
    """Load and structurally validate frames.json.  Order = narration order."""
    p = Path(path)
    if not p.exists():
        raise AnchorError(f"frames.json not found: {p}")
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except json.JSONDecodeError as e:
        raise AnchorError(f"frames.json is not valid JSON: {e}") from e

    if isinstance(data, dict) and "frames" in data:
        entries = data["frames"]
    else:
        entries = data
    if not isinstance(entries, list) or not entries:
        raise AnchorError("frames.json must contain a non-empty list of frames")

    frames: List[FrameSpec] = []
    seen_ids: set = set()
    for i, entry in enumerate(entries):
        where = f"frames.json entry #{i}"
        if not isinstance(entry, dict):
            raise AnchorError(f"{where}: must be an object")
        for key in FORBIDDEN_KEYS:
            if key in entry:
                raise AnchorError(
                    f"{where}: forbidden key '{key}' - frame durations are MEASURED, "
                    "never hand-set (spec rule 3)"
                )
        unknown = set(entry) - set(REQUIRED_KEYS) - set(OPTIONAL_KEYS)
        if unknown:
            raise AnchorError(f"{where}: unknown keys {sorted(unknown)}")
        for key in REQUIRED_KEYS:
            if key not in entry or not isinstance(entry[key], str) or not entry[key].strip():
                raise AnchorError(f"{where}: missing/empty required key '{key}'")

        fid = entry["id"].strip()
        if fid in seen_ids:
            raise AnchorError(f"{where}: duplicate id '{fid}'")
        seen_ids.add(fid)

        phrase = entry["anchor_phrase"].strip()
        n_words = len([t for t in phrase.split() if t.strip(".,;:!?\"'()[]{}")])
        if not (3 <= n_words <= 8):
            raise AnchorError(
                f"{where} (id '{fid}'): anchor_phrase must be 3-8 words, got {n_words}: {phrase!r}"
            )

        occ = entry.get("occurrence")
        if occ is not None:
            if not isinstance(occ, int) or occ == 0:
                raise AnchorError(
                    f"{where} (id '{fid}'): occurrence must be a non-zero integer, got {occ!r}"
                )

        trans = entry.get("transition")
        if trans is not None and (not isinstance(trans, str) or not trans.strip()):
            raise AnchorError(
                f"{where} (id '{fid}'): transition must be a non-empty string when given"
            )

        frames.append(
            FrameSpec(
                id=fid,
                anchor_phrase=phrase,
                image_prompt=entry["image_prompt"].strip(),
                negative_prompt=entry["negative_prompt"].strip(),
                occurrence=occ,
                transition=trans,
                raw=entry,
            )
        )
    return frames


def _select_occurrence(
    frame: FrameSpec, hits: List[int], words: List[Word]
) -> Tuple[int, str]:
    """Pick one occurrence; return (start_word_index, note)."""
    if not hits:
        raise AnchorError(f"MISSING anchor '{frame.anchor_phrase}' (id '{frame.id}')")
    if len(hits) == 1:
        if frame.occurrence not in (None, 1, -1):
            raise AnchorError(
                f"occurrence {frame.occurrence} requested for id '{frame.id}' but the "
                f"phrase appears only once in the narration"
            )
        return hits[0], "single occurrence"

    # Phrase repeats: an explicit occurrence is mandatory.
    if frame.occurrence is None:
        positions = ", ".join(str(h) for h in hits)
        raise AnchorError(
            f"AMBIGUOUS anchor '{frame.anchor_phrase}' (id '{frame.id}'): appears "
            f"{len(hits)} times (word starts {positions}). Set 'occurrence' explicitly "
            "(1-based; negative counts from the end)."
        )
    occ = frame.occurrence
    if occ > 0:
        idx = occ - 1
    else:
        idx = len(hits) + occ
    if not (0 <= idx < len(hits)):
        raise AnchorError(
            f"occurrence {occ} out of range for id '{frame.id}': phrase appears "
            f"{len(hits)} times"
        )
    return hits[idx], f"occurrence {occ} of {len(hits)}"


def match_anchors(frames: List[FrameSpec], narration_text: str) -> List[AnchorMatch]:
    """Match every frame anchor to the narration.  Raises AnchorError on any problem."""
    words = tokenize(narration_text)
    if not words:
        raise AnchorError("narration text is empty")

    matches: List[AnchorMatch] = []
    prev_start = -1
    for frame in frames:
        hits = find_phrase_occurrences(words, frame.anchor_phrase)
        start, note = _select_occurrence(frame, hits, words)
        if start <= prev_start:
            raise AnchorError(
                f"ORDER error: anchor for id '{frame.id}' starts at word {start}, "
                f"not strictly after previous anchor (word {prev_start}). Frames must "
                "appear in narration order with distinct, increasing start positions."
            )
        prev_start = start

        # End of phrase: span the same number of significant words as the phrase.
        phrase_sig = [t for t in frame.anchor_phrase.split() if t.strip(".,;:!?\"'()[]{}")]
        end = start
        remaining = len(phrase_sig)
        wi = start
        while wi < len(words) and remaining > 0:
            if words[wi].norm:
                remaining -= 1
            wi += 1
            end = wi
        matched_text = slice_text(narration_text, words, start, end)
        matches.append(
            AnchorMatch(
                frame=frame,
                match_word_start=start,
                match_word_end=end,
                matched_text=matched_text,
                note=note,
            )
        )
    return matches


def anchor_word_index(match: AnchorMatch) -> int:
    """Index of the first spoken word of the anchor phrase."""
    return match.match_word_start
