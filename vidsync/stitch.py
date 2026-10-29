"""Stitching: measured timeline -> video, with visible cross-fade transitions.

Camera is locked off (spec rule 8): still frames, no zoom, no drift.
Transitions are real blends with duration >= 0.8 s (spec rule 9), placed so
the incoming frame is fully visible exactly at its anchor timestamp:

    transition window = [anchor_t - T, anchor_t]

A transition therefore never obscures the beat it introduces: the new
picture is 100% on screen when its anchor words are spoken.

Clip-length math for a chained ffmpeg xfade graph (N frames, boundaries
b_1..b_{N-1} = hold starts of frames 1..N-1, measured narration end E,
transition-into-frame-i duration t_i):

    d_0         = b_1                            (first clip; its last t_1 is the blend)
    d_i         = b_{i+1} - b_i + t_i            (middle clips carry incoming overlap)
    d_{N-1}     = E - b_{N-1} + t_{N-1}          (last clip: window + incoming overlap)
    xfade offset at boundary i = b_i - t_i       (global seconds from video start)

Resulting video length == E exactly, so picture covers [0, E] with no gaps.

Transitions that would overlap a neighbouring beat are REFUSED unless
--allow-transition-overlap is given (explicit reviewed decision).  Even
then, a transition can never start before t=0: boundary durations are
clamped to the anchor time itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from . import media

MIN_TRANSITION_S = 0.8


class StitchError(Exception):
    """Refused render (conflicts, bad inputs, ffmpeg failure)."""


@dataclass
class ClipInput:
    image_path: Path
    hold_start_s: float
    hold_end_s: float
    transition: str = "fade"


def _incoming_durations(
    clips: Sequence[ClipInput], transition_s: float, allow_overlap: bool
) -> List[float]:
    """Per-boundary (incoming) transition duration; index 0 is unused (0.0)."""
    t_list = [0.0]
    for i in range(1, len(clips)):
        b_i = clips[i].hold_start_s
        if allow_overlap:
            # Explicit override: clamp so the transition never starts before
            # t=0 (offset = b_i - t_i >= 0).
            t_list.append(min(transition_s, b_i))
        else:
            # Conflicts were already refused above, which guarantees
            # transition_s <= previous hold <= b_i.
            t_list.append(transition_s)
    return t_list


def _clip_durations(
    clips: Sequence[ClipInput], total_s: float, t_list: Sequence[float]
) -> List[float]:
    """Per-clip input durations for the xfade chain (see module docstring)."""
    n = len(clips)
    if n == 1:
        return [total_s]
    ds: List[float] = [clips[1].hold_start_s]  # d_0 = b_1
    for i in range(1, n - 1):
        b_i = clips[i].hold_start_s
        b_next = clips[i + 1].hold_start_s
        ds.append(b_next - b_i + t_list[i])
    ds.append(total_s - clips[n - 1].hold_start_s + t_list[n - 1])
    return ds


def render(
    clips: Sequence[ClipInput],
    audio_path: Path,
    out_path: Path,
    *,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    transition_s: float = 1.0,
    total_s: Optional[float] = None,
    allow_transition_overlap: bool = False,
    ffmpeg_bin: Optional[str] = None,
) -> Path:
    """Render clips + narration audio to out_path (mp4)."""
    if not clips:
        raise StitchError("no clips to render")
    if transition_s < MIN_TRANSITION_S:
        raise StitchError(
            f"transition duration {transition_s}s < {MIN_TRANSITION_S}s minimum "
            "(spec rule 9: transitions must be visible)"
        )
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise StitchError(f"audio not found: {audio_path}")
    out_path = Path(out_path)

    if total_s is None:
        total_s = media.probe_duration(audio_path)
    if not (total_s > 0):
        raise StitchError("total duration must be measured and non-zero")

    # Validate windows: strict increase, positive holds, transition conflicts.
    prev = -1.0
    conflicts: List[str] = []
    for c in clips:
        if not Path(c.image_path).exists():
            raise StitchError(f"frame image missing: {c.image_path}")
        if c.hold_start_s <= prev:
            raise StitchError(
                f"clip hold starts must be strictly increasing (got {c.hold_start_s} after {prev})"
            )
        if c.hold_end_s <= c.hold_start_s:
            raise StitchError(
                f"clip {c.image_path.name}: empty hold [{c.hold_start_s}, {c.hold_end_s})"
            )
        if c.hold_start_s > 0 and prev >= 0:
            # The incoming transition [b-t, b] must fit inside the previous
            # frame's hold [prev, b]; otherwise transitions overlap and the
            # previous beat is never fully visible (silent frame loss).
            window = c.hold_start_s - prev
            if window < transition_s:
                conflicts.append(
                    f"{c.image_path.name}: previous hold {window:.2f}s < transition "
                    f"{transition_s:.2f}s - transitions would overlap"
                )
        prev = c.hold_start_s
    if clips[0].hold_start_s != 0.0:
        raise StitchError("first clip must cover from time zero")
    last_end = clips[-1].hold_end_s
    if abs(last_end - total_s) > 0.05:
        raise StitchError(
            f"last hold ends at {last_end:.3f}s but measured narration end is {total_s:.3f}s"
        )
    if conflicts and not allow_transition_overlap:
        raise StitchError(
            "TRANSITION CONFLICT(S) - resolve explicitly (adjust anchors/transition "
            "duration, or pass --allow-transition-overlap to force):\n  "
            + "\n  ".join(conflicts)
        )

    ff = ffmpeg_bin or media.find_ffmpeg()
    n = len(clips)
    t_list = _incoming_durations(clips, transition_s, allow_transition_overlap)
    durations = _clip_durations(clips, total_s, t_list)

    cmd: List[str] = [ff, "-hide_banner", "-loglevel", "error", "-y"]
    for c, d in zip(clips, durations):
        cmd += ["-loop", "1", "-framerate", str(fps), "-t", f"{d:.6f}", "-i", str(c.image_path)]
    cmd += ["-i", str(audio_path)]
    audio_idx = n

    # Per-input normalization; AVTB timebase is what xfade expects.
    parts: List[str] = []
    for i in range(n):
        parts.append(
            f"[{i}:v]scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,setsar=1,fps={fps},"
            f"format=yuv420p,settb=AVTB[v{i}]"
        )
    if n == 1:
        label = "v0"
    else:
        cur = "v0"
        for i in range(1, n):
            b_i = clips[i].hold_start_s
            t_i = t_list[i]
            offset = b_i - t_i
            if offset < 0 or t_i <= 0:
                raise StitchError(
                    f"transition at {b_i:.2f}s cannot be placed (duration {t_i:.3f}s)"
                )
            out = f"x{i}"
            trans = clips[i].transition or "fade"
            parts.append(
                f"[{cur}][v{i}]xfade=transition={trans}:duration={t_i:.6f}:"
                f"offset={offset:.6f}[{out}]"
            )
            cur = out
        label = cur
    parts.append(f"[{label}]null[vout]")

    filter_graph = ";".join(parts)
    cmd += [
        "-filter_complex", filter_graph,
        "-map", "[vout]",
        "-map", f"{audio_idx}:a:0",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "160k",
        "-t", f"{total_s:.6f}",
        "-movflags", "+faststart",
        str(out_path),
    ]

    proc = media._run(cmd)  # noqa: SLF001 - same package
    if proc.returncode != 0 or not out_path.exists():
        raise StitchError(
            "ffmpeg render failed:\n" + proc.stderr.decode("utf-8", "replace")[-2000:]
        )

    # Verify the produced file before declaring success.
    streams = media.probe_streams(out_path)
    if not (streams.has_video and streams.has_audio):
        raise StitchError(
            f"render verification failed: video={streams.has_video} audio={streams.has_audio}"
        )
    out_dur = media.probe_duration(out_path)
    if abs(out_dur - total_s) > 1.0:
        raise StitchError(
            f"rendered duration {out_dur:.2f}s deviates from narration {total_s:.2f}s by > 1s"
        )
    return out_path
