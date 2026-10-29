"""TTS backends.

RealEdgeTTS  - the only real synthesis path.  Refuses to run outside Kaggle
               (spec: the workspace must NEVER run TTS).  Explicitly requests
               WordBoundary metadata and records what the service actually
               returned - never assumed.

StandInTTS   - deterministic offline backend for tools/selftest.py.  Produces
               synthetic boundary events and a WAV of matching length.  It
               verifies pipeline logic ONLY, never actual engine behavior.
"""

from __future__ import annotations

import itertools
import json
import math
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import media

TICKS_PER_SECOND = 10_000_000


class TTSError(Exception):
    """Synthesis failure or environment refusal."""


def is_kaggle() -> bool:
    """True when running inside a Kaggle notebook environment."""
    import os

    if os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return True
    return Path("/kaggle/working").exists()


def require_kaggle_for_synthesis() -> None:
    if not is_kaggle():
        raise TTSError(
            "REFUSING to synthesize outside Kaggle (spec: the workspace must NEVER "
            "run TTS). Run this step in a Kaggle notebook with internet enabled."
        )


@dataclass
class BoundaryEvent:
    type: str  # "WordBoundary" | "SentenceBoundary"
    offset_s: float
    duration_s: float
    text: str

    @property
    def end_s(self) -> float:
        return self.offset_s + self.duration_s


@dataclass
class SynthesisResult:
    audio_path: Path
    events: List[BoundaryEvent]
    requested_boundary: str
    returned_boundary_types: List[str]  # distinct observed types, in order
    engine: Dict[str, Any]
    audio_duration_s: float  # measured from the audio file
    text_chunks: int = 1  # internal text splits used by the engine
    notes: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "audio_path": str(self.audio_path),
                "requested_boundary": self.requested_boundary,
                "returned_boundary_types": self.returned_boundary_types,
                "engine": self.engine,
                "audio_duration_s": self.audio_duration_s,
                "text_chunks": self.text_chunks,
                "notes": self.notes,
                "events": [
                    {
                        "type": e.type,
                        "offset_s": e.offset_s,
                        "duration_s": e.duration_s,
                        "text": e.text,
                    }
                    for e in self.events
                ],
            },
            indent=2,
        )


def _measure_duration(audio_path: Path) -> float:
    """Non-zero measured audio duration or TTSError."""
    dur = media.probe_duration(audio_path)
    if not (dur > 0):
        raise TTSError(f"audio duration is not non-zero: {dur}")
    return dur


class RealEdgeTTS:
    """Real synthesis via edge-tts, Kaggle-only, one continuous request."""

    def __init__(self, voice: str, rate: str = "+0%", volume: str = "+0%", pitch: str = "+0Hz"):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        self.pitch = pitch

    def synthesize(
        self,
        text: str,
        out_audio: Path,
        boundary: str = "WordBoundary",
    ) -> SynthesisResult:
        require_kaggle_for_synthesis()
        if boundary not in ("WordBoundary", "SentenceBoundary"):
            raise TTSError(f"invalid boundary request {boundary!r}")

        try:
            import edge_tts
        except ImportError as e:
            raise TTSError("edge-tts is not installed (pip install edge-tts)") from e

        # The boundary mode is requested EXPLICITLY.  edge-tts 7.x defaults to
        # SentenceBoundary; spec rule: never rely on an engine default.
        comm = edge_tts.Communicate(
            text,
            self.voice,
            rate=self.rate,
            volume=self.volume,
            pitch=self.pitch,
            boundary=boundary,
        )
        # edge-tts 7.x stores `texts` as a GENERATOR consumed by the stream
        # itself. Never len() it directly (TypeError) and never exhaust it
        # (the engine would receive zero chunks). tee() lets us count without
        # depriving the engine of its chunks. Older versions used a list.
        _texts = getattr(comm, "texts", None)
        if _texts is None:
            text_chunks = 1
        else:
            try:
                text_chunks = len(_texts)
            except TypeError:
                _count_view, comm.texts = itertools.tee(_texts)
                text_chunks = sum(1 for _ in _count_view)

        out_audio = Path(out_audio)
        out_audio.parent.mkdir(parents=True, exist_ok=True)
        events: List[BoundaryEvent] = []
        observed_types: List[str] = []
        with open(out_audio, "wb") as fh:
            for chunk in comm.stream_sync():
                ctype = chunk.get("type")
                if ctype == "audio":
                    fh.write(chunk["data"])
                elif ctype in ("WordBoundary", "SentenceBoundary"):
                    if ctype not in observed_types:
                        observed_types.append(ctype)
                    events.append(
                        BoundaryEvent(
                            type=ctype,
                            offset_s=float(chunk["offset"]) / TICKS_PER_SECOND,
                            duration_s=float(chunk["duration"]) / TICKS_PER_SECOND,
                            text=str(chunk.get("text", "")),
                        )
                    )

        if not out_audio.exists() or out_audio.stat().st_size == 0:
            raise TTSError("no audio was produced")
        audio_duration = _measure_duration(out_audio)

        notes: List[str] = []
        if text_chunks > 1:
            notes.append(
                f"engine split the text into {text_chunks} internal chunks "
                "(still one continuous synthesis stream; offsets are compensated)"
            )
        if not events:
            notes.append("NO boundary events were returned by the service")
        elif boundary not in observed_types:
            notes.append(
                f"requested {boundary} but observed types {observed_types} "
                "- engine did not honour the requested boundary mode"
            )

        return SynthesisResult(
            audio_path=out_audio,
            events=events,
            requested_boundary=boundary,
            returned_boundary_types=observed_types,
            engine={
                "name": "edge-tts",
                "version": getattr(edge_tts, "__version__", "unknown"),
                "voice": self.voice,
                "rate": self.rate,
                "volume": self.volume,
                "pitch": self.pitch,
            },
            audio_duration_s=audio_duration,
            text_chunks=text_chunks,
            notes=notes,
        )


class StandInTTS:
    """Deterministic offline stand-in used ONLY by tools/selftest.py.

    Produces boundary events at synthetic, deliberately uneven times and a
    PCM WAV whose duration matches the last event end.  It exercises the
    pipeline's logic; it says nothing about the real engine.
    """

    def __init__(
        self,
        word_times: Optional[List[float]] = None,
        sec_per_word: float = 0.30,
        jitter: bool = True,
        mode: str = "WordBoundary",
        sample_rate: int = 24000,
    ):
        self.word_times = word_times
        self.sec_per_word = sec_per_word
        self.jitter = jitter
        self.mode = mode  # "WordBoundary" (word-level) or "SentenceBoundary" (sentence-level)
        self.sample_rate = sample_rate

    def synthesize(
        self,
        text: str,
        out_audio: Path,
        boundary: str = "WordBoundary",
    ) -> SynthesisResult:
        from .textnorm import tokenize

        words = [w for w in tokenize(text) if w.norm]
        if not words:
            raise TTSError("no spoken words in text")

        # Deliberately uneven word times: alternate multipliers so holds are
        # non-uniform and any slotting by the pipeline becomes visible.
        times: List[float] = []
        if self.word_times is not None:
            times = list(self.word_times)
            if len(times) < len(words):
                t = times[-1] if times else 0.0
                while len(times) < len(words):
                    t += self.sec_per_word
                    times.append(t)
        else:
            t = 0.0
            for i in range(len(words)):
                times.append(t)
                mult = 1.6 if (self.jitter and i % 3 == 0) else (0.7 if (self.jitter and i % 3 == 1) else 1.0)
                t += self.sec_per_word * mult

        events: List[BoundaryEvent] = []
        if self.mode == "SentenceBoundary":
            # One event per sentence: text carries several words.
            sent_words: List[List[int]] = [[]]
            for i, w in enumerate(words):
                sent_words[-1].append(i)
                if w.text.rstrip('"\'')[-1:] in ".!?":
                    sent_words.append([])
            sent_words = [s for s in sent_words if s]
            for s in sent_words:
                start = times[s[0]]
                end = times[s[-1]] + self.sec_per_word
                events.append(
                    BoundaryEvent(
                        type="SentenceBoundary",
                        offset_s=start,
                        duration_s=end - start,
                        text=" ".join(words[i].text for i in s),
                    )
                )
            returned = ["SentenceBoundary"]
        else:
            for i, w in enumerate(words):
                dur = self.sec_per_word * (0.8 if self.jitter else 1.0)
                events.append(
                    BoundaryEvent(
                        type="WordBoundary",
                        offset_s=times[i],
                        duration_s=dur,
                        text=w.text,
                    )
                )
            returned = ["WordBoundary"]

        duration = events[-1].end_s + 0.25  # small trailing silence
        out_audio = Path(out_audio)
        out_audio.parent.mkdir(parents=True, exist_ok=True)
        self._write_wav(out_audio, duration)

        return SynthesisResult(
            audio_path=out_audio,
            events=events,
            requested_boundary=boundary,
            returned_boundary_types=returned,
            engine={
                "name": "stand-in (SYNTHETIC - pipeline logic test only)",
                "version": "0",
                "voice": "stand-in",
                "rate": "+0%",
            },
            audio_duration_s=_measure_duration(out_audio),
            text_chunks=1,
            notes=["stand-in backend: synthetic events, not an engine measurement"],
        )

    def _write_wav(self, path: Path, duration_s: float) -> None:
        """Write a quiet 16-bit mono PCM WAV (pure stdlib)."""
        import struct

        n = max(1, int(duration_s * self.sample_rate))
        # Very quiet tone so the file has real, decodable content.
        frames = bytearray()
        for i in range(n):
            val = int(800 * math.sin(2 * math.pi * 220 * (i / self.sample_rate)))
            frames += struct.pack("<h", val)
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(self.sample_rate)
            wf.writeframes(bytes(frames))


def load_events_json(path: Path) -> SynthesisResult:
    """Reload preserved synthesis artifacts (Phase 4 -> Phase 5 reuse)."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    events = [
        BoundaryEvent(
            type=e["type"],
            offset_s=float(e["offset_s"]),
            duration_s=float(e["duration_s"]),
            text=e["text"],
        )
        for e in data["events"]
    ]
    audio_path = Path(data["audio_path"])
    return SynthesisResult(
        audio_path=audio_path,
        events=events,
        requested_boundary=data["requested_boundary"],
        returned_boundary_types=data["returned_boundary_types"],
        engine=data["engine"],
        audio_duration_s=float(data["audio_duration_s"]),
        text_chunks=int(data.get("text_chunks", 1)),
        notes=list(data.get("notes", [])),
    )
