"""ffmpeg/ffprobe discovery and small media helpers.

Resolution order for ffmpeg:
  1. $FFMPEG_BIN
  2. ffmpeg on PATH (Kaggle images ship one)
  3. imageio-ffmpeg's bundled binary (workspaces without system ffmpeg)
ffprobe is optional; when absent, stream/duration info is parsed from
`ffmpeg -i` stderr, which is sufficient for this pipeline's checks.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple


class MediaError(Exception):
    """ffmpeg/ffprobe failure or missing binary."""


_FFMPEG_CACHE: Optional[str] = None
_FFPROBE_CACHE: Optional[str] = None


def find_ffmpeg() -> str:
    global _FFMPEG_CACHE
    if _FFMPEG_CACHE:
        return _FFMPEG_CACHE
    cand = os.environ.get("FFMPEG_BIN")
    if cand and Path(cand).exists():
        _FFMPEG_CACHE = cand
        return cand
    cand = shutil.which("ffmpeg")
    if cand:
        _FFMPEG_CACHE = cand
        return cand
    try:
        import imageio_ffmpeg  # type: ignore

        cand = imageio_ffmpeg.get_ffmpeg_exe()
        if cand and Path(cand).exists():
            _FFMPEG_CACHE = cand
            return cand
    except Exception:
        pass
    raise MediaError(
        "ffmpeg not found. Set FFMPEG_BIN, install ffmpeg, or `pip install imageio-ffmpeg`."
    )


def find_ffprobe() -> Optional[str]:
    global _FFPROBE_CACHE
    if _FFPROBE_CACHE:
        return _FFPROBE_CACHE
    cand = os.environ.get("FFPROBE_BIN")
    if cand and Path(cand).exists():
        _FFPROBE_CACHE = cand
        return cand
    cand = shutil.which("ffprobe")
    if cand:
        _FFPROBE_CACHE = cand
        return cand
    return None


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False
        )
    except FileNotFoundError as e:
        raise MediaError(f"failed to execute {cmd[0]}: {e}") from e


def _parse_duration_from_ffmpeg_stderr(stderr: str) -> Optional[float]:
    m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    if not m:
        return None
    h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
    return h * 3600 + mi * 60 + s


@dataclass
class StreamInfo:
    has_video: bool
    has_audio: bool
    video_streams: int
    audio_streams: int


def probe_duration(path: Path) -> float:
    """Media duration in seconds, measured from the file (never estimated)."""
    p = str(Path(path))
    ffprobe = find_ffprobe()
    if ffprobe:
        proc = _run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", p]
        )
        if proc.returncode == 0:
            out = proc.stdout.decode("utf-8", "replace").strip()
            try:
                dur = float(out.splitlines()[0])
                if dur > 0:
                    return dur
            except (ValueError, IndexError):
                pass
    proc = _run([find_ffmpeg(), "-hide_banner", "-i", p])
    stderr = proc.stderr.decode("utf-8", "replace")
    dur = _parse_duration_from_ffmpeg_stderr(stderr)
    if dur is None or dur <= 0:
        raise MediaError(f"could not measure duration of {p}")
    return dur


def probe_streams(path: Path) -> StreamInfo:
    """Detect audio/video streams from the container."""
    p = str(Path(path))
    ffprobe = find_ffprobe()
    if ffprobe:
        proc = _run(
            [ffprobe, "-v", "error", "-show_entries",
             "stream=codec_type", "-of", "csv=p=0", p]
        )
        if proc.returncode == 0:
            types = proc.stdout.decode("utf-8", "replace").split()
            nv = sum(1 for t in types if t.strip() == "video")
            na = sum(1 for t in types if t.strip() == "audio")
            return StreamInfo(nv > 0, na > 0, nv, na)
    proc = _run([find_ffmpeg(), "-hide_banner", "-i", p])
    stderr = proc.stderr.decode("utf-8", "replace")
    nv = len(re.findall(r"Stream #\d+:\d+[^:]*: Video:", stderr))
    na = len(re.findall(r"Stream #\d+:\d+[^:]*: Audio:", stderr))
    return StreamInfo(nv > 0, na > 0, nv, na)


def extract_frame(video: Path, t: float, out_png: Path) -> None:
    """Extract one accurate frame at time t (seconds) as PNG."""
    cmd = [
        find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y",
        "-i", str(video),
        "-ss", f"{t:.4f}",
        "-frames:v", "1",
        str(out_png),
    ]
    proc = _run(cmd)
    if proc.returncode != 0 or not Path(out_png).exists():
        raise MediaError(
            f"frame extraction failed at t={t}: {proc.stderr.decode('utf-8','replace')[-400:]}"
        )


def mean_rgb(png_path: Path) -> Tuple[float, float, float]:
    """Mean RGB of an image, as floats 0..255.  Requires Pillow."""
    from PIL import Image  # local import: only needed by QC/selftest paths

    with Image.open(png_path) as im:
        im = im.convert("RGB")
        im = im.resize((64, 36))  # mean is invariant enough; keeps it cheap
        px = list(im.getdata())
    n = len(px)
    r = sum(p[0] for p in px) / n
    g = sum(p[1] for p in px) / n
    b = sum(p[2] for p in px) / n
    return (r, g, b)
