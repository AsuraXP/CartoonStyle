"""Configuration loading for the narration-first pipeline.

kaggle/config.yaml is the single source of engine/visual settings.
estimate.sec_per_word is written ONLY from an actual Phase 0 Kaggle
measurement (never invented).  Until then it stays null and the offline
estimator falls back to a clearly-labelled diagnostic value.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Hard minimum transition duration (seconds) for flat 2D art.  Spec rule 9.
MIN_TRANSITION_S = 0.8

DEFAULTS: Dict[str, Any] = {
    "tts": {
        "voice": "en-US-EmmaMultilingualNeural",
        "rate": "+0%",
        "volume": "+0%",
        "pitch": "+0Hz",
        # WordBoundary is requested EXPLICITLY at synthesis time; never rely
        # on the engine default (edge-tts 7.x defaults to SentenceBoundary).
        "boundary": "WordBoundary",
    },
    "estimate": {
        # Filled from the measured Kaggle preflight (Phase 0).  None until then.
        "sec_per_word": None,
        # Diagnostics only.  Never present estimates produced with this as
        # if they were calibrated measurements.
        "fallback_sec_per_word": 0.40,
    },
    "transitions": {
        "default": "fade",
        "duration_s": 1.0,
        "min_duration_s": MIN_TRANSITION_S,
    },
    "video": {
        "width": 1280,
        "height": 720,
        "fps": 30,
    },
    "frames": {
        "dir": "assets/frames",
        "source_dir": "assets/frames/source",
    },
    "paths": {
        "narration": "narration/narration.md",
        "frames_json": "frames.json",
        "style": "assets/STYLE.md",
        "build_dir": "build",
    },
}


class ConfigError(Exception):
    """Invalid or unsafe configuration."""


def _merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(repo_root: Path) -> Dict[str, Any]:
    """Load kaggle/config.yaml merged over defaults; validate hard limits."""
    cfg_path = Path(repo_root) / "kaggle" / "config.yaml"
    data: Dict[str, Any] = {}
    if cfg_path.exists():
        with open(cfg_path, "r", encoding="utf-8") as fh:
            loaded = yaml.safe_load(fh)
        if loaded is None:
            data = {}
        elif not isinstance(loaded, dict):
            raise ConfigError(f"{cfg_path}: top level must be a mapping")
        else:
            data = loaded
    cfg = _merge(DEFAULTS, data)

    # Validate.
    boundary = cfg["tts"].get("boundary")
    if boundary not in ("WordBoundary", "SentenceBoundary"):
        raise ConfigError(
            f"tts.boundary must be 'WordBoundary' or 'SentenceBoundary', got {boundary!r}"
        )
    t_min = float(cfg["transitions"].get("min_duration_s", MIN_TRANSITION_S))
    if t_min < MIN_TRANSITION_S:
        raise ConfigError(
            f"transitions.min_duration_s must be >= {MIN_TRANSITION_S} (spec rule 9), got {t_min}"
        )
    t_dur = float(cfg["transitions"].get("duration_s", 1.0))
    if t_dur < t_min:
        raise ConfigError(
            f"transitions.duration_s ({t_dur}) must be >= transitions.min_duration_s ({t_min})"
        )
    spw = cfg["estimate"].get("sec_per_word")
    if spw is not None:
        try:
            spw = float(spw)
        except (TypeError, ValueError):
            raise ConfigError(f"estimate.sec_per_word must be a number or null, got {spw!r}")
        if spw <= 0:
            raise ConfigError(f"estimate.sec_per_word must be > 0, got {spw}")
        cfg["estimate"]["sec_per_word"] = spw
    for key in ("width", "height", "fps"):
        val = cfg["video"].get(key)
        if not isinstance(val, int) or val <= 0:
            raise ConfigError(f"video.{key} must be a positive integer, got {val!r}")
    return cfg


def sec_per_word_or_fallback(cfg: Dict[str, Any]):
    """Return (value, calibrated: bool, note)."""
    spw = cfg["estimate"].get("sec_per_word")
    if spw is not None:
        return float(spw), True, "measured (Phase 0 calibration)"
    fb = float(cfg["estimate"].get("fallback_sec_per_word", 0.40))
    return (
        fb,
        False,
        f"UNCalibrated fallback {fb} s/word - diagnostics only until Phase 0 runs",
    )


def narration_path(cfg: Dict[str, Any], repo_root: Path) -> Path:
    return Path(repo_root) / cfg["paths"]["narration"]


def frames_json_path(cfg: Dict[str, Any], repo_root: Path) -> Path:
    return Path(repo_root) / cfg["paths"]["frames_json"]


def build_dir(cfg: Dict[str, Any], repo_root: Path) -> Path:
    return Path(repo_root) / cfg["paths"]["build_dir"]
