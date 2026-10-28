"""Artifact manifest: ties Phase 4 measurements to their exact inputs.

The hash covers narration text, TTS settings, requested boundary mode and
the engine version.  Any change to those makes preserved artifacts stale
and forces re-measurement (spec: recheck calibration applicability when the
engine, version, voice, rate or boundary mode changes).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict


def manifest_payload(passage: str, cfg: Dict[str, Any], engine_version: str) -> str:
    canon = json.dumps(
        {
            "narration": passage,
            "voice": cfg["tts"]["voice"],
            "rate": cfg["tts"]["rate"],
            "volume": cfg["tts"]["volume"],
            "pitch": cfg["tts"]["pitch"],
            "boundary": cfg["tts"]["boundary"],
            "edge_tts_version": engine_version,
        },
        sort_keys=True,
    )
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()
