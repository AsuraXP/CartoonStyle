"""YAML config loading with dotted-key overrides."""

from __future__ import annotations

import os

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CONFIG_PATH = os.path.join(REPO_ROOT, "config", "default.yaml")


def load_config(path: str | None = None) -> dict:
    path = path or DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def apply_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply {"generation.width": 1344, ...} style overrides in place."""
    for dotted, value in overrides.items():
        if value is None:
            continue
        node = cfg
        parts = dotted.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return cfg


def save_config(cfg: dict, path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
