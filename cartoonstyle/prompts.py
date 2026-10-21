"""Build the job list from CSV / JSONL prompt files or auto-variety templates."""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
import os
import random


def job_key(index, prompt, negative, seed, width, height, steps, guidance, model_tag, lora):
    raw = f"{index}|{prompt}|{negative}|{seed}|{width}x{height}|{steps}|{guidance}|{model_tag}|{lora}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_csv_jobs(path: str) -> list[dict]:
    jobs: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            prompt = (row.get("prompt") or "").strip()
            if not prompt:
                continue
            count = int(row.get("count") or 1)
            seed = row.get("seed")
            seed = int(seed) if seed not in (None, "") else None
            negative = (row.get("negative") or "").strip()
            for _ in range(count):
                jobs.append({"prompt": prompt, "negative": negative, "seed": seed})
    return jobs


def load_jsonl_jobs(path: str) -> list[dict]:
    jobs: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            count = int(obj.get("count", 1))
            for _ in range(count):
                jobs.append(
                    {
                        "prompt": obj["prompt"],
                        "negative": obj.get("negative", ""),
                        "seed": obj.get("seed"),
                    }
                )
    return jobs


def template_jobs(cfg: dict, n: int, seed: int) -> list[dict]:
    """Deterministic random combos from the prompt template pools."""
    prompts_cfg = cfg.get("prompts") or {}
    pools = {k: list(v) for k, v in (prompts_cfg.get("templates") or {}).items() if v}
    if not pools:
        raise SystemExit(
            "No prompts.file set and no prompts.templates pools defined in the config."
        )
    prefix = (prompts_cfg.get("style_prefix") or "").strip()
    fallback_negative = (prompts_cfg.get("negative") or (cfg.get("generation") or {}).get("negative") or "").strip()
    rng = random.Random(seed)
    keys = sorted(pools)
    max_unique = 1
    for k in keys:
        max_unique *= len(pools[k])

    combos = list(itertools.product(*[pools[k] for k in keys]))
    rng.shuffle(combos)
    jobs: list[dict] = []
    for i in range(n):
        combo = combos[i % len(combos)]
        if i and i % len(combos) == 0:
            rng.shuffle(combos)  # reshuffle deck each full cycle
        body = ", ".join(combo)
        prompt = f"{prefix}, {body}" if prefix else body
        jobs.append({"prompt": prompt, "negative": fallback_negative, "seed": None})
    return jobs


def build_jobs(cfg: dict, total: int, base_seed: int) -> list[dict]:
    """Return exactly `total` jobs with index + concrete seed filled in.

    Each job is a fresh dict (duplicated prompt rows must not share state),
    and every job ends up with a unique, reproducible seed + index so resume
    keys never collide.
    """
    prompts_cfg = cfg.get("prompts") or {}
    path = prompts_cfg.get("file")
    if path:
        if not os.path.exists(path):
            raise SystemExit(f"prompts.file not found: {path}")
        if path.endswith(".jsonl"):
            base = load_jsonl_jobs(path)
        else:
            base = load_csv_jobs(path)
        if not base:
            raise SystemExit(f"No prompts found in {path}")
        prefix = (prompts_cfg.get("style_prefix") or "").strip()
        occurrence: dict[int, int] = {}
        jobs: list[dict] = []
        for i in range(total):
            src_idx = i % len(base)
            src = base[src_idx]
            nth = occurrence.get(src_idx, 0)
            occurrence[src_idx] = nth + 1
            prompt = src["prompt"]
            if prefix and not prompt.startswith(prefix):
                prompt = f"{prefix}, {prompt}"
            seed = src.get("seed")
            if seed is not None:
                seed = int(seed) + nth  # explicit CSV seed, unique per duplicate
            jobs.append({"prompt": prompt, "negative": src.get("negative") or "", "seed": seed})
    else:
        jobs = [
            {"prompt": j["prompt"], "negative": j.get("negative") or "", "seed": None}
            for j in template_jobs(cfg, total, base_seed)
        ]

    for idx, j in enumerate(jobs):
        j["index"] = idx
        if j.get("seed") is None:
            j["seed"] = base_seed + idx
    return jobs
