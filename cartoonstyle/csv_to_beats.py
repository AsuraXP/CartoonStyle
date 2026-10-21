"""Convert visuals.csv (slide_id,moment,moment_text,visual_prompt) into beats.jsonl.

Each row becomes one beat with sec=3.0s default. The TTS pipeline then replaces
those placeholder durations with real timing from edge-tts word timings.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--visuals", required=True)
    p.add_argument("--out", required=True, help="path to write beats.jsonl")
    p.add_argument("--default-sec", type=float, default=3.0)
    args = p.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.visuals, newline="", encoding="utf-8")))
    # Group consecutive rows into beats by slide_id (one beat per slide = one
    # narration block). This matches how the ancient-comm video was built.
    slides: dict[str, list[dict]] = {}
    order: list[str] = []
    for r in rows:
        sid = str(r["slide_id"])
        if sid not in slides:
            slides[sid] = []
            order.append(sid)
        slides[sid].append(r)

    t = 0.0
    with out_path.open("w", encoding="utf-8") as f:
        for i, sid in enumerate(order, start=1):
            texts = [r["moment_text"].strip() for r in slides[sid] if r.get("moment_text")]
            text = " ".join(texts).strip()
            # ~3 words/sec speaking pace, cap to keep things tight
            wc = len(text.split())
            sec = max(3.0, min(wc / 2.8 + 0.8, 20.0))
            words_to_cover = [w.strip(".,!?;:()\"'").lower()
                              for w in text.split()
                              if len(w.strip(".,!?;:()\"'")) > 2]
            rec = {
                "beat": i,
                "start": round(t, 2),
                "sec": round(sec, 2),
                "text": text,
                "words_to_cover": words_to_cover[:20],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            t += sec
    print(f"wrote {len(order)} beats to {out_path} (~{t:.0f}s estimated)")


if __name__ == "__main__":
    main()
