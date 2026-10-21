"""Vision QC - a small VLM judges generated images so you don't have to.

Two modes:
  1. INLINE (preferred): cartoonstyle.generate_flux2 --qc-retries N judges every
     image against its own prompt the moment it is drawn, and redraws failures
     with a fresh seed until they pass (self-healing pipeline).
  2. STANDALONE: this CLI audits an existing images/ folder and prints a
     surgery report (keys + problem phrases).

The judge flags the recurring failures of cheap distilled image models:
extra/missing limbs and hands, stick-figure animals / toy-scale objects,
empty or unreadable scenes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3-VL-4B-Instruct"

NAME_RE = re.compile(r"(beat\d+m\d+)_v(\d+)_s[0-9a-fA-F]+\.png$")


def parse_image_name(path: str) -> tuple[str, int] | None:
    m = NAME_RE.search(os.path.basename(path))
    return (m.group(1), int(m.group(2))) if m else None


CHECKS = (
    ("limbs", "Do all human figures in this image have exactly two arms, two legs "
              "and one head? Two people standing close together can look like one - "
              "count each person separately. If a person is only partly visible (for "
              "example just hands), judge only the visible parts. If no humans are "
              "visible, answer yes. Answer no only if you can SEE a figure with wrong "
              "limbs. Answer yes or no."),
    ("match", 'The words say what this picture should show: "{scene}" '
              'Does the picture show the main subject of those words? '
              'The main person, animal or big object just needs to be in the picture. '
              'Exact numbers of people, small objects, readable text, emotions or '
              'gestures do not matter. Answer yes or no.'),
)

PROBLEM_TEXT = {
    "limbs": "extra or missing limbs",
    "match": "does not match the words",
}


def classify(answers: list[str]) -> dict:
    """First word must be 'yes'. Anything else - empty, unsure, rambling - fails."""
    defects = {n: True for n, _ in CHECKS}   # missing answer = fail
    for (name, _q), ans in zip(CHECKS, answers):
        toks = [re.sub(r"[^a-z]", "", t) for t in (ans or "").lower().split()[:6]]
        verdict = "yes" if "yes" in toks else ("no" if "no" in toks else None)
        defects[name] = verdict != "yes"
    score = len(CHECKS) - sum(defects.values())
    probs = [PROBLEM_TEXT[n] for n, _ in CHECKS if defects[n]]
    return {
        "limbs_bad": defects["limbs"],
        "match_bad": defects["match"],
        "score": score,
        "verdict": "GOOD" if score == len(CHECKS) else "BAD",
        "problem": "; ".join(probs) if probs else "none",
    }


def load_judge(model_name: str = DEFAULT_MODEL, device: str = ""):
    """(model, processor) for judge_image. Heavy - call once. Pin to `device`
    (ideally NOT the drawing GPU) so judge and art model never fight for VRAM."""
    import qwen_vl_utils  # noqa: F401  - readiness probe: fail HERE, not mid-run
    import torch
    import transformers as T
    from transformers import AutoProcessor

    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[load] vision judge {model_name} - one time only", flush=True)
    processor = AutoProcessor.from_pretrained(model_name)
    model = None
    errors = []
    for cls_name in ("Qwen3VLForConditionalGeneration", "Qwen3VLMoeForConditionalGeneration",
                     "Qwen2_5_VLForConditionalGeneration", "Qwen2VLForConditionalGeneration",
                     "AutoModelForImageTextToText", "AutoModelForVision2Seq"):
        cls = getattr(T, cls_name, None)
        if cls is None:
            continue
        for kw in ({"load_in_8bit": True, "device_map": {"": device}},
                   {"torch_dtype": torch.float16, "device_map": {"": device}},
                   {"torch_dtype": torch.float32, "device_map": {"": "cpu"}}):
            try:
                model = cls.from_pretrained(model_name, **kw)
                break
            except Exception as e:
                errors.append(f"{cls_name} {kw}: {e}")
        if model is not None:
            break
    if model is None:
        raise RuntimeError("no loader class worked: " + " | ".join(errors))
    model.eval()
    print("[load] vision judge ready", flush=True)
    return model, processor


def judge_image(model, processor, image, scene: str = "", narration: str = "") -> dict:
    """image = path or PIL.Image -> verdict dict. Three concrete yes/no
    questions instead of one multi-part form (2B models stay honest that way)."""
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")

    scene = " ".join((scene or "").split())[:400]
    narration = " ".join((narration or "").split())[:200]
    answers = []
    for name, q in CHECKS:
        text = q.format(scene=scene, narration=narration)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": text},
        ]}]
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        imgs, vids = process_vision_info(messages)
        inputs = processor(text=[chat], images=imgs, videos=vids, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=16)
        answers.append(processor.batch_decode(
            gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True
        )[0])
    rec = classify(answers)
    rec["raw"] = " | ".join(a.strip() for a in answers)
    return rec


def judge_limbs(model, processor, image) -> dict:
    """The VLM's one reliable job: counting limbs (yes/no)."""
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    if isinstance(image, (str, Path)):
        image = Image.open(image).convert("RGB")
    q = CHECKS[0][1]
    messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                             {"type": "text", "text": q}]}]
    chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    imgs, vids = process_vision_info(messages)
    inputs = processor(text=[chat], images=imgs, videos=vids, return_tensors="pt").to(model.device)
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=16)
    ans = processor.batch_decode(gen[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)[0]
    toks = [re.sub(r"[^a-z]", "", t) for t in ans.lower().split()[:6]]
    verdict = "yes" if "yes" in toks else ("no" if "no" in toks else None)
    return {"limbs_bad": verdict != "yes", "raw": ans.strip()}


def main() -> None:
    p = argparse.ArgumentParser(description="VLM image QC (standalone audit)")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out", required=True, help="directory for qc.jsonl")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--prompts", default="", help="visuals.csv - binds checks to each prompt")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--verbose", action="store_true", help="print the judge's full reasoning")
    args = p.parse_args()

    scenes: dict[str, tuple[str, str]] = {}
    if args.prompts:
        import csv as _csv
        for r in _csv.DictReader(open(args.prompts, encoding="utf-8")):
            sid = str(r.get("slide_id", "")).zfill(2)
            mom = str(r.get("moment") or "").zfill(2)
            scenes[f"beat{sid}m{mom}"] = ((r.get("visual_prompt") or r.get("prompt") or "").strip(),
                                          (r.get("moment_text") or "").strip())

    files = sorted(glob.glob(os.path.join(args.images_dir, "*.png")))
    if args.limit:
        files = files[: args.limit]
    if not files:
        raise SystemExit(f"no PNGs in {args.images_dir}")
    print(f"[qc] {len(files)} images | model {args.model}", flush=True)

    model, processor = load_judge(args.model)
    from .scorer import load_scorer, score_pairs

    print(f"[qc] {len(files)} images | limbs: {args.model} | match: siglip scores", flush=True)
    scorer, sproc = load_scorer()
    texts = []
    for f in files:
        parsed = parse_image_name(f)
        key = parsed[0] if parsed else os.path.basename(f)
        sc, _n = scenes.get(key, ("the scene described in the narration", ""))
        texts.append(" ".join(sc.split())[:400])
    scores, best_txt, own_rank = score_pairs(scorer, sproc, files, texts)
    aligned = sum(1 for i, b in enumerate(best_txt) if i == b)
    print(f"[qc] alignment self-check: {aligned}/{len(files)} pictures match their OWN words best",
          flush=True)

    import statistics as _st
    med = _st.median(scores)
    mad = _st.median([abs(s - med) for s in scores])
    span = max(3.0 * mad, 0.04)
    match_bad = [bool(s < med - span or s < 0.08) for s in scores]
    print(f"[qc] match scores: median {med:.3f}  min {min(scores):.3f}  max {max(scores):.3f}", flush=True)
    print("[qc] lowest 8 (score  own-word-rank  file):", flush=True)
    for s, f, r in sorted(zip(scores, files, own_rank), key=lambda x: x[0])[:8]:
        print(f"        {s:.3f}   rank #{r}/{len(files)}   {os.path.basename(f)}", flush=True)

    out_path = Path(args.out) / "qc.jsonl"
    bad = []
    with open(out_path, "w", encoding="utf-8") as fh:
        for i, f in enumerate(files):
            parsed = parse_image_name(f)
            key, variant = parsed if parsed else (os.path.basename(f), 0)
            limb = judge_limbs(model, processor, f)
            probs = []
            if limb["limbs_bad"]:
                probs.append("extra or missing limbs")
            if match_bad[i]:
                probs.append("does not match the words")
            rec = {"key": key, "variant": variant, "file": os.path.basename(f),
                   "match_score": round(scores[i], 3),
                   "limbs_bad": limb["limbs_bad"], "match_bad": bool(match_bad[i]),
                   "score": 2 - len(probs), "verdict": "GOOD" if not probs else "BAD",
                   "problem": "; ".join(probs) if probs else "none",
                   "raw": limb.get("raw", "")}
            fh.write(json.dumps(rec) + "\n")
            mark = rec["verdict"]
            extra = f"  {rec['problem']}" if mark == "BAD" else ""
            print(f"[qc] {i + 1}/{len(files)} {rec['file']} {mark} (match {scores[i]:.2f}){extra}",
                  flush=True)
            if args.verbose:
                print(f"        | limbs: {limb.get('raw', '')}", flush=True)
            if mark == "BAD":
                bad.append(rec)

    print(f"\n[qc] report -> {out_path}", flush=True)
    if bad:
        keys = sorted({r["key"] for r in bad})
        print(f"[qc] {len(keys)} moment(s) need surgery:", flush=True)
        for k in keys:
            probs = [r["problem"] for r in bad if r["key"] == k and r["problem"] and r["problem"] != "none"]
            print(f"    {k}  {'; '.join(probs) if probs else 'flagged'}", flush=True)
    else:
        print("[qc] ALL CLEAN - nothing to fix", flush=True)


if __name__ == "__main__":
    main()
