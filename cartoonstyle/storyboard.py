"""Turn a narration script into a slideshow image job list.

Contract: **one script paragraph = one slide = one visual prompt.**
Blank-line paragraphs become slides (long ones split at sentence boundaries).
A short punctuation-less first line is treated as the title. Stage directions
like [SFX ...] and NARRATOR: labels are stripped.

Workflow (the quality path):
  1. write your narration script with blank lines between slides,
  2. write ONE visual scene prompt per slide by hand or with an LLM
     (or let --mode auto draft them),
  3. this tool emits prompts.csv (consumed by cartoonstyle.generate) plus
     slides.jsonl mapping narration -> output image indices for video editing.

    python -m cartoonstyle.storyboard --script my_script.txt \
        --out-dir storyboard_out --images-total 600 --trigger ckstyle \
        --visuals my_visuals.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re

STOPWORDS = set(
    "a an the and or but if then than so of to in on at for from by with about into over "
    "after before between out up down off again further once here there all any both each "
    "few more most other some such no nor not only own same too very can will just should "
    "now i you he she it we they me him her us them my your his its our their this that "
    "these those is are was were be been being have has had do does did doing would could "
    "im ive youve dont doesnt didnt isnt arent wasnt werent hasnt havent hadnt wont wouldnt "
    "shouldnt couldnt".split()
)

SENT_RE = re.compile(r"[^.!?\n]+[.!?]?")


def clean_script(text: str) -> list[str]:
    """Strip stage directions / speaker labels, return narration sentences."""
    text = re.sub(r"\[[^\]]*\]", " ", text)  # [SFX ...], [music], ...
    text = re.sub(r"(?m)^\s*[A-Z][A-Z ]{2,}:\s*", "", text)  # NARRATOR:
    text = re.sub(r"\s+", " ", text).strip()
    return [s.strip() for s in SENT_RE.findall(text) if s.strip()]


def pack_sentences(sentences: list[str], per_slide_words: int) -> list[str]:
    slides: list[str] = []
    current: list[str] = []
    words = 0
    for s in sentences:
        sw = len(s.split())
        if current and words + sw > per_slide_words * 1.3:
            slides.append(" ".join(current))
            current, words = [], 0
        current.append(s)
        words += sw
    if current:
        slides.append(" ".join(current))
    return slides


def split_script(text: str, target_slides: int | None = None, max_words: int = 55):
    """Return (title, slides). Paragraph-based; sentence-packs long blocks."""
    blocks = re.split(r"\n\s*\n+", text)
    slides: list[str] = []
    title = None
    for b in blocks:
        sentences = clean_script(b)
        if not sentences:
            continue  # pure stage direction block
        joined = " ".join(sentences)
        words = joined.split()
        if title is None and len(words) <= 8 and not re.search(r"[.!?]$", joined):
            title = joined  # e.g. "The 2:07 AM Glitch"
            continue
        if len(words) <= max_words:
            slides.append(joined)
        else:
            slides.extend(pack_sentences(sentences, max_words))

    if not slides:
        raise SystemExit("Script is empty after cleaning.")
    # Wall-of-text fallback: no usable paragraph structure
    if len(slides) == 1 and target_slides and target_slides > 1:
        all_sentences = clean_script(text)
        per = max(8, round(sum(len(s.split()) for s in all_sentences) / target_slides))
        slides = pack_sentences(all_sentences, per)
    return title, slides


def auto_visual_prompt(narration: str) -> str:
    """Crude offline visualizer: salient keywords as a scene description.

    Fine for pipeline tests. For production, write the visuals yourself or with
    an LLM and pass them via --visuals — the visual prompt IS the image quality.
    """
    words = re.findall(r"[a-z][a-z'-]+", narration.lower())
    keywords = [w for w in words if w not in STOPWORDS][:14]
    return "a clear simple scene of " + ", ".join(keywords)


def load_visuals(path: str) -> dict[int, str]:
    visuals: dict[int, str] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sid = int(row["slide_id"])
            visuals[sid] = (row.get("visual_prompt") or row.get("prompt") or "").strip()
    return visuals


def main():
    p = argparse.ArgumentParser(description="Narration script -> slideshow image jobs")
    p.add_argument("--script", required=True, help="narration .txt (blank line between slides)")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--images-total", type=int, default=600, help="total images to generate")
    p.add_argument("--images-per-slide", type=int, default=None,
                   help="overrides --images-total (e.g. 5 candidates per slide)")
    p.add_argument("--slides", type=int, default=None,
                   help="target slide count (only used for unstructured wall-of-text scripts)")
    p.add_argument("--trigger", default="ckstyle", help="LoRA trigger token")
    p.add_argument("--style-prefix", default="flat 2d cartoon style, flat solid colors, vibrant bold saturated colors, thick black outlines, stickman character with round white head, small black oval eyes",
                   help="optional style words (usually leave empty: the generator's "
                        "config supplies style_prefix automatically)")
    p.add_argument("--visuals", default=None,
                   help="CSV with slide_id,visual_prompt (hand-written / LLM) — recommended")
    args = p.parse_args()

    with open(args.script, encoding="utf-8") as f:
        raw = f.read()
    target_slides = args.slides or max(1, round(args.images_total / (args.images_per_slide or 5)))
    title, narrations = split_script(raw, target_slides)
    n = len(narrations)

    visuals = load_visuals(args.visuals) if args.visuals else {}
    if args.visuals and len(visuals) != n:
        preview = " | ".join(f"{i+1}:{s[:40]}..." for i, s in enumerate(narrations[:6]))
        raise SystemExit(
            f"visuals/slide mismatch: {len(visuals)} rows in {args.visuals} vs {n} slides.\n"
            f"Slides are the script's blank-line paragraphs. First slides: {preview}\n"
            f"Fix the slide_id list (1..{n}) or add blank lines to the script."
        )

    if args.images_per_slide:
        per = [args.images_per_slide] * n
    else:
        base = args.images_total // n
        per = [base + (1 if i < args.images_total % n else 0) for i in range(n)]

    os.makedirs(args.out_dir, exist_ok=True)
    prompts_path = os.path.join(args.out_dir, "prompts.csv")
    slides_path = os.path.join(args.out_dir, "slides.jsonl")

    idx = 0
    with open(prompts_path, "w", newline="", encoding="utf-8") as f_prompts, \
            open(slides_path, "w", encoding="utf-8") as f_slides:
        w = csv.writer(f_prompts)
        w.writerow(["prompt", "count"])
        for i, narration in enumerate(narrations, 1):
            visual = visuals.get(i) or auto_visual_prompt(narration)
            prompt_parts = [args.trigger.strip(), args.style_prefix.strip(), visual]
            prompt = ", ".join(x for x in prompt_parts if x)
            count = per[i - 1]
            w.writerow([prompt, count])
            rec = {
                "slide_id": i,
                "narration": narration,
                "visual_prompt": visual,
                "prompt": prompt,
                "count": count,
                "index_start": idx,
                "index_end": idx + count - 1,
            }
            f_slides.write(json.dumps(rec, ensure_ascii=False) + "\n")
            idx += count

    print(f"[storyboard] title: {title or '(none)'}")
    print(f"[storyboard] {n} slides ({'paragraph split' if not args.visuals else 'matched to visuals'})")
    print(f"[storyboard] {idx} images total ({min(per)}-{max(per)} per slide)")
    print(f"[storyboard] prompts -> {prompts_path}")
    print(f"[storyboard] edit map -> {slides_path}  (narration -> image index ranges)")
    print(f"\nNext: python -m cartoonstyle.generate --prompts {prompts_path} "
          f"--images {idx} --gpus 2 --out /kaggle/working/cartoonstyle-out")


if __name__ == "__main__":
    main()
