"""Stage 1 of the narration-first workflow: script writer rules + validator.

Pipeline (the user's design):
  1. SCRIPT WRITER  topic -> professional non-repetitive script (beats)
  2. HUMAN APPROVAL edit until it sings
  3. TTS + PROMPTS  narration words -> per-beat visual prompts (every content
                    word of a beat must be visible in its image)
  4. ONE KAGGLE PASS the whole video's images in a single unattended run

This tool structures and validates scripts; it never invents facts.

Commands:
  validate --script draft.txt
      quality report: hook, repeated openers, filler openers, beat timing
  package  --script draft.txt --out-dir examples/my_video/
      writes script.md (numbered beats + timecodes), beats.jsonl,
      and visuals.csv skeleton with a per-beat words-to-cover checklist
  cover    --script draft.txt --visuals examples/my_video/visuals.csv
      coverage gate: every content word of each beat must appear in its
      visual_prompt (this is what "images react to the narrator" means)
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

WPM = 150.0  # narration speaking rate: words per minute
MIN_BEAT_WORDS = 8
MAX_BEAT_WORDS = 45

STOPWORDS = set(
    """a an the and or but if then than that this those these of in on at to for
    with without from by as is are was were be been being it its it's his her their
    our your my me him them us we you i he she they so not no yes very just also
    too can will would could should may might must do does did done have has had
    about into over under again more most some any all each other another when
    while where which who whom what how why there here out up down off only own
    same such get got make made take took go going went come came back even ever
    yet still one two three first next last before after during through between
    because until since upon both few many much little big small old new""".split()
)

FILLER_OPENERS = {"so", "okay", "ok", "hey", "well", "um", "uh", "basically", "actually", "anyway", "right"}


def split_beats(script_text: str) -> list[str]:
    """Beats are blocks separated by blank lines - one beat = one spoken moment."""
    blocks = re.split(r"\n\s*\n", script_text.strip())
    return [" ".join(b.split()) for b in blocks if b.strip()]


def content_words(text: str) -> list[str]:
    words = re.findall(r"[a-zA-Z][a-zA-Z'-]+", text.lower())
    out, seen = [], set()
    for w in words:
        w = w.strip("'-")
        if len(w) < 3 or w in STOPWORDS or w in seen:
            continue
        seen.add(w)
        out.append(w)
    return out


def opener(beat: str) -> str:
    return " ".join(beat.split()[:3]).lower().strip(".,!?:;")


def validate(beats: list[str]) -> tuple[list[str], list[dict]]:
    issues: list[str] = []
    table: list[dict] = []
    openers: dict[str, int] = {}
    t = 0.0
    for i, b in enumerate(beats, 1):
        n = len(b.split())
        secs = n / WPM * 60.0
        words = content_words(b)
        table.append({"beat": i, "words": n, "sec": round(secs, 1),
                      "starts": round(t, 1), "key_words": words[:8]})
        t += secs
        is_sfx = b.lower().startswith("sfx")
        if n < MIN_BEAT_WORDS and not is_sfx:
            issues.append(f"beat {i}: only {n} words - too thin to visualize ({MIN_BEAT_WORDS}+ needed)")
        if n > MAX_BEAT_WORDS:
            issues.append(f"beat {i}: {n} words - split it (max {MAX_BEAT_WORDS}, ~{MAX_BEAT_WORDS / WPM * 60:.0f}s)")
        op = opener(b)
        openers[op] = openers.get(op, 0) + 1
        if op.split()[0] in FILLER_OPENERS:
            issues.append(f'beat {i}: starts with filler "{op.split()[0]}" - cut it')
    for op, c in openers.items():
        if c > 1:
            issues.append(f'beats start alike ("{op}" x{c}) - make openings non-repetitive')
    first = beats[0].lower() if beats else ""
    if not re.search(r"\?|imagine|what if|never|always|secret|question|why |how |here's", first):
        issues.append("beat 1: no hook - open with a question or a bold claim")
    return issues, table


def fmt_time(sec: float) -> str:
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def cmd_validate(args: argparse.Namespace) -> None:
    beats = split_beats(Path(args.script).read_text(encoding="utf-8"))
    issues, table = validate(beats)
    total = table[-1]["starts"] + table[-1]["sec"] if table else 0
    print(f"{len(beats)} beats | ~{fmt_time(total)} of narration at {WPM:.0f} wpm\n")
    for r in table:
        print(f"  beat {r['beat']:>2}  {fmt_time(r['starts'])}  {r['words']:>2}w  "
              f"{', '.join(r['key_words'][:5])}")
    print()
    if issues:
        print("FIX BEFORE APPROVAL:")
        for m in issues:
            print("  -", m)
    else:
        print("PASS: hook present, no repeated/filler openers, beat sizes healthy.")


def cmd_package(args: argparse.Namespace) -> None:
    beats = split_beats(Path(args.script).read_text(encoding="utf-8"))
    issues, table = validate(beats)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    lines = [f"# Script package\n"]
    t = 0.0
    for i, b in enumerate(beats, 1):
        lines.append(f"## Beat {i}  ({fmt_time(t)} -> {fmt_time(t + table[i-1]['sec'])})\n\n{b}\n")
        t += table[i - 1]["sec"]
    (out / "script.md").write_text("\n".join(lines), encoding="utf-8")

    with open(out / "beats.jsonl", "w", encoding="utf-8") as f:
        t = 0.0
        for i, b in enumerate(beats, 1):
            f.write(json.dumps({"beat": i, "start": round(t, 1),
                                "sec": table[i - 1]["sec"], "text": b,
                                "words_to_cover": content_words(b)}) + "\n")
            t += table[i - 1]["sec"]

    with open(out / "visuals.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["slide_id", "moment", "moment_text", "visual_prompt"])
        for i, b in enumerate(beats, 1):
            w.writerow([i, 1, b, ""])  # split into more moments as the words demand

    print(f"wrote {out}/script.md, beats.jsonl, visuals.csv")
    print("NEXT: fill visual_prompt per beat so every word in words_to_cover is visible,")
    print("then run:  python -m cartoonstyle.scriptwriter cover --script ... --visuals ...")
    if issues:
        print("\nwarnings (fix in the script if you can):")
        for m in issues:
            print("  -", m)


def cmd_cover(args: argparse.Namespace) -> None:
    """Moment-level coverage gate.

    Two checks per beat:
      1. every moment_text is findable in the beat (so word timings can lock it)
      2. every content word of the beat is claimed by some moment_text
         (the images react to EVERY word the narrator says)
    """
    beats = split_beats(Path(args.script).read_text(encoding="utf-8"))
    rows = list(csv.DictReader(open(args.visuals, encoding="utf-8")))
    groups: dict[int, list[dict]] = {}
    for r in rows:
        groups.setdefault(int(r["slide_id"]), []).append(r)

    def norm(t: str) -> list[str]:
        return re.findall(r"[a-z0-9]+", t.lower())

    def found_in(needle: list[str], hay: list[str]) -> bool:
        n = len(needle)
        return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))

    fails = 0
    for i, b in enumerate(beats, 1):
        ms = groups.get(i, [])
        hay = norm(b)
        unfindable = [m["moment_text"][:32] for m in ms if not found_in(norm(m["moment_text"]), hay)]
        claimed = " ".join(m["moment_text"] for m in ms).lower()
        missing = [w for w in content_words(b) if w not in claimed and w[:-1] not in claimed]
        status = "OK " if not unfindable and not missing else "MISS"
        print(f"beat {i:>2} [{status}] {len(ms)} moment(s) | {len(content_words(b))} words")
        for t in unfindable:
            print(f"          moment text not found in beat: \"{t}...\"")
            fails += 1
        if missing:
            print(f"          words no moment claims: {', '.join(missing)}")
            fails += 1
    print()
    print("COVERAGE PASS: every narration word has its image moment." if not fails
          else f"COVERAGE FAIL: {fails} problem(s) - the narrator would say things the viewer never sees.")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("validate"); v.add_argument("--script", required=True); v.set_defaults(fn=cmd_validate)
    pk = sub.add_parser("package"); pk.add_argument("--script", required=True); pk.add_argument("--out-dir", required=True); pk.set_defaults(fn=cmd_package)
    cv = sub.add_parser("cover"); cv.add_argument("--script", required=True); cv.add_argument("--visuals", required=True); cv.set_defaults(fn=cmd_cover)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
