"""Final stage: ffmpeg auto-stitch - one image per beat, exact TTS timing.

Reads beats_with_timing.json (from cartoonstyle.tts) and a directory of
generated images named beatNN_vMM_sSEED.png (from cartoonstyle.generate_flux2),
picks one variant per beat, and assembles narration-driven VIDEO.mp4.

Usage:
  python -m cartoonstyle.stitch --timing examples/my_video/beats_with_timing.json \
      --images-dir /kaggle/working/cartoonstyle-out/images \
      --out /kaggle/working/VIDEO.mp4 [--pick 0] [--fallback-dir ...]

--pick N  which variant to show per beat (v00, v01, ...). Choose after review,
          or generate several videos with different picks.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


def find_image(images_dir: str, key: str, pick: int, fallback_dir: str | None) -> str | None:
    pat = f"{key}_v{pick:02d}_s*.png"
    hits = sorted(glob.glob(str(Path(images_dir) / pat)))
    if not hits and fallback_dir:
        hits = sorted(glob.glob(str(Path(fallback_dir) / pat)))
    return str(Path(hits[0]).resolve()) if hits else None


def beat_segment(cmd_base: list[str], work: Path, beat: int, entries: list[tuple[str, float]],
                 audio: Path) -> Path:
    """One beat = one audio file + one or more images (word-locked)."""
    lst = work / f"imgs{beat:02d}.txt"
    lines = []
    for img, sec in entries:
        lines.append(f"file '{img}'\nduration {sec:.3f}\n")
    lines.append(f"file '{entries[-1][0]}'\n")  # concat demuxer wants the last file repeated
    lst.write_text("".join(lines), encoding="utf-8")
    seg = work / f"seg{beat:02d}.mp4"
    subprocess.run(cmd_base + ["-f", "concat", "-safe", "0", "-i", str(lst),
                               "-i", str(audio),
                               "-c:v", "libx264", "-tune", "stillimage", "-r", "30",
                               "-c:a", "aac", "-b:a", "192k",
                               "-pix_fmt", "yuv420p", "-shortest", str(seg)],
                   check=True, capture_output=True)
    return seg



def ass_time(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 3600)}:{int(t % 3600 // 60):02d}:{t % 60:05.2f}"


def caption_events(m: dict, max_words: int = 5) -> list[tuple[float, float, str]]:
    """One or more caption lines for a moment; long chunks roll in time."""
    text = (m.get("text") or "").strip()
    if not text or text.lower().startswith("sfx"):
        return []
    words = text.replace("{", "").replace("}", "").replace("\\", "").split()
    if not words:
        return []
    if len(words) <= max_words:
        return [(m["start"], m["start"] + m["sec"], " ".join(words))]
    chunks = [words[i:i + max_words] for i in range(0, len(words), max_words)]
    total = float(len(words))
    evts = []
    t = m["start"]
    for c in chunks:
        dur = m["sec"] * len(c) / total
        evts.append((t, t + dur, " ".join(c)))
        t += dur
    return evts


def write_ass(moments: list[dict], out: Path, font_name: str, size: int = 58) -> Path:
    """Word-locked captions: white sans, dark outline, bottom-center, 1920x1080."""
    head = (
        "[Script Info]\nScriptType: v4.00+\nPlayResX: 1920\nPlayResY: 1080\nWrapStyle: 0\n\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
        "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
        "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
        f"Style: Cap,{font_name},{size},&H00FFFFFF,&H000000FF,&H00181818,&H96000000,"
        "1,0,0,0,100,100,0,0,1,3,2,2,140,140,88,1\n\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )
    lines = [head]
    n = 0
    for m in sorted(moments, key=lambda x: x["start"]):
        for a, b, text in caption_events(m):
            lines.append(f"Dialogue: 0,{ass_time(a)},{ass_time(b)},Cap,,0,0,0,,{text}\n")
            n += 1
    out.write_text("".join(lines), encoding="utf-8")
    print(f"[stitch] captions: {n} lines -> {out.name}", flush=True)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="stitch images + narration into VIDEO.mp4")
    p.add_argument("--timing", required=True, help="beats_with_timing.json")
    p.add_argument("--images-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pick", type=int, default=0, help="variant number per beat")
    p.add_argument("--fallback-dir", default="")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--captions", action="store_true", help="burn word-locked subtitles")
    p.add_argument("--font", default="", help="path to Montserrat.ttf (any sans .ttf)")
    p.add_argument("--no-loudnorm", action="store_true", help="skip -14 LUFS normalization")
    p.add_argument("--qc", default="", help="qc.jsonl from cartoonstyle.qc - auto-picks the best candidate per moment")
    args = p.parse_args()

    if not shutil.which("ffmpeg"):
        raise SystemExit("ffmpeg not found - install ffmpeg (Kaggle/Colab images include it)")

    beats = json.loads(Path(args.timing).read_text(encoding="utf-8"))
    mom_path = Path(args.timing).parent / "moments_timing.json"
    moments = json.loads(mom_path.read_text(encoding="utf-8")) if mom_path.exists() else []
    by_beat: dict[int, list[dict]] = {}
    for m in moments:
        by_beat.setdefault(int(m["beat"]), []).append(m)

    work = Path(tempfile.mkdtemp(prefix="stitch_"))
    cmd_base = ["ffmpeg", "-y", "-loglevel", "error"]

    narr = Path(args.timing).parent / "narration.mp3"
    if narr.exists() and moments:
        # ONE continuous take: every moment image in one gapless stream + narration
        qc_best: dict[str, tuple[int, int]] = {}
        if args.qc and Path(args.qc).exists():
            for line in Path(args.qc).read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                r = json.loads(line)
                key, var = r["key"], int(r.get("variant", 0))
                score = int(r.get("score", 0))
                cur = qc_best.get(key)
                if cur is None or (score, -var) > (cur[0], -cur[1]):
                    qc_best[key] = (score, var)
            swapped = sum(1 for k, v in qc_best.items() if v[1] != args.pick)
            print(f"[stitch] qc auto-pick: {len(qc_best)} moments scored, {swapped} use a non-default candidate", flush=True)

        entries = []
        for m in sorted(moments, key=lambda x: x["start"]):
            pick = qc_best[m["key"]][1] if m["key"] in qc_best else args.pick
            img = find_image(args.images_dir, m["key"], pick, args.fallback_dir or None)
            if img:
                entries.append((img, float(m["sec"])))
            else:
                print(f"[stitch] WARNING: no image for {m['key']} - skipped", flush=True)
        if not entries:
            raise SystemExit("no images found")
        lst = work / "all.txt"
        lines = []
        for img, sec in entries:
            lines.append(f"file '{img}'\nduration {sec:.3f}\n")
        lines.append(f"file '{entries[-1][0]}'\n")
        lst.write_text("".join(lines), encoding="utf-8")
        vopts = []
        if args.captions:
            font_path = Path(args.font) if args.font else Path("Montserrat.ttf")
            family = (font_path.stem.split("[")[0].split("-")[0].strip() or "Montserrat")
            if font_path.exists():
                fontsdir = str(font_path.parent.resolve())
                vopts = ["-vf", f"ass={work / 'captions.ass'}:fontsdir={fontsdir}"]
            else:
                print("[stitch] WARNING: font file missing - captions use system default sans", flush=True)
                vopts = ["-vf", f"ass={work / 'captions.ass'}"]
            write_ass(moments, work / "captions.ass", family)
        aopts = [] if args.no_loudnorm else ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"]
        subprocess.run(cmd_base + ["-f", "concat", "-safe", "0", "-i", str(lst),
                                   "-i", str(narr),
                                   *vopts,
                                   "-c:v", "libx264", "-tune", "stillimage", "-r", "30",
                                   *aopts,
                                   "-c:a", "aac", "-b:a", "192k",
                                   "-pix_fmt", "yuv420p", "-shortest", args.out],
                       check=True, capture_output=True)
        print(f"[stitch] VIDEO -> {args.out}  ({len(entries)} word-locked images, one narration take)", flush=True)
        return

    segments = []
    missing = []

    for b in beats:
        n = int(b["beat"])
        audio = Path(args.timing).parent / "audio" / f"beat{n:02d}.mp3"
        entries: list[tuple[str, float]] = []
        for m in sorted(by_beat.get(n, []), key=lambda x: x["start"]):
            img = find_image(args.images_dir, m["key"], args.pick, args.fallback_dir or None)
            if img:
                entries.append((img, float(m["sec"])))
        if not entries:
            img = find_image(args.images_dir, f"beat{n:02d}", args.pick, args.fallback_dir or None)
            if img:
                entries.append((img, float(b["sec"])))
        if not entries or not audio.exists():
            missing.append(n)
            continue
        seg = beat_segment(cmd_base, work, n, entries, audio)
        segments.append(seg)
        print(f"[stitch] beat {n:>2}  {b['sec']:>5.1f}s  {len(entries)} image(s)", flush=True)

    if missing:
        print(f"[stitch] WARNING: no image/audio for beats {missing} - they are skipped", flush=True)
    if not segments:
        raise SystemExit("nothing to stitch")

    lst = work / "list.txt"
    lst.write_text("".join(f"file '{s}'\n" for s in segments), encoding="utf-8")
    subprocess.run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", args.out], check=True, capture_output=True)
    total = sum(Path(s).stat().st_size for s in segments) / 1e6
    print(f"[stitch] VIDEO -> {args.out}  ({len(segments)} beats, ~{total:.0f}MB source)", flush=True)


if __name__ == "__main__":
    main()
