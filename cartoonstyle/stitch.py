"""Final stage: ffmpeg auto-stitch - one image per beat, exact TTS timing.

Reads beats_with_timing.json (from cartoonstyle.tts) and a directory of
generated images named beatNN_vMM_sSEED.png (from cartoonstyle.generate_flux2),
picks one variant per beat, and assembles narration-driven VIDEO.mp4.

Images are kept at NATIVE resolution (no upscaling/downscaling). Captions are
burned with PIL directly onto frames — avoids libass segfaults and fontconfig
mismatches on Kaggle/Colab ffmpeg builds.

Usage:
  python -m cartoonstyle.stitch --timing examples/my_video/beats_with_timing.json \
      --images-dir /kaggle/working/cartoonstyle-out/images \
      --out /kaggle/working/VIDEO.mp4 --captions --font Inter-Bold.ttf
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont


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
    lines.append(f"file '{entries[-1][0]}'\n")
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


def pil_burn_captions(entries: list[tuple[str, float]], moments: list[dict],
                      font_path: Path, out_dir: Path,
                      W: int, H: int,
                      max_words: int = 6) -> list[tuple[str, float]]:
    """PIL caption burner — renders captions directly onto each image on a
    uniform WxH black canvas (caller computes W,H). Returns new entries."""
    out_dir.mkdir(parents=True, exist_ok=True)

    size = max(32, int(W * 0.030))
    margin_v = max(50, int(H * 0.07))
    pad_between_lines = max(10, int(size * 0.18))
    line_h = size + pad_between_lines
    try:
        font = ImageFont.truetype(str(font_path), size) if font_path.exists() else ImageFont.load_default(size=size)
    except Exception:
        font = ImageFont.load_default(size=size)
    print(f"[stitch] native frame {nat_w}x{nat_h} -> canvas {W}x{H}, font {size}pt", flush=True)

    # ---- one caption chunk per image entry (in entry order) -----------------
    evts: list[list[str]] = []
    for i, (_img, _sec) in enumerate(entries):
        m = moments[i] if i < len(moments) else None
        lines_for_this: list[str] = []
        if m:
            for _a, _b, txt in caption_events(m, max_words):
                lines_for_this.append(txt)
        evts.append(lines_for_this)

    new_entries: list[tuple[str, float]] = []
    for i, (img_p, sec) in enumerate(entries):
        src = Image.open(img_p).convert("RGB")
        # Pad to even canvas size (black edge, indistinguishable)
        if src.size != (W, H):
            canvas = Image.new("RGB", (W, H), (0, 0, 0))
            canvas.paste(src, (0, 0))
            src = canvas
        draw = ImageDraw.Draw(src)
        lines = evts[i]
        if lines:
            n_lines = len(lines)
            pad_x = 60
            bar_pad_y = 24
            bar_h = line_h * n_lines + bar_pad_y * 2
            bar_y = H - margin_v - bar_h
            # Translucent black strip drawn as a solid dark rectangle (no RGBA
            # paste — that's what crashed Pillow on Kaggle's build). Use 0,0,0
            # with a second darker-pen rectangle for the same visual weight.
            draw.rectangle([(0, bar_y), (W, bar_y + bar_h)], fill=(0, 0, 0))
            # 1px divider line at top of bar for crispness
            draw.rectangle([(0, bar_y), (W, bar_y + 1)], fill=(80, 80, 80))
            y = bar_y + bar_pad_y
            for line in lines:
                bb = draw.textbbox((0, 0), line, font=font)
                tw = bb[2] - bb[0]
                x = (W - tw) // 2
                # Soft black shadow (offset 2,2) for legibility
                draw.text((x + 2, y + 2), line, font=font, fill=(0, 0, 0))
                draw.text((x, y),       line, font=font, fill=(255, 255, 255))
                y += line_h
        out_p = out_dir / f"cap_{i:04d}.png"
        # PNG without `optimize=True` (triggers Pillow _idat crash on 10.4+ with RGBA)
        src.save(out_p, "PNG", optimize=False, compress_level=6)
        new_entries.append((str(out_p), sec))
    print(f"[stitch] PIL caption burn -> {len(new_entries)} frames in {out_dir}", flush=True)
    return new_entries


def write_ass(moments: list[dict], out: Path, font_name: str, play_w: int, play_h: int,
              size: int = 58) -> Path:
    """Word-locked ASS captions at native resolution."""
    head = (
        f"[Script Info]\nScriptType: v4.00+\nPlayResX: {play_w}\nPlayResY: {play_h}\nWrapStyle: 0\n\n"
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
    print(f"[stitch] ASS captions: {n} lines @ {play_w}x{play_h} -> {out.name}", flush=True)
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
    p.add_argument("--caption-method", choices=["ass", "pil", "auto"], default="pil",
                   help="pil = Pillow (default, robust on all hosts), ass = libass, auto = pil")
    p.add_argument("--font", default="", help="path to a bold sans .ttf (e.g. Inter-Bold.ttf)")
    p.add_argument("--no-loudnorm", action="store_true", help="skip -14 LUFS normalization")
    p.add_argument("--qc", default="", help="qc.jsonl from cartoonstyle.qc")
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
            print(f"[stitch] qc auto-pick: {len(qc_best)} moments scored, {swapped} non-default picks", flush=True)

        entries: list[tuple[str, float]] = []
        for m in sorted(moments, key=lambda x: x["start"]):
            pick = qc_best[m["key"]][1] if m["key"] in qc_best else args.pick
            img = find_image(args.images_dir, m["key"], pick, args.fallback_dir or None)
            if img:
                entries.append((img, float(m["sec"])))
            else:
                print(f"[stitch] WARNING: no image for {m['key']} - skipped", flush=True)
        if not entries:
            raise SystemExit("no images found")

        # Detect canvas: find MAX w/h across all frames, snap to even for x264
        max_w = max_h = 0
        for img_p, _sec in entries:
            with Image.open(img_p) as probe:
                ww, hh = probe.size
            max_w = max(max_w, ww); max_h = max(max_h, hh)
        W = max_w + (max_w % 2)
        H = max_h + (max_h % 2)
        print(f"[stitch] max frame {max_w}x{max_h} -> canvas {W}x{H}", flush=True)

        def normalize(ent: list[tuple[str, float]], outdir: Path) -> list[tuple[str, float]]:
            """Copy/paste every frame onto a uniform WxH black canvas (PIL) so
            ffmpeg concat gets identical-sized inputs with no filter needed."""
            outdir.mkdir(parents=True, exist_ok=True)
            new_ent: list[tuple[str, float]] = []
            for i, (img_p, sec) in enumerate(ent):
                im = Image.open(img_p).convert("RGB")
                if im.size != (W, H):
                    canvas = Image.new("RGB", (W, H), (0, 0, 0))
                    canvas.paste(im, (0, 0))
                    im = canvas
                out_p = outdir / f"frm_{i:04d}.png"
                im.save(out_p, "PNG", optimize=False, compress_level=6)
                new_ent.append((str(out_p), sec))
            return new_ent

        def run_encode(ent: list[tuple[str, float]], vf: list[str]):
            lst2 = work / "all_active.txt"
            lines2 = []
            for img, sec in ent:
                lines2.append(f"file '{img}'\nduration {sec:.3f}\n")
            lines2.append(f"file '{ent[-1][0]}'\n")
            lst2.write_text("".join(lines2), encoding="utf-8")
            # No -vf needed if inputs are already at uniform WxH (PIL did it).
            subprocess.run(cmd_base + ["-f", "concat", "-safe", "0", "-i", str(lst2),
                                       "-i", str(narr),
                                       *vf,
                                       "-c:v", "libx264", "-tune", "stillimage", "-r", str(args.fps),
                                       *aopts,
                                       "-c:a", "aac", "-b:a", "192k",
                                       "-pix_fmt", "yuv420p", "-shortest", args.out],
                           check=True, capture_output=True)

        aopts = [] if args.no_loudnorm else ["-af", "loudnorm=I=-14:TP=-1.5:LRA=11"]

        font_path = Path(args.font) if args.font else Path("Montserrat.ttf")

        # ---- caption path selection ---------------------------------------
        did_encode = False

        # Default = pil (robust). Only attempt ASS if explicitly requested.
        if args.captions and args.caption_method == "ass":
            family = (font_path.stem.split("[")[0].split("-")[0].strip() or "Sans")
            try:
                if font_path.exists():
                    fontsdir = str(font_path.parent.resolve())
                    vopts = ["-vf", f"ass={work / 'captions.ass'}:fontsdir={fontsdir}"]
                else:
                    print("[stitch] WARNING: font file missing - using system sans", flush=True)
                    vopts = ["-vf", f"ass={work / 'captions.ass'}"]
                size_ass = max(32, int(W * 0.030))
                write_ass(moments, work / "captions.ass", family, W, H, size_ass)
                run_encode(entries, vopts)
                did_encode = True
            except subprocess.CalledProcessError as e:
                print(f"[stitch] ASS encode failed ({e.returncode}) — falling back to PIL", flush=True)
                if e.stderr:
                    print(e.stderr.decode(errors="replace")[-400:], flush=True)
            except Exception as e:
                print(f"[stitch] ASS setup failed ({e}) — falling back to PIL", flush=True)

        if args.captions and not did_encode:
            pil_dir = work / "pil_captioned"
            cap_entries = pil_burn_captions(entries, moments,
                                            font_path if font_path.exists() else Path("/nonexistent"),
                                            pil_dir, W, H)
            # pil_burn_captions already outputs uniform WxH frames at the native
            # canvas size; still run normalize() to guarantee conformity.
            norm_dir = work / "pil_norm"
            cap_entries = normalize(cap_entries, norm_dir)
            lst_cap = work / "all_pil.txt"
            _write_concat(lst_cap, cap_entries)
            subprocess.run(cmd_base + ["-f", "concat", "-safe", "0", "-i", str(lst_cap),
                                       "-i", str(narr),
                                       "-c:v", "libx264", "-tune", "stillimage", "-r", str(args.fps),
                                       *aopts,
                                       "-c:a", "aac", "-b:a", "192k",
                                       "-pix_fmt", "yuv420p", "-shortest", args.out],
                           check=True, capture_output=True)
            did_encode = True
        elif not args.captions:
            # Pre-normalize frames to identical even canvas via PIL so concat
            # demuxer sees uniform inputs (ffmpeg's pad filter post-concat fails).
            norm_dir = work / "normalized"
            norm_entries = normalize(entries, norm_dir)
            run_encode(norm_entries, [])
            did_encode = True

        print(f"[stitch] VIDEO -> {args.out}  ({len(entries)} word-locked images, one narration take)", flush=True)
        return

    segments = []
    missing = []

    for b in beats:
        n = int(b["beat"])
        audio = Path(args.timing).parent / "audio" / f"beat{n:02d}.mp3"
        seg_entries: list[tuple[str, float]] = []
        for m in sorted(by_beat.get(n, []), key=lambda x: x["start"]):
            img = find_image(args.images_dir, m["key"], args.pick, args.fallback_dir or None)
            if img:
                seg_entries.append((img, float(m["sec"])))
        if not seg_entries:
            img = find_image(args.images_dir, f"beat{n:02d}", args.pick, args.fallback_dir or None)
            if img:
                seg_entries.append((img, float(b["sec"])))
        if not seg_entries or not audio.exists():
            missing.append(n)
            continue
        seg = beat_segment(cmd_base, work, n, seg_entries, audio)
        segments.append(seg)
        print(f"[stitch] beat {n:>2}  {b['sec']:>5.1f}s  {len(seg_entries)} image(s)", flush=True)

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


def _write_concat(path: Path, entries: list[tuple[str, float]]) -> Path:
    lines = []
    for img, sec in entries:
        lines.append(f"file '{img}'\nduration {sec:.3f}\n")
    lines.append(f"file '{entries[-1][0]}'\n")
    path.write_text("".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    main()
