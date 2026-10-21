# Is humanity inherently good or evil? — "How We Got Here" ep2

## Single cell — draw (4B, 12 steps, char-only ref, no judge) + narration + captioned VIDEO

```python
import os, shutil, subprocess, sys

if not os.path.exists("/kaggle/working/CartoonStyle"):
    subprocess.run("git clone --depth 1 -b arena/22c9cd35-cartoonstyle "
                   "https://github.com/AsuraXP/CartoonStyle.git",
                   shell=True, cwd="/kaggle/working", check=True)
os.chdir("/kaggle/working/CartoonStyle")
subprocess.run("git pull", shell=True, check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "edge-tts", "diffusers>=0.35", "transformers>=4.50", "accelerate",
                "sentencepiece", "protobuf"], check=True)

OUT = "/kaggle/working/good-evil-v1"
shutil.rmtree(OUT, ignore_errors=True)
os.makedirs(OUT, exist_ok=True)

subprocess.run([sys.executable, "-m", "cartoonstyle.generate_flux2",
                "--prompts", "examples/inherent_good_evil/visuals.csv",
                "--images", "75", "--gpus", "1", "--steps", "12",
                "--out", OUT, "--qc-retries", "0"], check=True)

subprocess.run([sys.executable, "-m", "cartoonstyle.tts",
                "--beats", "examples/inherent_good_evil/beats.jsonl",
                "--visuals", "examples/inherent_good_evil/visuals.csv",
                "--out-dir", "/kaggle/working/good-evil-tts"], check=True)
subprocess.run("curl -sL -o /kaggle/working/Montserrat.ttf "
               "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf",
               shell=True, check=True)
subprocess.run([sys.executable, "-m", "cartoonstyle.stitch",
                "--timing", "/kaggle/working/good-evil-tts/beats_with_timing.json",
                "--images-dir", OUT + "/images",
                "--out", "/kaggle/working/VIDEO.mp4",
                "--captions", "--font", "/kaggle/working/Montserrat.ttf"], check=True)
print("DONE -> /kaggle/working/VIDEO.mp4")
```

## Startup sanity
- `code version:` latest (`b70c9f1` or later)
- `1 character anchors` (stock refs removed — no 6-arm gremlin)
- NO `vision QC armed`
- `[i/75] ... ~38-42s saved`
- `[tts] total narration: ~3:08` (not 1 minute)
- Final VIDEO.mp4 ~3 min long, images flip on Andrew's words
