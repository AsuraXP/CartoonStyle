# Night 1 — Trapped in My Bedroom (Video 6)

**Status:** READY TO BUILD on Kaggle. 44 frames across 20 beats, built via `run_bedroom_kaggle.sh` using the shared `cartoonstyle` pipeline (edge-tts + word-locked ffmpeg stitch, native resolution, no captions).

- **Type:** standalone second-person horror-comedy short (same B&W inkwash stick-figure house style as Video 5)
- **Setting:** your bedroom, second-floor apartment, Tuesday night / Wednesday morning. Entire video inside one room (plus the final exterior end card).
- **Runtime:** ~7:30–8:00 (TTS-driven; pipeline auto-keys frame cuts to spoken words)
- **POV:** second-person ("you …"), deadpan
- **Voice:** en-US-AndrewNeural, +10% rate
- **Style:** B&W gritty ink-wash, high-contrast chiaroscuro, Walking-Dead-comic texture, stick figures. Accents: warm YELLOW (desk lamp / flashlight), RED (Henderson's sneakers, alarm sound waves, clock radio), ORANGE (the cat).
- **Aspect:** 16:9 (YouTube), native resolution, no captions
- **Recurring gimmick:** the dead RETURN and REPEAT (Mr. Henderson on a 75-minute cycle). Silence is survival. Loud = somewhere else. The living knock three times.

## Build (Kaggle)

Open a fresh Kaggle notebook with Internet ON, run three cells:

```
!rm -rf CartoonStyle img_pool
!git clone --depth 1 --branch arena/22c9cd35-cartoonstyle https://github.com/AsuraXP/CartoonStyle.git
!bash CartoonStyle/examples/bedroom_night1_video/run_bedroom_kaggle.sh
```

Output: `/kaggle/working/bedroom_night1_DELIVERABLE.zip` containing VIDEO.mp4 + thumbnail + metadata + script + youtube description.

## Contents
- `script.md` — full narration (~1,470 words)
- `visuals.csv` — 44 rows (20 beats / slide_ids), one row per frame, with per-frame visual prompts
- `frames/` — 44 PNG story frames (beat01m01.png through beat20m04.png)
- `refs/` — 5 locked style anchors (test01–test05: MC+desk, door-gap/Henderson, dark barricade, cat under-bed, cat floor-loaf)
- `run_bedroom_kaggle.sh` — one-shot Kaggle build script (matches Video 5 pipeline conventions)
- `continuity.md` — locked style/character/set bible (lighting progression, drift watch)
- `brief.md`, `metadata.json`, `youtube.txt`, `thumbnail.png` — packaging for YouTube upload
- `STATE.md` — production log
