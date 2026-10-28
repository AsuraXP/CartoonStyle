# Night 1 — Trapped in My Bedroom (Video 6)

**Status:** FRAMES + ASSEMBLY SCRIPT READY. Build on Kaggle using `run_bedroom_kaggle.sh`.

- **Type:** standalone second-person horror-comedy short (same stick-figure B&W inkwash house style as Video 5)
- **Setting:** your bedroom, second-floor apartment, Tuesday night / Wednesday morning. Entire video seen from inside one room (plus the final exterior end card).
- **Runtime:** ~7:55 (44 frames, 1,469 words of narration, scheduled to beats.jsonl)
- **POV:** second-person ("you …"), deadpan
- **Voice:** en-US-AndrewNeural, +10% rate (generated at build time via edge-tts)
- **Style:** B&W gritty ink-wash, high-contrast chiaroscuro, Walking-Dead-comic texture, stick figures. Accents: warm YELLOW (desk lamp / flashlight), RED (Henderson's sneakers, alarm-clock sound waves, clock radio), ORANGE (the cat).
- **Aspect:** 16:9 (YouTube)
- **Captions:** none
- **Recurring gimmick:** the dead RETURN and REPEAT (Mr. Henderson on a 75-minute cycle). Silence is survival. Loud = somewhere else. The living knock three times.

## Build (Kaggle)

```
!rm -rf CartoonStyle img_pool
!git clone --depth 1 --branch arena/22c9cd35-cartoonstyle https://github.com/AsuraXP/CartoonStyle.git
!bash CartoonStyle/examples/bedroom_night1_video/run_bedroom_kaggle.sh
```

Output: `bedroom_night1.mp4` in the working directory.

## Contents
- `script.md` — full narration (1,469 words)
- `visuals.csv` — 44 frames / 20 beats, with per-frame visual prompts
- `beats.jsonl` — 44 rows mapping each frame PNG to its narration and scheduled duration (seconds)
- `frames/` — 44 PNG story frames (beat01m01.png through beat20m04.png)
- `refs/` — locked style anchors (test01–test05)
- `run_bedroom_kaggle.sh` — build script (edge-tts + ffmpeg concat)
- `continuity.md` — locked style/character/set bible
- `brief.md`, `metadata.json`, `youtube.txt` — packaging for YouTube upload
- `STATE.md` — production log
