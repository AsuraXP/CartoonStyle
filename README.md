# CartoonStyle — Narration-First Explainer Video Pipeline

Production workflow for explainer episodes: **research → script → anchors →
frames → measured render**. Narration is the master timeline; frames react to
the words. Frame durations are *measured* between anchor timestamps — never
hand-set. The governing spec (paste it into fresh sessions) is
[docs/WORKFLOW_SPEC.md](docs/WORKFLOW_SPEC.md); episode progress and
calibration records live in [PRODUCTION.md](PRODUCTION.md).

Repo: `https://github.com/AsuraXP/CartoonStyle.git` ·
branch `arena/943137f3-cartoonstyle` (drop the `-b` once this lands on `main`).

## Three locations

| Location | Does | Never |
|---|---|---|
| **Workspace** | research, script, anchors, frame generation + visual QC, code/config | runs TTS |
| **GitHub** | source of truth, cloneable without auth | holds large binaries |
| **Kaggle** | TTS + measured timing, stitching | authors content, edits prompts, regenerates frames |

## Repository layout

```
docs/WORKFLOW_SPEC.md    the workflow spec (v2.1) — the pasteable prompt
PRODUCTION.md            episode progress + calibration record
frames.json              the ONLY source of truth for frame count (Phase 2)
narration/preflight.md   Phase 0 calibration passage (~60 words, TTS-marked)
narration/narration.md   episode script (Phase 1, TTS-marked + ## Sources)
kaggle/config.yaml       engine + visual settings (sec_per_word filled by measurement)
kaggle/preflight.py      Phase 0 entry point (Kaggle-only synthesis)
kaggle/run.py            Phase 2/4/5 entry point: --estimate / --dry-run / --out
vidsync/                 shared pipeline library (matching, timing, stitch, media)
tools/prep_frames.py     Phase 3: compress to 1280x720 JPG + contact sheet
tools/selftest.py        Phase 4: offline pipeline proof (stand-in TTS, no network)
assets/STYLE.md          frozen art style tokens (Phase 3, after user's choice)
assets/frames/           reviewed compressed frame JPGs (committed)
build/                   synthesis + timing artifacts (gitignored, Kaggle)
```

## Phase 0 — Kaggle preflight cells (run in a Kaggle notebook, internet ON)

New notebook (first cell):

```python
!git clone --branch arena/943137f3-cartoonstyle https://github.com/AsuraXP/CartoonStyle.git /kaggle/working/video-repo
```

Later sessions (skip cloning again):

```python
!git -C /kaggle/working/video-repo pull --ff-only
```

Install and run the calibration:

```python
!pip install -q --upgrade --force-reinstall edge-tts pyyaml
!python -c "import edge_tts; print('edge-tts', edge_tts.__version__)"
!python /kaggle/working/video-repo/kaggle/preflight.py
```

No dataset upload, no ZIP, no manual copying — everything comes from the
public clone. Share the printed report: it is recorded in PRODUCTION.md and
`kaggle/config.yaml -> estimate.sec_per_word`.

## Later phases — Kaggle cells (after anchors + frames are pushed)

Measure the full narration (one continuous request; artifacts are preserved):

```python
!git -C /kaggle/working/video-repo pull --ff-only
!python /kaggle/working/video-repo/kaggle/run.py --repo /kaggle/working/video-repo --dry-run
```

Render (REUSES the preserved Phase 4 audio/timing; never re-synthesizes):

```python
!python /kaggle/working/video-repo/kaggle/run.py --repo /kaggle/working/video-repo --out final.mp4
```

`final.mp4` and `build/timings.csv` land in `/kaggle/working/video-repo/build/`.

## Workspace commands (never synthesize here)

```bash
python kaggle/preflight.py --validate-only            # offline preflight checks
python kaggle/run.py --estimate --repo .              # Gate 2: anchor/estimate check
python tools/prep_frames.py --repo . --sheet --strict # Gate 3: compress + contact sheet
python tools/selftest.py                              # Gate 4: pipeline selftest (43/43)
```

These are the only tests needed — run them, never write new harnesses.

Dependencies: `pip install -r requirements.txt` (edge-tts, pyyaml, pillow;
imageio-ffmpeg is an optional ffmpeg fallback for the selftest).
