# Multi-Object Tracking

Detection + multi-object tracking built on YOLO (Ultralytics) with ByteTrack / BoT-SORT, evaluated with `motmetrics`.

## Setup (Windows)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e .
pytest
```

## Run

```powershell
python scripts/track.py --source path\to\video.mp4
python scripts/track.py --source 0 --tracker botsort.yaml   # webcam
```

Results are written to `outputs/track*/`.

## VisDrone2019-MOT

Extract the Task 4 zips into `data/raw/VisDrone/` (each split folder holds `sequences/` and `annotations/`), then:

```powershell
python scripts/run_sequences.py --split data/raw/VisDrone/VisDrone2019-MOT-val
python scripts/evaluate.py --gt data/raw/VisDrone/VisDrone2019-MOT-val/annotations --pred outputs/VisDrone2019-MOT-val_bytetrack
```

Evaluation follows the official protocol: per class (pedestrian, car, van, truck, bus), IoU ≥ 0.5,
boxes inside ignored regions dropped, `mean` row = class average. HOTA is computed with a
re-implementation of the TrackEval algorithm (`src/mot/eval/hota.py`).

`run_sequences.py` also writes `speed.json`: per-frame time for detection (pre/inference/post),
tracker update, and end-to-end wall clock including image loading (first 10 frames of each sequence excluded).

### Baseline (YOLO26n COCO weights, imgsz 1280, ByteTrack, RTX 4060 Ti)

| split | HOTA | MOTA | IDF1 | IDsw | FPS (end-to-end) | FPS (detect+track) |
|---|---|---|---|---|---|---|
| val | 19.1 | -6.2 | 21.2 | 547 | 51.9 | 63.9 |
| test-dev | 26.4 | -0.1 | 31.8 | 1852 | 55.3 | 66.1 |

Per-class test-dev (HOTA / MOTA / IDF1): pedestrian 27.8 / 19.9 / 33.7, car 44.7 / 39.9 / 55.0,
van 0 / 0 / 0 (no COCO class), truck 27.2 / 1.9 / 30.0, bus 32.2 / -62.4 / 40.2.

## Layout

```
configs/          YAML run configs
data/raw/         input videos / datasets (e.g. MOT17) — not tracked by git
data/processed/   converted / cached data
models/           model weights — not tracked by git
notebooks/        exploration
outputs/          tracking results and rendered videos
scripts/          entry-point scripts
src/mot/
  detectors/      detector wrappers
  trackers/       tracker implementations
  utils/          I/O, visualization, metrics helpers
tests/
```
