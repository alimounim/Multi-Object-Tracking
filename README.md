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
boxes inside ignored regions dropped, `mean` row = class average.

### Baseline (val, YOLO26n COCO weights, imgsz 1280, ByteTrack)

| class | MOTA | IDF1 | IDsw |
|---|---|---|---|
| pedestrian | 10.9 | 35.6 | 324 |
| car | 21.5 | 44.3 | 214 |
| van | 0.0 | 0.0 | 0 |
| truck | -38.3 | 25.9 | 9 |
| bus | -25.0 | 0.0 | 0 |
| **mean** | **-6.2** | **21.2** | 547 |

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
