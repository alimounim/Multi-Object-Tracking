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

### Fine-tuned detector + tuned ByteTrack

Build the training set and fine-tune (VisDrone MOT train + VisDrone2019-DET train, 10 classes):

```powershell
python scripts/prepare_yolo_dataset.py --det --out data/processed/visdrone_mot_det_yolo
python scripts/train_detector.py --data data/processed/visdrone_mot_det_yolo/data.yaml --name yolo26n_visdrone_det --epochs 40 --patience 10
python scripts/run_sequences.py --split data/raw/VisDrone/VisDrone2019-MOT-test-dev --weights models/yolo26n_visdrone_det.pt --class-map visdrone --tracker configs/trackers/bytetrack_visdrone.yaml --out outputs/test-dev_final
```

Training stopped early at epoch 36 (best val mAP50-95 0.249 at epoch 26; 0.178 with MOT train only).
Tracker settings in `configs/trackers/bytetrack_visdrone.yaml` were tuned on val with `scripts/tune_tracker.py`.

Test-dev ablation (rows without FPS replay cached detections of the same detector):

| detector training data | tracker settings | HOTA | IDF1 | MOTA | IDsw | FPS (end-to-end) |
|---|---|---|---|---|---|---|
| none (COCO weights) | default | 26.4 | 31.8 | -0.1 | 1852 | 55.3 |
| MOT train | default | 35.3 | 40.2 | -12.5 | 4002 | 54.7 |
| MOT train | tuned | 38.3 | 46.7 | 6.9 | 1494 | — |
| MOT + DET train | default | 41.0 | 47.9 | 10.2 | 4469 | — |
| **MOT + DET train** | **tuned** | **45.6** | **57.6** | **37.1** | **1417** | **53.8** |

Per-class test-dev for the final system (HOTA / MOTA / IDF1): pedestrian 34.0 / 30.9 / 44.0,
car 55.4 / 62.5 / 71.5, van 43.0 / 29.9 / 52.0, truck 44.4 / 40.3 / 54.6, bus 51.2 / 22.2 / 65.6.

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
