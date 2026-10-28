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

### Final system: fine-tuned detector + BoT-SORT with camera-motion compensation

Build the training set and fine-tune (VisDrone MOT train + VisDrone2019-DET train, 10 classes), then track:

```powershell
python scripts/prepare_yolo_dataset.py --det --out data/processed/visdrone_mot_det_yolo
python scripts/train_detector.py --data data/processed/visdrone_mot_det_yolo/data.yaml --name yolo26n_visdrone_det --epochs 40 --patience 10
python scripts/run_sequences.py --split data/raw/VisDrone/VisDrone2019-MOT-test-dev --weights models/yolo26n_visdrone_det.pt --class-map visdrone --tracker configs/trackers/botsort_visdrone.yaml --gmc-width 480 --out outputs/test-dev_final
```

`--gmc-width 480` runs camera-motion compensation (sparse optical flow) on frames at most 480 px wide.
Ultralytics always halves the frame instead, which costs 39 ms/frame on VisDrone's 4K sequences.

Training stopped early at epoch 36 (best val mAP50-95 0.249 at epoch 26; 0.178 with MOT train only).
Tracker settings in `configs/trackers/` were tuned on val with `scripts/tune_tracker.py`.

Test-dev ablation (rows without FPS replay cached detections of the same detector):

| detector training data | tracker settings | HOTA | IDF1 | MOTA | IDsw | FPS (end-to-end) |
|---|---|---|---|---|---|---|
| none (COCO weights) | default | 26.4 | 31.8 | -0.1 | 1852 | 55.3 |
| MOT train | default | 35.3 | 40.2 | -12.5 | 4002 | 54.7 |
| MOT train | tuned | 38.3 | 46.7 | 6.9 | 1494 | — |
| MOT + DET train | default | 41.0 | 47.9 | 10.2 | 4469 | — |
| MOT + DET train | tuned ByteTrack | 45.6 | 57.6 | 37.1 | 1417 | 53.8 |
| MOT + DET train | tuned BoT-SORT | 46.7 | 58.8 | 40.6 | 1411 | 62.8 |
| **MOT + DET train** | **tuned BoT-SORT + GMC (width 480)** | **49.9** | **63.5** | **42.1** | **527** | **54.0** |

Per-class test-dev for the final system (HOTA / MOTA / IDF1): pedestrian 36.7 / 32.5 / 48.3,
car 60.8 / 64.3 / 79.5, van 45.3 / 30.1 / 53.7, truck 49.2 / 41.2 / 60.4, bus 57.2 / 42.1 / 75.5.

### Tracker comparison

All Ultralytics trackers were tuned on val with the same grid (`track_high_thresh`, `new_track_thresh`,
`match_thresh`) on cached detections; every tracker's best setting was `new_track_thresh` 0.65,
`match_thresh` 0.9. Best val HOTA without camera-motion compensation (GMC): BoT-SORT 43.8, OC-SORT 41.9,
ByteTrack 41.8, FastTrack 41.6 (Deep OC-SORT without ReID/GMC is identical to OC-SORT).

Live test-dev runs (end-to-end FPS on an RTX 4060 Ti; val FPS in brackets, val has many 4K frames):

| tracker | GMC | HOTA | IDF1 | MOTA | IDsw | FPS test-dev (val) |
|---|---|---|---|---|---|---|
| ByteTrack | — | 45.6 | 57.6 | 37.1 | 1417 | 53.8 (58.7) |
| BoT-SORT | none | 46.7 | 58.8 | 40.6 | 1411 | 62.8 (62.1) |
| BoT-SORT | Ultralytics 2x downscale | 49.8 | 63.5 | 42.2 | 531 | 39.6 (22.0) |
| BoT-SORT | width 960 | 49.8 | 63.5 | 42.1 | 537 | 46.8 (41.6) |
| BoT-SORT | width 640 | 49.9 | 63.5 | 42.0 | 530 | 53.2 (49.7) |
| **BoT-SORT** | **width 480** | **49.9** | **63.5** | **42.1** | **527** | **54.0 (52.3)** |
| TrackTrack (defaults) | Ultralytics 2x downscale | 48.0 | 60.8 | 41.4 | 287 | 41.8 (19.3) |

Render several trackers side by side on one sequence:

```powershell
python scripts/compare_video.py --seq data/raw/VisDrone/VisDrone2019-MOT-test-dev/sequences/uav0000120_04775_v `
    --panel "ByteTrack=outputs/test-dev_bytetrack" --panel "BoT-SORT + GMC=outputs/test-dev_final" `
    --out outputs/uav0000120_04775_v_grid.mp4
```

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
