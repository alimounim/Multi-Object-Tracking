"""Run detection + tracking over every sequence of a VisDrone MOT split.

Writes one VisDrone-format result file per sequence:
    frame, track_id, x, y, w, h, score, category, -1, -1

Usage:
    python scripts/run_sequences.py --split data/raw/VisDrone/VisDrone2019-MOT-val
    python scripts/run_sequences.py --split ... --tracker botsort.yaml --out outputs/val_botsort
    python scripts/run_sequences.py --split ... --tracker botsort.yaml --gmc-width 640  # cheaper camera-motion compensation
"""
import argparse
import json
import time
from pathlib import Path

import yaml
from tqdm import tqdm
from ultralytics import YOLO

from mot.datasets.visdrone import CLASS_MAPS, list_sequences
from mot.trackers.gmc import fixed_width_gmc
from mot.utils.timing import STAGES, FrameTimer, time_tracker_updates


def track_sequence(model, seq, det, tracker, class_map, timer):
    # persist=True: without it Ultralytics resets the tracker whenever the file path changes,
    # i.e. on every frame of an image folder. Reset explicitly between sequences instead.
    for t in getattr(model.predictor, "trackers", []):
        t.reset()
    rows, total_ms, detect_ms, track_ms = [], [], [], []
    results = model.track(
        source=str(seq.image_dir),
        tracker=tracker,
        conf=det["conf"],
        iou=det["iou"],
        imgsz=det["imgsz"],
        classes=sorted(class_map),
        device=det["device"],
        stream=True,
        persist=True,
        verbose=False,
    )
    with time_tracker_updates(track_ms):
        prev = time.perf_counter()
        for r in tqdm(results, total=len(seq), desc=seq.name, leave=False):
            now = time.perf_counter()
            total_ms.append((now - prev) * 1000)
            prev = now
            detect_ms.append(sum(r.speed.values()))
            frame = int(Path(r.path).stem)
            b = r.boxes
            if b.id is None:
                continue
            for (x1, y1, x2, y2), tid, score, c in zip(b.xyxy.tolist(), b.id.int().tolist(), b.conf.tolist(), b.cls.int().tolist()):
                cat = class_map.get(c, -1)
                rows.append(f"{frame},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{score:.4f},{cat},-1,-1")
    timer.add_sequence(total_ms, detect_ms, track_ms)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", help="Overrides dataset.split in the config")
    parser.add_argument("--tracker", help="Overrides tracker.type in the config")
    parser.add_argument("--out", help="Output folder (default: outputs/<split>_<tracker>)")
    parser.add_argument("--weights", help="Overrides detector.weights in the config")
    parser.add_argument("--class-map", choices=sorted(CLASS_MAPS), help="Overrides detector.class_map")
    parser.add_argument("--imgsz", type=int, help="Overrides detector.imgsz")
    parser.add_argument("--gmc-width", type=int, help="Run camera-motion compensation on frames at most this wide "
                        "(default: Ultralytics' fixed 2x downscale)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    det = cfg["detector"]
    det["weights"] = args.weights or det["weights"]
    det["class_map"] = args.class_map or det["class_map"]
    det["imgsz"] = args.imgsz or det["imgsz"]
    tracker = args.tracker or cfg["tracker"]["type"]
    split = Path(args.split or cfg["dataset"]["split"])
    class_map = CLASS_MAPS[det["class_map"]]
    out = Path(args.out or Path(cfg["output"]["dir"]) / f"{split.name}_{Path(tracker).stem}")
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(det["weights"])
    timer = FrameTimer()
    with fixed_width_gmc(args.gmc_width):
        for seq in tqdm(list_sequences(split), desc="sequences"):
            rows = track_sequence(model, seq, det, tracker, class_map, timer)
            (out / f"{seq.name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))

    speed = {"weights": det["weights"], "imgsz": det["imgsz"], "tracker": tracker, "gmc_width": args.gmc_width,
             **timer.summary()}
    (out / "speed.json").write_text(json.dumps(speed, indent=2))
    print(f"Results written to {out}")
    print(
        f"{speed['fps']:.1f} FPS end-to-end ({speed['fps_model_only']:.1f} FPS detect+track) | per frame: "
        + ", ".join(f"{s} {speed[f'{s}_ms']:.1f} ms" for s in STAGES)
    )


if __name__ == "__main__":
    main()
