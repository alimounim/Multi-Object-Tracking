"""Run detection + tracking over every sequence of a VisDrone MOT split.

Writes one VisDrone-format result file per sequence:
    frame, track_id, x, y, w, h, score, category, -1, -1

Usage:
    python scripts/run_sequences.py --split data/raw/VisDrone/VisDrone2019-MOT-val
    python scripts/run_sequences.py --split ... --tracker botsort.yaml --out outputs/val_botsort
"""
import argparse
from pathlib import Path

import yaml
from tqdm import tqdm
from ultralytics import YOLO

from mot.datasets.visdrone import COCO_TO_VISDRONE, list_sequences


def track_sequence(model, seq, det, tracker, class_map):
    # persist=True: without it Ultralytics resets the tracker whenever the file path changes,
    # i.e. on every frame of an image folder. Reset explicitly between sequences instead.
    for t in getattr(model.predictor, "trackers", []):
        t.reset()
    rows = []
    results = model.track(
        source=str(seq.image_dir),
        tracker=tracker,
        conf=det["conf"],
        iou=det["iou"],
        imgsz=det["imgsz"],
        classes=sorted(class_map) if class_map else None,
        device=det["device"],
        stream=True,
        persist=True,
        verbose=False,
    )
    for r in tqdm(results, total=len(seq), desc=seq.name, leave=False):
        frame = int(Path(r.path).stem)
        b = r.boxes
        if b.id is None:
            continue
        for (x1, y1, x2, y2), tid, score, c in zip(b.xyxy.tolist(), b.id.int().tolist(), b.conf.tolist(), b.cls.int().tolist()):
            cat = class_map.get(c, -1) if class_map else c
            rows.append(f"{frame},{tid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},{score:.4f},{cat},-1,-1")
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", help="Overrides dataset.split in the config")
    parser.add_argument("--tracker", help="Overrides tracker.type in the config")
    parser.add_argument("--out", help="Output folder (default: outputs/<split>_<tracker>)")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    det = cfg["detector"]
    tracker = args.tracker or cfg["tracker"]["type"]
    split = Path(args.split or cfg["dataset"]["split"])
    class_map = COCO_TO_VISDRONE if det.get("coco_classes") else None
    out = Path(args.out or Path(cfg["output"]["dir"]) / f"{split.name}_{Path(tracker).stem}")
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(det["weights"])
    for seq in tqdm(list_sequences(split), desc="sequences"):
        rows = track_sequence(model, seq, det, tracker, class_map)
        (out / f"{seq.name}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
    print(f"Results written to {out}")


if __name__ == "__main__":
    main()
