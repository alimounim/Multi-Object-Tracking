"""Run the detector once over a split and cache its detections for offline tracker tuning.

Usage:
    python scripts/cache_detections.py --weights models/yolo26n_visdrone.pt --class-map visdrone \
        --split data/raw/VisDrone/VisDrone2019-MOT-val --out outputs/cache/val_yolo26n_visdrone
"""
import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from mot.datasets.visdrone import CLASS_MAPS, list_sequences


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True)
    parser.add_argument("--class-map", choices=sorted(CLASS_MAPS), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--conf", type=float, default=0.05, help="Keep low; trackers threshold later")
    parser.add_argument("--max-det", type=int, default=1000)
    args = parser.parse_args()

    class_map = CLASS_MAPS[args.class_map]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.weights)
    lut = np.full(max(model.names) + 1, -1)
    for k, v in class_map.items():
        lut[k] = v

    for seq in tqdm(list_sequences(args.split), desc="sequences"):
        frame, xyxy, conf, cls = [], [], [], []
        results = model.predict(
            source=str(seq.image_dir), imgsz=args.imgsz, conf=args.conf, max_det=args.max_det,
            classes=sorted(class_map), device=0, stream=True, verbose=False,
        )
        for r in tqdm(results, total=len(seq), desc=seq.name, leave=False):
            b = r.boxes.cpu().numpy()
            frame.append(np.full(len(b), int(Path(r.path).stem)))
            xyxy.append(b.xyxy)
            conf.append(b.conf)
            cls.append(lut[b.cls.astype(int)])
        np.savez_compressed(
            out / f"{seq.name}.npz",
            frame=np.concatenate(frame), xyxy=np.concatenate(xyxy), conf=np.concatenate(conf), cls=np.concatenate(cls),
        )
    print(f"Detections cached to {out}")


if __name__ == "__main__":
    main()
