"""Convert VisDrone2019-MOT splits into an Ultralytics YOLO detection dataset.

- every `--stride`-th frame is kept (consecutive frames are near-duplicates)
- the 10 VisDrone object categories become YOLO classes 0..9 (category - 1)
- ignored regions (category 0 / score 0) and 'others' (11) are painted gray so unlabeled
  objects there are not learned as background; objects mostly inside them are dropped
  (as in evaluation) and the remaining labeled objects are left unpainted
- frames with a longer side above `--max-side` are downscaled (labels are normalized)

Usage:
    python scripts/prepare_yolo_dataset.py
    python scripts/prepare_yolo_dataset.py --stride 10 --out data/processed/visdrone_mot_yolo_s10
    python scripts/prepare_yolo_dataset.py --det --out data/processed/visdrone_mot_det_yolo   # + VisDrone-DET train
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml
from tqdm import tqdm

from mot.datasets.visdrone import CATEGORIES, IGNORED, OTHERS, TRAIN_CLASSES, list_sequences, load_det_annotations
from mot.eval.visdrone_eval import ignored_overlap

FILL = (114, 114, 114)  # Ultralytics' letterbox gray


def convert_frame(img, rows, max_side):
    h, w = img.shape[:2]
    ignore = np.isin(rows[:, 7], (IGNORED, OTHERS)) | (rows[:, 6] == 0)
    regions = rows[ignore][:, 2:6]
    objects = rows[~ignore & np.isin(rows[:, 7], TRAIN_CLASSES)]
    # objects mostly inside an ignored region are dropped, as in evaluation
    objects = objects[ignored_overlap(objects[:, 2:6], regions) <= 0.5]

    original = img.copy()
    for x, y, bw, bh in regions.astype(int):
        img[max(y, 0) : y + bh, max(x, 0) : x + bw] = FILL
    for x, y, bw, bh in objects[:, 2:6].astype(int):  # keep labeled objects visible
        sl = (slice(max(y, 0), y + bh), slice(max(x, 0), x + bw))
        img[sl] = original[sl]

    labels = []
    for x, y, bw, bh, cat in objects[:, [2, 3, 4, 5, 7]]:
        x1, y1 = np.clip([x, y], 0, [w, h])
        x2, y2 = np.clip([x + bw, y + bh], 0, [w, h])
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
        labels.append(f"{int(cat) - 1} {cx:.6f} {cy:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")

    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    return img, labels


def write_samples(samples, out, name, max_side):
    """samples: iterable of (output stem, image path, (N, 10) MOT-layout annotation rows)."""
    img_dir, lbl_dir = out / "images" / name, out / "labels" / name
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    n_img = n_box = 0
    for stem, path, rows in samples:
        img, labels = convert_frame(cv2.imread(str(path)), rows, max_side)
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        (lbl_dir / f"{stem}.txt").write_text("\n".join(labels) + ("\n" if labels else ""))
        n_img += 1
        n_box += len(labels)
    return n_img, n_box


def mot_samples(split_dir, stride):
    for seq in tqdm(list_sequences(split_dir), desc=split_dir.name):
        ann = seq.annotations()
        for path in seq.frames[::stride]:
            yield f"{seq.name}_{path.stem}", path, ann[ann[:, 0] == int(path.stem)]


def det_samples(det_dir):
    for path in tqdm(sorted((det_dir / "images").glob("*.jpg")), desc=det_dir.name):
        yield f"det_{path.stem}", path, load_det_annotations(det_dir / "annotations" / f"{path.stem}.txt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="data/raw/VisDrone")
    parser.add_argument("--out", default="data/processed/visdrone_mot_yolo")
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-side", type=int, default=1920)
    parser.add_argument("--det", action="store_true", help="Also add VisDrone2019-DET-train images to training")
    args = parser.parse_args()

    root, out = Path(args.root), Path(args.out)
    parts = {
        "train": mot_samples(root / "VisDrone2019-MOT-train", args.stride),
        "val": mot_samples(root / "VisDrone2019-MOT-val", args.stride),
    }
    if args.det:
        parts["train_det"] = det_samples(root / "VisDrone2019-DET-train")
    for name, samples in parts.items():
        n_img, n_box = write_samples(samples, out, name, args.max_side)
        print(f"{name}: {n_img} images, {n_box} boxes")

    data = {
        "path": str(out.resolve()),
        "train": ["images/train", "images/train_det"] if args.det else "images/train",
        "val": "images/val",
        "names": {i: CATEGORIES[c] for i, c in enumerate(TRAIN_CLASSES)},
    }
    (out / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    print(f"Dataset written to {out}")


if __name__ == "__main__":
    main()
