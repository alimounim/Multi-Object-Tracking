"""VisDrone2019-MOT (Task 4) dataset access.

Annotation line: frame, track_id, x, y, w, h, score, category, truncation, occlusion
Category 0 marks ignored regions; only EVAL_CLASSES count in the official evaluation.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CATEGORIES = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}
IGNORED = 0
OTHERS = 11
EVAL_CLASSES = (1, 4, 5, 6, 9)  # pedestrian, car, van, truck, bus
TRAIN_CLASSES = tuple(range(1, 11))  # categories a detector is trained on (YOLO index i -> category i + 1)

# Detector class index -> VisDrone category
CLASS_MAPS = {
    "coco": {0: 1, 1: 3, 2: 4, 3: 10, 5: 9, 7: 6},  # stock COCO weights
    "visdrone": {i: c for i, c in enumerate(TRAIN_CLASSES)},  # weights fine-tuned on VisDrone
}

COLUMNS = ("frame", "id", "x", "y", "w", "h", "score", "category", "truncation", "occlusion")


def load_annotations(path):
    """Load an annotation/result file as an (N, 10) float array (empty if no rows)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, len(COLUMNS)))
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.shape[1] < len(COLUMNS):  # results files may omit trailing columns
        pad = np.full((len(data), len(COLUMNS) - data.shape[1]), -1.0)
        data = np.hstack([data, pad])
    return data[:, : len(COLUMNS)]


def load_det_annotations(path):
    """Load a VisDrone-DET annotation file (x, y, w, h, score, category, truncation, occlusion)
    into the same (N, 10) layout as MOT annotations, with frame = 0 and id = -1."""
    data = np.loadtxt(path, delimiter=",", ndmin=2, usecols=range(8)) if Path(path).stat().st_size else np.zeros((0, 8))
    return np.hstack([np.zeros((len(data), 1)), np.full((len(data), 1), -1.0), data])


@dataclass
class Sequence:
    name: str
    image_dir: Path
    annotation_path: Path | None = None
    frames: list[Path] = field(default_factory=list)

    def __post_init__(self):
        if not self.frames:
            self.frames = sorted(self.image_dir.glob("*.jpg"))

    def __len__(self):
        return len(self.frames)

    def annotations(self):
        if self.annotation_path is None:
            return np.zeros((0, len(COLUMNS)))
        return load_annotations(self.annotation_path)


def list_sequences(split_dir):
    """Return the sequences of a split dir containing `sequences/` and (optionally) `annotations/`."""
    split_dir = Path(split_dir)
    seq_root, ann_root = split_dir / "sequences", split_dir / "annotations"
    if not seq_root.is_dir():
        raise FileNotFoundError(f"No sequences/ folder in {split_dir}")
    seqs = []
    for d in sorted(p for p in seq_root.iterdir() if p.is_dir()):
        ann = ann_root / f"{d.name}.txt"
        seqs.append(Sequence(d.name, d, ann if ann.exists() else None))
    return seqs
