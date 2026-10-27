"""CLEAR-MOT / identity metrics for VisDrone2019-MOT, following the official protocol:

- per class over EVAL_CLASSES, then averaged across classes
- boxes (GT and predictions) mostly inside ignored regions (category 0) are discarded before matching
- a match needs IoU >= 0.5
"""
from pathlib import Path

import motmetrics as mm
import numpy as np
import pandas as pd

from mot.datasets.visdrone import CATEGORIES, EVAL_CLASSES, IGNORED, load_annotations

METRICS = [
    "mota", "idf1", "idp", "idr", "precision", "recall",
    "num_switches", "num_false_positives", "num_misses", "mostly_tracked", "mostly_lost",
]


def intersection(a, b):
    """Pairwise intersection areas of (N, 4) and (M, 4) xywh boxes -> (N, M)."""
    a, b = a[:, None, :], b[None, :, :]
    iw = np.clip(np.minimum(a[..., 0] + a[..., 2], b[..., 0] + b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 1] + a[..., 3], b[..., 1] + b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    return iw * ih


def iou_distance(gt, pred, iou_thr):
    """1 - IoU, with NaN where IoU < iou_thr (motmetrics' iou_matrix breaks on NumPy 2)."""
    if len(gt) == 0 or len(pred) == 0:
        return np.empty((len(gt), len(pred)))
    inter = intersection(gt, pred)
    union = (gt[:, 2] * gt[:, 3])[:, None] + (pred[:, 2] * pred[:, 3])[None, :] - inter
    iou = inter / np.maximum(union, 1e-9)
    return np.where(iou >= iou_thr, 1 - iou, np.nan)


def ignored_overlap(boxes, regions):
    """Fraction of each box's area covered by its most-overlapping ignored region."""
    if len(boxes) == 0 or len(regions) == 0:
        return np.zeros(len(boxes))
    area = np.maximum(boxes[:, 2] * boxes[:, 3], 1e-9)
    return intersection(boxes, regions).max(axis=1) / area


def accumulate(gt, pred, cls, iou_thr=0.5, ignore_thr=0.5):
    """Build a MOTAccumulator for one sequence and one class."""
    acc = mm.MOTAccumulator(auto_id=False)
    regions = gt[gt[:, 7] == IGNORED]
    gt = gt[(gt[:, 7] == cls) & (gt[:, 6] > 0)]
    pred = pred[pred[:, 7] == cls]

    frames = np.union1d(gt[:, 0], pred[:, 0]).astype(int)
    for f in frames:
        g = gt[gt[:, 0] == f]
        p = pred[pred[:, 0] == f]
        r = regions[regions[:, 0] == f][:, 2:6]
        g = g[ignored_overlap(g[:, 2:6], r) <= ignore_thr]
        p = p[ignored_overlap(p[:, 2:6], r) <= ignore_thr]
        dist = iou_distance(g[:, 2:6], p[:, 2:6], iou_thr)
        acc.update(g[:, 1].astype(int), p[:, 1].astype(int), dist, frameid=f)
    return acc


def evaluate(gt_dir, pred_dir, classes=EVAL_CLASSES):
    """Evaluate every sequence in gt_dir against same-named result files in pred_dir.

    Returns a DataFrame with one row per class plus a 'mean' row (the official score).
    Sequences with no result file are scored as empty predictions.
    """
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    gt_files = sorted(gt_dir.glob("*.txt"))
    if not gt_files:
        raise FileNotFoundError(f"No annotation files in {gt_dir}")

    data = [(load_annotations(f), load_annotations(pred_dir / f.name)) for f in gt_files]
    mh = mm.metrics.create()
    rows = {}
    for cls in classes:
        accs = [accumulate(gt, pred, cls) for gt, pred in data]
        summary = mh.compute_many(accs, metrics=METRICS, names=[f.stem for f in gt_files], generate_overall=True)
        rows[CATEGORIES[cls]] = summary.loc["OVERALL"]

    table = pd.DataFrame(rows).T
    rates = ["mota", "idf1", "idp", "idr", "precision", "recall"]
    table.loc["mean"] = table.sum()  # counts are summed, rates averaged
    table.loc["mean", rates] = table.loc[list(rows), rates].mean()
    return table
