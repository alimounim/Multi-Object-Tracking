"""CLEAR-MOT, identity and HOTA metrics for VisDrone2019-MOT, following the official protocol:

- per class over EVAL_CLASSES, then averaged across classes
- boxes (GT and predictions) mostly inside ignored regions (category 0) are discarded before matching
- a CLEAR-MOT / identity match needs IoU >= 0.5 (HOTA averages over IoU thresholds 0.05..0.95)
"""
from pathlib import Path

import motmetrics as mm
import numpy as np
import pandas as pd

from mot.datasets.visdrone import CATEGORIES, EVAL_CLASSES, IGNORED, load_annotations
from mot.eval import hota

METRICS = [
    "mota", "idf1", "idp", "idr", "precision", "recall",
    "num_switches", "num_false_positives", "num_misses", "mostly_tracked", "mostly_lost",
]
HOTA_METRICS = ["hota", "deta", "assa", "loca"]


def intersection(a, b):
    """Pairwise intersection areas of (N, 4) and (M, 4) xywh boxes -> (N, M)."""
    a, b = a[:, None, :], b[None, :, :]
    iw = np.clip(np.minimum(a[..., 0] + a[..., 2], b[..., 0] + b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    ih = np.clip(np.minimum(a[..., 1] + a[..., 3], b[..., 1] + b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    return iw * ih


def iou(gt, pred):
    """Pairwise IoU of xywh boxes (motmetrics' iou_matrix breaks on NumPy 2)."""
    if len(gt) == 0 or len(pred) == 0:
        return np.empty((len(gt), len(pred)))
    inter = intersection(gt, pred)
    union = (gt[:, 2] * gt[:, 3])[:, None] + (pred[:, 2] * pred[:, 3])[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def ignored_overlap(boxes, regions):
    """Fraction of each box's area covered by its most-overlapping ignored region."""
    if len(boxes) == 0 or len(regions) == 0:
        return np.zeros(len(boxes))
    area = np.maximum(boxes[:, 2] * boxes[:, 3], 1e-9)
    return intersection(boxes, regions).max(axis=1) / area


def frame_pairs(gt, pred, cls, ignore_thr=0.5):
    """Per-frame (frame, gt_ids, pred_ids, iou) for one sequence and class, ignored regions removed."""
    regions = gt[gt[:, 7] == IGNORED]
    gt = gt[(gt[:, 7] == cls) & (gt[:, 6] > 0)]
    pred = pred[pred[:, 7] == cls]

    out = []
    for f in np.union1d(gt[:, 0], pred[:, 0]).astype(int):
        g = gt[gt[:, 0] == f]
        p = pred[pred[:, 0] == f]
        r = regions[regions[:, 0] == f][:, 2:6]
        g = g[ignored_overlap(g[:, 2:6], r) <= ignore_thr]
        p = p[ignored_overlap(p[:, 2:6], r) <= ignore_thr]
        out.append((f, g[:, 1].astype(int), p[:, 1].astype(int), iou(g[:, 2:6], p[:, 2:6])))
    return out


def accumulate(pairs, iou_thr=0.5):
    """Build a MOTAccumulator from frame_pairs output."""
    acc = mm.MOTAccumulator(auto_id=False)
    for f, g, p, s in pairs:
        acc.update(g, p, np.where(s >= iou_thr, 1 - s, np.nan), frameid=f)
    return acc


def evaluate(gt_dir, pred_dir, classes=EVAL_CLASSES):
    """Evaluate every sequence in gt_dir against same-named result files in pred_dir.

    Returns a DataFrame with one row per class, a 'mean' row (rates averaged over classes, counts
    summed) and an 'all' row that pools the counts of every class and sequence before computing the
    rates, as the official VisDrone MOT toolkit (evaluateBenchmark.m) does for MOTA and IDF1.
    Sequences with no result file are scored as empty predictions.
    """
    gt_dir, pred_dir = Path(gt_dir), Path(pred_dir)
    gt_files = sorted(gt_dir.glob("*.txt"))
    if not gt_files:
        raise FileNotFoundError(f"No annotation files in {gt_dir}")

    data = [(load_annotations(f), load_annotations(pred_dir / f.name)) for f in gt_files]
    mh = mm.metrics.create()
    names = [f.stem for f in gt_files]
    rows, all_accs, all_stats = {}, [], []

    def row(accs, stats, names):
        summary = mh.compute_many(accs, metrics=METRICS, names=names, generate_overall=True)
        h = hota.combine(stats)
        return pd.concat([pd.Series({k: h[k] for k in HOTA_METRICS}), summary.loc["OVERALL"]])

    for cls in classes:
        pairs = [frame_pairs(gt, pred, cls) for gt, pred in data]
        accs = [accumulate(p) for p in pairs]
        stats = [hota.sequence_stats([q[1:] for q in p]) for p in pairs]
        rows[CATEGORIES[cls]] = row(accs, stats, names)
        all_accs += accs
        all_stats += stats

    table = pd.DataFrame(rows).T
    rates = HOTA_METRICS + ["mota", "idf1", "idp", "idr", "precision", "recall"]
    table.loc["mean"] = table.sum()  # counts are summed, rates averaged
    table.loc["mean", rates] = table.loc[list(rows), rates].mean()
    table.loc["all"] = row(all_accs, all_stats, [f"{c}_{n}" for c in classes for n in names])
    return table
