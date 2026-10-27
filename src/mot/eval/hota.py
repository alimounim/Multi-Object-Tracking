"""HOTA (Luiten et al., IJCV 2021), following the TrackEval reference implementation.

Input per sequence: a list of frames, each (gt_ids, pred_ids, similarity) where similarity is the
(len(gt_ids), len(pred_ids)) IoU matrix. Scores are averaged over localization thresholds
alpha = 0.05, 0.10, ..., 0.95.
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

ALPHAS = np.arange(0.05, 0.99, 0.05)
EPS = np.finfo(float).eps


def _remap(frames):
    """Map arbitrary GT / predicted track IDs to contiguous 0..n-1 indices."""
    gt_map = {v: i for i, v in enumerate(np.unique(np.concatenate([f[0] for f in frames] or [[]])))}
    pr_map = {v: i for i, v in enumerate(np.unique(np.concatenate([f[1] for f in frames] or [[]])))}
    out = [
        (np.array([gt_map[v] for v in g], int), np.array([pr_map[v] for v in p], int), s)
        for g, p, s in frames
    ]
    return out, len(gt_map), len(pr_map)


def sequence_stats(frames):
    """Raw HOTA statistics for one sequence: per-alpha TP/FN/FP, AssA and LocA."""
    frames, n_gt, n_pr = _remap(frames)
    n = len(ALPHAS)
    stats = {"tp": np.zeros(n), "fn": np.zeros(n), "fp": np.zeros(n), "ass_a": np.zeros(n), "loc_a": np.zeros(n)}
    num_gt_dets = sum(len(g) for g, _, _ in frames)
    num_pr_dets = sum(len(p) for _, p, _ in frames)
    if num_gt_dets == 0 or num_pr_dets == 0:
        stats["fn"][:] = num_gt_dets
        stats["fp"][:] = num_pr_dets
        return stats

    # Global alignment score between every GT and predicted track
    potential = np.zeros((n_gt, n_pr))
    gt_count, pr_count = np.zeros((n_gt, 1)), np.zeros((1, n_pr))
    for g, p, s in frames:
        if len(g) and len(p):
            denom = s.sum(0)[None, :] + s.sum(1)[:, None] - s
            sim_iou = np.where(denom > EPS, s / np.maximum(denom, EPS), 0)
            potential[g[:, None], p[None, :]] += sim_iou
        gt_count[g] += 1
        pr_count[0, p] += 1
    global_align = potential / np.maximum(gt_count + pr_count - potential, EPS)

    matches = np.zeros((n, n_gt, n_pr))
    loc_sum = np.zeros(n)
    for g, p, s in frames:
        if len(g) == 0 or len(p) == 0:
            continue
        score = global_align[g[:, None], p[None, :]] * s
        rows, cols = linear_sum_assignment(-score)
        for a, alpha in enumerate(ALPHAS):
            ok = s[rows, cols] >= alpha - EPS
            r, c = rows[ok], cols[ok]
            stats["tp"][a] += len(r)
            matches[a, g[r], p[c]] += 1
            loc_sum[a] += s[r, c].sum()

    for a in range(n):
        tp = stats["tp"][a]
        stats["fn"][a] = num_gt_dets - tp
        stats["fp"][a] = num_pr_dets - tp
        m = matches[a]
        ass = m / np.maximum(gt_count + pr_count - m, EPS)
        stats["ass_a"][a] = (m * ass).sum() / max(1.0, tp)
        stats["loc_a"][a] = loc_sum[a] / max(1.0, tp)
    return stats


def combine(stats_list):
    """Combine sequences (TP-weighted association/localization) and compute final scores."""
    tp = sum(s["tp"] for s in stats_list)
    fn = sum(s["fn"] for s in stats_list)
    fp = sum(s["fp"] for s in stats_list)
    w = np.maximum(1.0, tp)
    ass_a = sum(s["ass_a"] * s["tp"] for s in stats_list) / w
    loc_a = sum(s["loc_a"] * s["tp"] for s in stats_list) / w
    det_a = tp / np.maximum(1.0, tp + fn + fp)
    hota = np.sqrt(det_a * ass_a)
    return {
        "hota": hota.mean(),
        "deta": det_a.mean(),
        "assa": ass_a.mean(),
        "loca": loc_a.mean(),
        "hota_alpha": hota,
    }
