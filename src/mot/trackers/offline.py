"""Run Ultralytics trackers on cached detections, without the detector.

Detections are cached once per (detector, split) with scripts/cache_detections.py; tracker settings can
then be swept in seconds. Cache format per sequence (.npz): frame (N,), xyxy (N, 4), conf (N,), cls (N,)
with cls already mapped to VisDrone categories.
"""
from pathlib import Path

import numpy as np
from ultralytics.trackers.track import TRACKER_MAP
from ultralytics.utils import YAML, IterableSimpleNamespace
from ultralytics.utils.checks import check_yaml


class Detections:
    """Minimal NumPy stand-in for ultralytics Boxes, as expected by tracker.update()."""

    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls
        self.xywh = np.column_stack([(xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]])

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return Detections(self.xyxy[idx], self.conf[idx], self.cls[idx])


def tracker_config(tracker="bytetrack.yaml", **overrides):
    cfg = YAML.load(check_yaml(tracker))
    cfg.update(overrides)
    return IterableSimpleNamespace(**cfg)


def load_cache(path):
    with np.load(path) as d:
        return {k: d[k] for k in ("frame", "xyxy", "conf", "cls")}


def track_cached(cache, cfg, frames, min_conf=0.0):
    """Track one cached sequence over the given frame numbers; returns VisDrone-format rows (N, 10)."""
    tracker = TRACKER_MAP[cfg.tracker_type](args=cfg)
    keep = cache["conf"] >= min_conf
    frame, xyxy, conf, cls = (cache[k][keep] for k in ("frame", "xyxy", "conf", "cls"))
    order = np.argsort(frame, kind="stable")
    frame, xyxy, conf, cls = frame[order], xyxy[order], conf[order], cls[order]

    rows = []
    for f in sorted(frames):
        s, e = np.searchsorted(frame, f, "left"), np.searchsorted(frame, f, "right")
        tracks = tracker.update(Detections(xyxy[s:e], conf[s:e], cls[s:e]))
        for x1, y1, x2, y2, tid, score, c, _ in tracks:
            rows.append((f, tid, x1, y1, x2 - x1, y2 - y1, score, c, -1, -1))
    return np.array(rows, dtype=float).reshape(-1, 10)


def write_results(rows, path):
    Path(path).write_text(
        "".join(f"{int(r[0])},{int(r[1])},{r[2]:.2f},{r[3]:.2f},{r[4]:.2f},{r[5]:.2f},{r[6]:.4f},{int(r[7])},-1,-1\n" for r in rows)
    )
