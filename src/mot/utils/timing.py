"""Per-frame speed measurement for the detect + track pipeline.

Stages per frame (milliseconds):
    detect  = Ultralytics' preprocess + inference + postprocess (CUDA-synchronized)
    track   = tracker.update(), timed by wrapping the tracker classes
    total   = wall-clock time between consecutive frames from the stream, i.e. everything
              including image loading and result writing
    other   = total - detect - track (mostly image decoding / loading)
"""
import time
from contextlib import contextmanager

import numpy as np
import torch
from ultralytics.trackers.track import TRACKER_MAP

STAGES = ("total", "detect", "track", "other")


@contextmanager
def time_tracker_updates(store):
    """Append the duration (ms) of every tracker.update() call to `store` while active."""
    originals = {cls: cls.update for cls in set(TRACKER_MAP.values())}

    def wrap(fn):
        def timed(self, *args, **kwargs):
            t = time.perf_counter()
            out = fn(self, *args, **kwargs)
            store.append((time.perf_counter() - t) * 1000)
            return out
        return timed

    for cls, fn in originals.items():
        cls.update = wrap(fn)
    try:
        yield store
    finally:
        for cls, fn in originals.items():
            cls.update = fn


class FrameTimer:
    """Collects per-frame stage times; the first `warmup` frames of each sequence are dropped."""

    def __init__(self, warmup=10):
        self.warmup = warmup
        self.times = {s: [] for s in STAGES}

    def add_sequence(self, total, detect, track):
        total, detect, track = (np.asarray(x[self.warmup:], float) for x in (total, detect, track))
        self.times["total"].extend(total)
        self.times["detect"].extend(detect)
        self.times["track"].extend(track)
        self.times["other"].extend(total - detect - track)

    def summary(self):
        out = {"frames_timed": len(self.times["total"])}
        for s in STAGES:
            out[f"{s}_ms"] = float(np.mean(self.times[s])) if self.times[s] else float("nan")
        out["fps"] = 1000 / out["total_ms"]
        out["fps_model_only"] = 1000 / (out["detect_ms"] + out["track_ms"])
        out["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
        return out
