"""Grid-search tracker settings on cached detections (see cache_detections.py), scored on a split.

Each setting is tracked offline and evaluated; results are written to <out>/tuning.csv sorted by HOTA.

Usage:
    python scripts/tune_tracker.py --cache outputs/cache/val_yolo26n-visdrone \
        --split data/raw/VisDrone/VisDrone2019-MOT-val --out outputs/tuning/val_bytetrack \
        --set track_high_thresh=0.2,0.3,0.4 --set new_track_thresh=0.3,0.4,0.5 --set min_conf=0.1
"""
import argparse
import itertools
import shutil
import tempfile
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from mot.datasets.visdrone import list_sequences
from mot.eval.visdrone_eval import evaluate
from mot.trackers.offline import load_cache, track_cached, tracker_config, write_results

REPORT = ["hota", "deta", "assa", "mota", "idf1", "num_switches", "num_false_positives", "num_misses"]


def parse_value(v):
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return {"true": True, "false": False}.get(v.lower(), v)


def run(seqs, cache_dir, split_dir, tracker, params):
    params = dict(params)
    min_conf = params.pop("min_conf", 0.0)
    cfg = tracker_config(tracker, **params)
    tmp = Path(tempfile.mkdtemp())
    try:
        for s in seqs:
            rows = track_cached(load_cache(cache_dir / f"{s.name}.npz"), cfg, [int(p.stem) for p in s.frames], min_conf)
            write_results(rows, tmp / f"{s.name}.txt")
        return evaluate(split_dir / "annotations", tmp).loc["mean", REPORT]
    finally:
        shutil.rmtree(tmp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True, help="Folder of cached detections (.npz per sequence)")
    parser.add_argument("--split", required=True, help="Split folder with sequences/ and annotations/")
    parser.add_argument("--tracker", default="bytetrack.yaml")
    parser.add_argument("--out", required=True)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=V1,V2",
                        help="Tracker parameter (or min_conf) and the values to try; repeatable")
    args = parser.parse_args()

    grid = {}
    for item in args.set:
        key, values = item.split("=", 1)
        grid[key] = [parse_value(v) for v in values.split(",")]
    combos = [dict(zip(grid, vals)) for vals in itertools.product(*grid.values())] or [{}]

    split = Path(args.split)
    seqs = list_sequences(split)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for params in tqdm(combos, desc="settings"):
        rows.append({**params, **run(seqs, Path(args.cache), split, args.tracker, params).to_dict()})
        table = pd.DataFrame(rows).sort_values("hota", ascending=False)
        table.to_csv(out / "tuning.csv", index=False)  # saved as it goes

    shown = table.copy()
    shown[["hota", "deta", "assa", "mota", "idf1"]] *= 100
    print(shown.round(2).head(15).to_string(index=False))


if __name__ == "__main__":
    main()
