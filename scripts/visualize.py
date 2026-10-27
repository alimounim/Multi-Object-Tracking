"""Render tracking results (or ground truth) for one sequence as an .mp4.

Each track ID gets a fixed color, a label and a short trail of its recent centers.

Usage:
    python scripts/visualize.py --seq data/raw/VisDrone/VisDrone2019-MOT-val/sequences/uav0000137_00458_v \
                                --results outputs/VisDrone2019-MOT-val_bytetrack/uav0000137_00458_v.txt
"""
import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from mot.datasets.visdrone import CATEGORIES, IGNORED, load_annotations


def color(track_id):
    rng = np.random.default_rng(int(track_id))
    return tuple(int(c) for c in rng.integers(64, 256, 3))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True, help="Folder of frames")
    parser.add_argument("--results", required=True, help="VisDrone-format results or annotation file")
    parser.add_argument("--out", help="Output .mp4 (default: next to results)")
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--width", type=int, default=1920, help="Output width (frames are downscaled)")
    parser.add_argument("--trail", type=int, default=30, help="Trail length in frames")
    args = parser.parse_args()

    frames = sorted(Path(args.seq).glob("*.jpg"))
    data = load_annotations(args.results)
    data = data[data[:, 7] != IGNORED]
    by_frame = defaultdict(list)
    for row in data:
        by_frame[int(row[0])].append(row)

    results = Path(args.results)
    out = Path(args.out) if args.out else results.with_suffix(".mp4")
    h0, w0 = cv2.imread(str(frames[0])).shape[:2]
    scale = min(1.0, args.width / w0)
    size = (int(w0 * scale), int(h0 * scale))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
    trails = defaultdict(lambda: deque(maxlen=args.trail))
    thick = max(1, round(2 * w0 / 1920))
    font = 0.5 * w0 / 1920 + 0.2

    for path in tqdm(frames, desc=results.stem):
        img = cv2.imread(str(path))
        f = int(path.stem)
        rows = by_frame.get(f, [])
        for _, tid, x, y, w, h, _, cat, *_ in rows:
            c = color(tid)
            p1, p2 = (int(x), int(y)), (int(x + w), int(y + h))
            cv2.rectangle(img, p1, p2, c, thick)
            label = f"{int(tid)} {CATEGORIES.get(int(cat), '')}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font, 1)
            cv2.rectangle(img, (p1[0], p1[1] - th - 4), (p1[0] + tw + 2, p1[1]), c, -1)
            cv2.putText(img, label, (p1[0] + 1, p1[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, font, (0, 0, 0), 1, cv2.LINE_AA)
            trails[tid].append((int(x + w / 2), int(y + h)))
        for tid in {row[1] for row in rows}:  # only tracks visible in this frame
            if len(trails[tid]) > 1:
                cv2.polylines(img, [np.array(trails[tid])], False, color(tid), thick)
        cv2.putText(img, f"frame {f}  tracks {len(rows)}", (20, int(50 * w0 / 1920) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font * 2, (255, 255, 255), thick + 1, cv2.LINE_AA)
        writer.write(cv2.resize(img, size) if scale < 1 else img)
    writer.release()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
