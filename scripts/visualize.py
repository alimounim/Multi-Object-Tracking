"""Render tracking results (or ground truth) for one sequence as an .mp4.

Each track ID gets a fixed color, a label and a short trail of its recent centers.

Usage:
    python scripts/visualize.py --seq data/raw/VisDrone/VisDrone2019-MOT-val/sequences/uav0000137_00458_v \
                                --results outputs/VisDrone2019-MOT-val_bytetrack/uav0000137_00458_v.txt
"""
import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from mot.utils.drawing import TrackPainter, load_by_frame


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
    by_frame = load_by_frame(args.results)

    results = Path(args.results)
    out = Path(args.out) if args.out else results.with_suffix(".mp4")
    h0, w0 = cv2.imread(str(frames[0])).shape[:2]
    scale = min(1.0, args.width / w0)
    size = (int(w0 * scale), int(h0 * scale))
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, size)
    painter = TrackPainter(w0, args.trail)

    for path in tqdm(frames, desc=results.stem):
        img = cv2.imread(str(path))
        f = int(path.stem)
        rows = by_frame.get(f, [])
        painter.draw(img, rows)
        cv2.putText(img, f"frame {f}  tracks {len(rows)}", (20, int(50 * w0 / 1920) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, painter.font * 2, (255, 255, 255), painter.thick + 1, cv2.LINE_AA)
        writer.write(cv2.resize(img, size) if scale < 1 else img)
    writer.release()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
