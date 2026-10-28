"""Render several trackers' results for one sequence side by side in a grid .mp4.

Each panel shows one tracker with its name, the number of tracks in the frame and the number of
distinct IDs used so far (a fast-growing count means tracks are being broken and restarted).

Usage:
    python scripts/compare_video.py --seq data/raw/VisDrone/VisDrone2019-MOT-test-dev/sequences/uav0000120_04775_v \
        --panel "ByteTrack=outputs/VisDrone2019-MOT-test-dev_yolo26n-visdrone-det_bytetrack-tuned" \
        --panel "BoT-SORT + GMC=outputs/compare/test_botsort_gmc" --out outputs/compare/uav0000120_04775_v_grid.mp4
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from mot.utils.drawing import TrackPainter, load_by_frame, put_title


def parse_panel(spec, seq_name):
    """'NAME=PATH' -> (NAME, results file); PATH may be a results folder or a .txt file."""
    name, _, path = spec.partition("=")
    path = Path(path)
    if path.is_dir():
        path = path / f"{seq_name}.txt"
    if not name or not path.is_file():
        raise SystemExit(f"Bad --panel {spec!r}: expected NAME=PATH with an existing results file ({path})")
    return name, path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", required=True, help="Folder of frames")
    parser.add_argument("--panel", action="append", required=True,
                        help="NAME=PATH (results folder or .txt file); repeat once per tracker")
    parser.add_argument("--out", required=True, help="Output .mp4")
    parser.add_argument("--cols", type=int, default=2)
    parser.add_argument("--panel-width", type=int, default=960)
    parser.add_argument("--fps", type=float, default=25)
    parser.add_argument("--trail", type=int, default=30, help="Trail length in frames")
    parser.add_argument("--max-frames", type=int, help="Stop after this many frames")
    args = parser.parse_args()

    seq = Path(args.seq)
    frames = sorted(seq.glob("*.jpg"))[:args.max_frames]
    panels = [parse_panel(p, seq.name) for p in args.panel]
    data = [load_by_frame(path) for _, path in panels]

    h0, w0 = cv2.imread(str(frames[0])).shape[:2]
    pw = args.panel_width
    ph = round(h0 * pw / w0)
    cols = min(args.cols, len(panels))
    rows = -(-len(panels) // cols)
    painters = [TrackPainter(w0, args.trail) for _ in panels]
    seen = [set() for _ in panels]
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (cols * pw, rows * ph))

    for path in tqdm(frames, desc=seq.name):
        img = cv2.imread(str(path))
        f = int(path.stem)
        grid = np.zeros((rows * ph, cols * pw, 3), np.uint8)
        for i, ((name, _), by_frame) in enumerate(zip(panels, data)):
            tracks = by_frame.get(f, [])
            seen[i].update(int(r[1]) for r in tracks)
            panel = cv2.resize(painters[i].draw(img.copy(), tracks), (pw, ph))
            put_title(panel, f"{name} | frame {f} | tracks {len(tracks)} | IDs so far {len(seen[i])}", pw / 1600)
            r, c = divmod(i, cols)
            grid[r * ph:(r + 1) * ph, c * pw:(c + 1) * pw] = panel
        writer.write(grid)
    writer.release()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
