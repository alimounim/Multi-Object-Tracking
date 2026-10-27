"""Run detection + tracking on a video, webcam, or image folder.

Usage:
    python scripts/track.py --source path/to/video.mp4
    python scripts/track.py --config configs/default.yaml --tracker botsort.yaml
"""
import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--source", help="Overrides input.source in the config")
    parser.add_argument("--tracker", help="Overrides tracker.type in the config")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    det, trk, out = cfg["detector"], cfg["tracker"], cfg["output"]
    source = args.source if args.source is not None else cfg["input"]["source"]

    model = YOLO(det["weights"])  # downloads official weights to this path if missing

    model.track(
        source=source,
        tracker=args.tracker or trk["type"],
        conf=det["conf"],
        iou=det["iou"],
        imgsz=det["imgsz"],
        classes=det["classes"],
        device=det["device"],
        save=out["save_video"],
        show=out["show"],
        project=out["dir"],
        name="track",
        persist=True,
    )


if __name__ == "__main__":
    main()
