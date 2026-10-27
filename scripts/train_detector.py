"""Fine-tune a YOLO detector on the VisDrone MOT detection dataset.

Build the dataset first with scripts/prepare_yolo_dataset.py. The best checkpoint is copied
to models/<name>.pt for use with run_sequences.py (--class-map visdrone).

Usage:
    python scripts/train_detector.py
    python scripts/train_detector.py --model yolo26s.pt --batch 4 --name yolo26s_visdrone
    python scripts/train_detector.py --resume runs/train/yolo26n_visdrone/weights/last.pt
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/yolo26n.pt", help="Starting weights (COCO-pretrained)")
    parser.add_argument("--data", default="data/processed/visdrone_mot_yolo/data.yaml")
    parser.add_argument("--name", default="yolo26n_visdrone")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=8, help="8 fits YOLO26n @1280 on 8 GB")
    parser.add_argument("--patience", type=int, default=20, help="Stop after this many epochs without val improvement")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--fraction", type=float, default=1.0, help="Fraction of training images (smoke tests)")
    parser.add_argument("--resume", help="Path to last.pt to resume an interrupted run")
    args = parser.parse_args()

    if args.resume:
        model = YOLO(args.resume)
        results = model.train(resume=True)
    else:
        model = YOLO(args.model)
        results = model.train(
            data=args.data,
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            patience=args.patience,
            workers=args.workers,
            fraction=args.fraction,
            device=0,
            project=str(Path("runs/train").resolve()),
            name=args.name,
            exist_ok=True,
            plots=True,
        )

    best = Path(results.save_dir) / "weights" / "best.pt"
    target = Path("models") / f"{Path(results.save_dir).name}.pt"
    shutil.copy(best, target)
    print(f"Best weights copied to {target}")


if __name__ == "__main__":
    main()
