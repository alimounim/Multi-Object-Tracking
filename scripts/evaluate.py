"""Evaluate VisDrone MOT results against ground truth.

Usage:
    python scripts/evaluate.py --gt data/raw/VisDrone/VisDrone2019-MOT-val/annotations \
                               --pred outputs/VisDrone2019-MOT-val_bytetrack
"""
import argparse
from pathlib import Path

from mot.eval.visdrone_eval import evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True, help="Folder of ground-truth annotation .txt files")
    parser.add_argument("--pred", required=True, help="Folder of result .txt files with matching names")
    args = parser.parse_args()

    table = evaluate(args.gt, args.pred)
    pct = ["mota", "idf1", "idp", "idr", "precision", "recall"]
    shown = table.copy()
    shown[pct] = shown[pct] * 100
    print(shown.round(1).to_string())
    table.to_csv(Path(args.pred) / "metrics.csv")


if __name__ == "__main__":
    main()
