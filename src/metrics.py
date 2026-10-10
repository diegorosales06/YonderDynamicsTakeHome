"""Run YOLO val on the project dataset and save per-class metrics to CSV."""

import csv
from pathlib import Path

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

MODEL = ROOT / "modelWeights" / "8nBest.pt"
DATA = REPO / "Sampled-YD-Object-Detection-2" / "data.yaml"
OUT = ROOT / "metrics.csv"


def main():
    m = YOLO(str(MODEL))
    r = m.val(data=str(DATA), split="val")

    names = r.names  # {0: 'bottle', 1: 'mallet'}
    class_ids = sorted(names.keys())

    rows = []
    for i, cid in enumerate(class_ids):
        rows.append({
            "class": names[cid],
            "precision": float(r.box.p[i]),
            "recall": float(r.box.r[i]),
            "f1": float(r.box.f1[i]),
            "mAP50": float(r.box.ap50[i]),
            "mAP50-95": float(r.box.maps[cid]),
        })

    rows.append({
        "class": "all",
        "precision": float(r.box.mp),
        "recall": float(r.box.mr),
        "f1": float(sum(row["f1"] for row in rows) / len(rows)),
        "mAP50": float(r.box.map50),
        "mAP50-95": float(r.box.map),
    })

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "mAP50", "mAP50-95"])
        w.writeheader()
        w.writerows(rows)

    print(f"\nWrote {OUT}")
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
