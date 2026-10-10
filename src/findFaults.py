"""Find validation-set mistakes: false positives, false negatives, misclassifications.

Writes:
  src/errors/errors.txt         — plain-text report
  src/errors/<image_name>.jpg   — annotated prediction for every image with errors
"""

from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent

MODEL = ROOT / "modelWeights" / "8nBest.pt"
VAL_IMAGES = REPO / "Sampled-YD-Object-Detection-2" / "valid" / "images"
VAL_LABELS = REPO / "Sampled-YD-Object-Detection-2" / "valid" / "labels"
OUT = ROOT / "errors"

CONF = 0.25        # confidence threshold for "a prediction"
IOU_MATCH = 0.5    # IoU needed to call a prediction/GT the same object


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1])
    ub = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (ua + ub - inter) if (ua + ub - inter) > 0 else 0.0


def load_gt(label_path, w, h):
    if not label_path.exists():
        return []
    out = []
    for line in label_path.read_text().splitlines():
        c, xc, yc, bw, bh = line.split()
        xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
        out.append((int(c),
                    (xc - bw / 2) * w, (yc - bh / 2) * h,
                    (xc + bw / 2) * w, (yc + bh / 2) * h))
    return out


def main():
    OUT.mkdir(exist_ok=True)
    model = YOLO(str(MODEL))
    names = model.names

    report_lines = []
    error_images = 0

    for img_path in sorted(VAL_IMAGES.glob("*.jpg")):
        r = model.predict(str(img_path), conf=CONF, verbose=False)[0]
        h, w = r.orig_shape
        gts = load_gt(VAL_LABELS / f"{img_path.stem}.txt", w, h)

        preds = [(int(c), float(cf), *xy) for xy, c, cf in zip(
            r.boxes.xyxy.cpu().numpy(),
            r.boxes.cls.cpu().numpy(),
            r.boxes.conf.cpu().numpy(),
        )]

        matched_g, matched_p, misclassified = set(), set(), []
        pairs = sorted(
            ((iou(p[2:], g[1:]), pi, gi) for pi, p in enumerate(preds) for gi, g in enumerate(gts)),
            reverse=True,
        )
        for i, pi, gi in pairs:
            if i < IOU_MATCH or pi in matched_p or gi in matched_g:
                continue
            matched_p.add(pi); matched_g.add(gi)
            if preds[pi][0] != gts[gi][0]:
                misclassified.append((pi, gi))

        fps = [p for i, p in enumerate(preds) if i not in matched_p]
        fns = [g for i, g in enumerate(gts) if i not in matched_g]

        if fps or fns or misclassified:
            error_images += 1
            report_lines.append(f"{img_path.name}")
            for pi, gi in misclassified:
                report_lines.append(f"  MISCLASS: predicted {names[preds[pi][0]]} ({preds[pi][1]:.2f}) over GT {names[gts[gi][0]]}")
            for p in fps:
                report_lines.append(f"  FP: {names[p[0]]} conf={p[1]:.2f}")
            for g in fns:
                report_lines.append(f"  FN: {names[g[0]]}")
            report_lines.append("")
            r.save(str(OUT / img_path.name))

    (OUT / "errors.txt").write_text("\n".join(report_lines))
    print(f"{error_images}/{len(list(VAL_IMAGES.glob('*.jpg')))} images had errors.")
    print(f"Report: {OUT / 'errors.txt'}")
    print(f"Annotated images: {OUT}/")


if __name__ == "__main__":
    main()