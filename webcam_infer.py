"""Live YOLO inference on a webcam feed.

Draws detections with class names and confidence scores, and overlays an HUD
with FPS, inference latency and detection count.

    python webcam_infer.py                  # default: best.pt, camera 0
    python webcam_infer.py --conf 0.5       # raise the confidence threshold
    python webcam_infer.py --camera 1       # pick a different camera
"""

import argparse
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent

# BGR, one per class index (cycles if the model has more classes than this).
PALETTE = [
    (56, 168, 0),
    (0, 140, 255),
    (255, 90, 90),
    (255, 0, 200),
    (0, 215, 255),
    (180, 105, 255),
]

def draw_box(frame, xyxy, label, color):
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    # Keep the caption on screen when the box is flush against the top edge.
    top = y1 - th - base - 4
    if top < 0:
        top = y1 + 2
    cv2.rectangle(frame, (x1, top), (x1 + tw + 6, top + th + base + 4), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 3, top + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_hud(frame, lines):
    pad, line_h = 8, 20
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)[0][0] for t in lines)

    panel = frame[: pad * 2 + line_h * len(lines), : width + pad * 2].copy()
    panel[:] = (0, 0, 0)
    cv2.addWeighted(panel, 0.5, frame[: panel.shape[0], : panel.shape[1]], 0.5, 0,
                    frame[: panel.shape[0], : panel.shape[1]])

    for i, text in enumerate(lines):
        cv2.putText(
            frame,
            text,
            (pad, pad + line_h * (i + 1) - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 120),
            1,
            cv2.LINE_AA,
        )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(ROOT / "best.pt"), help="path to .pt weights")
    ap.add_argument("--camera", type=int, default=0, help="webcam index")
    ap.add_argument("--conf", type=float, default=0.05, help="confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--imgsz", type=int, default=640, help="inference image size")
    ap.add_argument("--width", type=int, default=1280, help="requested capture width")
    ap.add_argument("--height", type=int, default=720, help="requested capture height")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | 0")
    args = ap.parse_args()

    device = "cpu"
    model = YOLO(args.model)
    names = model.names

    # First call on MPS/CUDA pays a multi-second graph-build cost; absorb it here
    # so the first displayed frame isn't a stall.
    model.predict(
        np.zeros((args.imgsz, args.imgsz, 3), dtype=np.uint8),
        imgsz=args.imgsz,
        device=device,
        verbose=False,
    )

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if not cap.isOpened():
        raise SystemExit(
            f"Could not open camera {args.camera}. Check the index and that the "
            "terminal has camera permission (System Settings > Privacy & Security > Camera)."
        )

    window = "YOLO webcam  -  [q] quit  [s] snapshot"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame_times = deque(maxlen=30)  # rolling window for a stable FPS readout
    infer_times = deque(maxlen=30)
    shots = 0

    try:
        while True:
            loop_start = time.perf_counter()
            ok, frame = cap.read()
            if not ok:
                print("Dropped frame from camera, stopping.")
                break

            t0 = time.perf_counter()
            result = model.predict(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                device=device,
                verbose=False,
            )[0]
            infer_times.append((time.perf_counter() - t0) * 1000)

            boxes = result.boxes
            for xyxy, cls, conf in zip(
                boxes.xyxy.cpu().numpy(),
                boxes.cls.cpu().numpy().astype(int),
                boxes.conf.cpu().numpy(),
            ):
                draw_box(frame, xyxy, f"{names[cls]} {conf:.2f}", PALETTE[cls % len(PALETTE)])

            frame_times.append(time.perf_counter() - loop_start)
            fps = len(frame_times) / sum(frame_times)

            best = ""
            if len(boxes):
                top = int(boxes.conf.argmax())
                best = f"  top: {names[int(boxes.cls[top])]} {float(boxes.conf[top]):.2f}"

            draw_hud(
                frame,
                [
                    f"FPS {fps:5.1f}   inference {sum(infer_times) / len(infer_times):5.1f} ms",
                    f"objects {len(boxes)}{best}",
                    f"device {device}   conf>={args.conf:.2f}   imgsz {args.imgsz}",
                ],
            )

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                shots += 1
                out = ROOT / f"snapshot_{shots:03d}.jpg"
                cv2.imwrite(str(out), frame)
                print(f"Saved {out}")
    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
