# ============================================================
#  Real-Time Object Detection & Tracking  |  YOLOv8 + OpenCV
#  Author : Divyanshu Jaiswal  |  IIT (BHU) Varanasi
# ============================================================
#
#  HOW TO RUN:
#    python detect.py                  # webcam (default)
#    python detect.py --source video.mp4
#    python detect.py --source image.jpg
#    python detect.py --conf 0.4       # confidence threshold
#
#  CONTROLS (when window is open):
#    Q  →  quit
#    S  →  save current frame as screenshot
#    P  →  pause / resume
# ============================================================

import cv2
import argparse
import time
import os
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit("ultralytics not found. Run:  pip install ultralytics")

# ── CLI ARGUMENTS ────────────────────────────────────────────
parser = argparse.ArgumentParser(description="YOLOv8 Real-Time Object Detection & Tracking")
parser.add_argument("--source",  default="0",      help="0=webcam | path to video/image file")
parser.add_argument("--model",   default="yolov8n.pt", help="YOLOv8 model (n/s/m/l/x)")
parser.add_argument("--conf",    type=float, default=0.35, help="Confidence threshold (0-1)")
parser.add_argument("--iou",     type=float, default=0.45, help="IoU threshold for NMS")
parser.add_argument("--track",   action="store_true", default=True, help="Enable object tracking")
parser.add_argument("--save",    action="store_true", default=False, help="Save output video")
args = parser.parse_args()

# ── COLOUR PALETTE (auto-assign per class) ───────────────────
COLOURS = [
    (255, 56,  56),  (255, 157,  151), (255, 112,  31),
    (255, 178,  29), (207, 210,   49), (72,  249, 100),
    (146, 230, 161), (166, 255, 247),  (79,  195, 246),
    (47,  109, 252), (130,  59, 255),  (209,  97, 214),
]

def get_colour(class_id: int):
    return COLOURS[class_id % len(COLOURS)]

# ── OVERLAY HELPERS ──────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, label, conf, colour, track_id=None):
    """Draw bounding box + label with a filled header bar."""
    thick = max(1, int((frame.shape[0] + frame.shape[1]) / 1000))
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, thick)

    id_str  = f"#{track_id} " if track_id is not None else ""
    caption = f"{id_str}{label}  {conf:.0%}"
    (tw, th), bl = cv2.getTextSize(caption, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

    bar_y1 = max(y1 - th - 8, 0)
    cv2.rectangle(frame, (x1, bar_y1), (x1 + tw + 6, y1), colour, -1)
    cv2.putText(frame, caption, (x1 + 3, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

def draw_hud(frame, fps, n_objects, paused):
    """Top-left HUD: FPS, object count, mode."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (230, 80), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    status = "PAUSED" if paused else "LIVE"
    cv2.putText(frame, f"FPS : {fps:5.1f}",       (8, 22),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,120), 1)
    cv2.putText(frame, f"Objects : {n_objects}",  (8, 46),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,120), 1)
    cv2.putText(frame, f"Mode : {status}",        (8, 70),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,200,255), 1)

def draw_controls(frame):
    h, w = frame.shape[:2]
    hints = ["Q: Quit", "S: Screenshot", "P: Pause"]
    for i, hint in enumerate(hints):
        cv2.putText(frame, hint, (w - 150, h - 60 + i*20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

# ── MAIN PIPELINE ────────────────────────────────────────────
def run():
    print(f"\n{'='*55}")
    print(f"  YOLOv8 Object Detection & Tracking")
    print(f"  Model : {args.model}  |  Conf : {args.conf}  |  Tracking : {args.track}")
    print(f"  Source: {args.source}")
    print(f"{'='*55}\n")

    # Load model (auto-downloads on first run)
    model = YOLO(args.model)
    names = model.names  # {0: 'person', 1: 'bicycle', ...}

    # Open source
    src = 0 if args.source == "0" else args.source
    is_image = isinstance(src, str) and Path(src).suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp"}

    if is_image:
        _run_image(model, names, src)
        return

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open source: {src}")

    # Video writer (optional)
    writer = None
    if args.save:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = f"output_{ts}.mp4"
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), 25, (w, h))
        print(f"Saving output to: {out_path}")

    # State
    paused   = False
    fps_list = []
    t_prev   = time.time()

    print("Press  Q → quit | S → screenshot | P → pause\n")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("Stream ended.")
                break

            # ── INFERENCE ────────────────────────────────────
            if args.track:
                results = model.track(frame, conf=args.conf, iou=args.iou,
                                      persist=True, verbose=False)
            else:
                results = model.predict(frame, conf=args.conf, iou=args.iou,
                                        verbose=False)

            # ── DRAW DETECTIONS ───────────────────────────────
            n_objects = 0
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1,y1,x2,y2 = map(int, box.xyxy[0])
                    cls   = int(box.cls[0])
                    conf  = float(box.conf[0])
                    label = names[cls]
                    tid   = int(box.id[0]) if (args.track and box.id is not None) else None
                    colour = get_colour(cls)
                    draw_box(frame, x1, y1, x2, y2, label, conf, colour, tid)
                    n_objects += 1

            # ── FPS ───────────────────────────────────────────
            now = time.time()
            fps_list.append(1.0 / max(now - t_prev, 1e-6))
            t_prev = now
            fps = sum(fps_list[-20:]) / len(fps_list[-20:])

            draw_hud(frame, fps, n_objects, paused)
            draw_controls(frame)

            if writer:
                writer.write(frame)

        cv2.imshow("YOLOv8 — Object Detection & Tracking  |  Divyanshu Jaiswal", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("Quitting.")
            break
        elif key == ord('s'):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"screenshot_{ts}.jpg"
            cv2.imwrite(fname, frame)
            print(f"Screenshot saved: {fname}")
        elif key == ord('p'):
            paused = not paused
            print("Paused." if paused else "Resumed.")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def _run_image(model, names, path):
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Cannot read image: {path}")
    results = model.predict(frame, conf=args.conf, iou=args.iou, verbose=False)
    n = 0
    for r in results:
        for box in r.boxes:
            x1,y1,x2,y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0]); conf = float(box.conf[0])
            draw_box(frame, x1, y1, x2, y2, names[cls], conf, get_colour(cls))
            n += 1
    print(f"Detected {n} objects.")
    cv2.imshow("Detection Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
