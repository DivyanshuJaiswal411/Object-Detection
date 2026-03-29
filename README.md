# Real-Time Object Detection & Tracking
**YOLOv8 + OpenCV | Divyanshu Jaiswal | IIT (BHU) Varanasi**

## What It Does
- Detects 80 object classes in real-time from webcam, video file, or image
- Tracks objects across frames with unique IDs (multi-object tracking)
- Displays FPS, object count, and bounding boxes with confidence scores
- Saves screenshots or full output video

## Setup
```bash
pip install ultralytics opencv-python
```

## Run
```bash
# Webcam (default)
python detect.py

# Video file
python detect.py --source myvideo.mp4

# Image
python detect.py --source photo.jpg

# Higher accuracy model (slower)
python detect.py --model yolov8s.pt

# Custom confidence threshold
python detect.py --conf 0.5

# Save output video
python detect.py --save
```

## Controls
| Key | Action |
|-----|--------|
| Q | Quit |
| S | Save screenshot |
| P | Pause / Resume |

## Models Available
| Model | Speed | Accuracy |
|-------|-------|----------|
| yolov8n.pt | Fastest | Good |
| yolov8s.pt | Fast | Better |
| yolov8m.pt | Medium | Great |
| yolov8l.pt | Slow | Excellent |

Models auto-download from Ultralytics on first run.

## Tech Stack
- **YOLOv8** (Ultralytics) — state-of-the-art object detection
- **OpenCV** — frame capture, preprocessing, visualisation
- **Python argparse** — clean CLI interface
