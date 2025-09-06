import cv2
import os, sys
from deep_sort_realtime.deepsort_tracker import DeepSort
from ultralytics import YOLO
from db import TrackingDB, compute_timestamp_ms
import torch
_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
_MODEL = os.path.join(_ROOT, 'model', 'yolo11n.pt') 
_VIDEO = os.path.join(_ROOT, 'video', 'highway.mp4')
print(f"{_ROOT=}")
print(f"{_MODEL=}")
print(f"{_VIDEO=}")
model = YOLO(_MODEL).to(_DEVICE)
print(f"YOLO model device: {next(model.model.parameters()).device}")
print(f"Model is on CUDA: {next(model.model.parameters()).is_cuda}")
tracker = DeepSort(max_age=5)
db = TrackingDB(os.path.join(_ROOT, 'tracking.sqlite3'))
cap = cv2.VideoCapture(_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps and fps > 0 else 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
_OUT = os.path.join(_ROOT, 'tracked_deepsort.mp4')
out = cv2.VideoWriter(_OUT, fourcc, fps, (width, height))
frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    
    result = model.predict(frame, conf=0.3, imgsz=640, verbose=False)
    detections = []
    for r in result:
        boxes = r.boxes
        for b in boxes:
            x1, y1, x2, y2 = map(float, b.xyxy[0].tolist())
            cls = int(b.cls)
            conf = float(b.conf)
            detections.append(([x1, y1, x2-x1, y2-y1], conf, cls))

    tracks = tracker.update_tracks(detections, frame=frame)
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        ltrb = track.to_ltrb()
        bbox = ( int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3]) )
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
        cv2.putText(frame, f"{track_id}", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # Log to DB
        ts_ms = compute_timestamp_ms(frame_idx, fps)
        db.log_event(
            video=_VIDEO,
            tracker="deepsort",
            model=_MODEL,
            frame_idx=frame_idx,
            timestamp_ms=ts_ms,
            track_id=int(track_id),
            class_id=None,
            conf=None,
            x1=float(bbox[0]),
            y1=float(bbox[1]),
            x2=float(bbox[2]),
            y2=float(bbox[3]),
        )

    out.write(frame)
    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"processed frames: {frame_idx}")

cap.release()
out.release()
print(f"wrote output: {_OUT}")