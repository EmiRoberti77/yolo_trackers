import cv2
import os
import torch
from ultralytics import YOLO
_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
_MODEL = os.path.join(_ROOT, 'model', 'yolo11l.pt') 
_VIDEO = os.path.join(_ROOT, 'video', 'highway.mp4')
_TRACKER = os.path.join(_ROOT, 'tracker', 'bytetrack.yaml')
model = YOLO(_MODEL).to(_DEVICE)
print(f"YOLO model device: {next(model.model.parameters()).device}")
print(f"Model is on CUDA: {next(model.model.parameters()).is_cuda}")
cap = cv2.VideoCapture(_VIDEO)
fps = cap.get(cv2.CAP_PROP_FPS)
fps = fps if fps and fps > 0 else 30.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
_OUT = os.path.join(_ROOT, 'tracker_bytetrack.mp4')
out = cv2.VideoWriter(_OUT, fourcc, fps, (width, height))
frame_idx = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    
    results = model.track(frame, conf=0.3, tracker=_TRACKER, persist=True, stream=True, verbose=False)
    for r in results:
        boxes = r.boxes
        for b in boxes:
            x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
            obj_name = model.names[int(b.cls)]
            conf = float(b.conf)
            tid = int(b.id.item()) if b.id is not None else -1
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f"{tid}:{obj_name}:{conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, cv2.LINE_AA)
        
    out.write(frame)
    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"{frame_idx}")
    
cap.release()
out.release()
print(f"{_OUT=}")