import os
import cv2
from datetime import datetime
from ultralytics import YOLO
from config import config

DEBUG = config.general.getboolean("debug")
DEBUG_DIR = config.general.get("debug_dir", "debug/debug-jpg")
MODEL_PATH = config.yolo.get("model_path")

os.makedirs(DEBUG_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

def yolo_detect(frame):
    results = model(frame, verbose=False)[0]
    bboxes = []
    for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
        if int(cls) != 0:
            continue
        x1, y1, x2, y2 = map(int, box)
        bbox = {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": float(conf)}
        bboxes.append(bbox)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
        cv2.putText(frame, f"Person {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    if DEBUG:
        last_path = os.path.join(DEBUG_DIR, "yolo_last_debug.jpg")
        cv2.imwrite(last_path, frame)

    return bboxes
