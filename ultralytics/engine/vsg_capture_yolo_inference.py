#_vsg_capture_yolo_inference.py
"""
inference.py - Defines YoloInference class to
encapsulate YOLO model loading, frame inference,
and metadata packing for GStreamer streaming.
"""
import struct
from ultralytics import YOLO
from typing import List, Tuple

detect_t = Tuple[float, Tuple[int, int, int, int]]

class YoloInference:
    """
    Encapsulates YOLO model loading and inference logic.
    """
    def __init__(self, model_path: str, img_size: int, conf_thresh: float):
        self.model = YOLO(model_path)
        self.img_size = img_size
        self.conf_thresh = conf_thresh

    def run(self, frame) -> List[detect_t]:
        """
        Run inference on a single frame and return list of
        (confidence, (x1, y1, x2, y2)) detections.
        """
        results = self.model.predict(
            frame, imgsz=self.img_size,
            conf=self.conf_thresh, classes=[0]
        )
        detections = []
        for r in results:
            for det in r.boxes:
                x1, y1, x2, y2 = det.xyxy[0].int().tolist()
                score = float(det.conf[0])
                detections.append((score, (x1, y1, x2, y2)))
        return detections

    @staticmethod
    def pack_metadata(detections: List[detect_t]) -> bytes:
        """
        Pack metadata into bytes for UDP streaming:
        uint16 number of detections,
        followed by repeating uint8(conf)*4 uint16(x1,y1,x2,y2).
        """
        buf = struct.pack('>H', len(detections))
        for score, (x1, y1, x2, y2) in detections:
            c = min(int(score * 255), 255)
            buf += struct.pack('>B4H', c, x1, y1, x2, y2)
        return buf
