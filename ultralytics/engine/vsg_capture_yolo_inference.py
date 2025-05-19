"""
vsg_capture_yolo_inference.py - Defines the YoloInference class to
load a YOLO model, perform inference on frames, and pack metadata
for GStreamer streaming.
"""

import struct
from ultralytics import YOLO
from typing import List, Tuple

DetectType = Tuple[float, Tuple[int, int, int, int]]

class YoloInference:
    """
    Encapsulates YOLO model loading and inference logic.
    """

    def __init__(self, model_path: str, img_size: int, conf_threshold: float):
        """
        Initialize the YOLO model.
        :param model_path: Path to the YOLO model weights.
        :param img_size: Size to which frames are resized for inference.
        :param conf_threshold: Confidence threshold for detections.
        """
        self.model = YOLO(model_path)
        self.img_size = img_size
        self.conf_threshold = conf_threshold

    def run(self, frame) -> List[DetectType]:
        """
        Run inference on a single frame and return a list of detections.
        :param frame: Input frame in BGR format.
        :return: List of tuples (confidence, (x1, y1, x2, y2)).
        """
        results = self.model.predict(
            frame, imgsz=self.img_size,
            conf=self.conf_threshold, classes=[0]
        )
        detections: List[DetectType] = []
        for r in results:
            for det in r.boxes:
                x1, y1, x2, y2 = det.xyxy[0].int().tolist()
                score = float(det.conf[0])
                detections.append((score, (x1, y1, x2, y2)))
        return detections

    @staticmethod
    def pack_metadata(detections: List[DetectType]) -> bytes:
        """
        Pack detection metadata into bytes for UDP streaming.
        Format:
        - uint16 number of detections (native endian)
        - For each detection:
          - uint8 confidence (0-255, native endian)
          - uint16 x1, y1, x2, y2 (native endian)
        :param detections: List of detection tuples.
        :return: Byte string containing packed metadata.
        """
        # Use native host byte order (little-endian on x86 and Renesas)
        buf = struct.pack('H', len(detections))
        for score, (x1, y1, x2, y2) in detections:
            c = min(int(score * 255), 255)
            buf += struct.pack('B4H', c, x1, y1, x2, y2)
        return buf
