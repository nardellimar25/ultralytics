import cv2
import torch
import time
import multiprocessing
from ultralytics import YOLO

# YOLO model configuration constants
MODEL_PATH = "yolov8n.pt"
IMG_SZ = 128
CONF_THRESHOLD = 0.5

class CaptureInferenceProcess(multiprocessing.Process):
    """
    Process to capture video frames from a webcam, perform YOLO inference,
    and call a callback with the results.
    """
    def __init__(self, detection_callback, pipeline):
        """
        :param detection_callback: callable that takes (frame, detection_results, pipeline)
        :param pipeline: global pipeline dictionary containing queues.
        """
        super().__init__(daemon=True)
        self.detection_callback = detection_callback
        self.pipeline = pipeline
        self.model = YOLO(MODEL_PATH)

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open webcam.")
            return

        # Set capture resolution.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        time.sleep(0.5)  # Warm-up time for camera

        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            # Perform YOLO inference on the captured frame.
            with torch.no_grad():
                results = self.model.predict(frame, imgsz=IMG_SZ, conf=CONF_THRESHOLD, classes=[0])

            # Call the detection callback with the frame and inference results.
            self.detection_callback(frame.copy(), results, self.pipeline)

        cap.release()
