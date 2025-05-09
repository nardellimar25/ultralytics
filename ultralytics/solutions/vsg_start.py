# main.py

import multiprocessing
import threading
import time

from models.ultralytics_model.ultralytics.ultralytics.engine.vsg_capture_yolo_inference import CaptureInferenceProcess
from models.ultralytics_model.ultralytics.ultralytics.engine.vsg_detection_handler import process_detections
from models.ultralytics_model.ultralytics.ultralytics.solutions.vsg_udp_worker import UDPWorker


DEBUG = False  # Global flag to enable saving the latest frame to disk

def create_global_pipeline():
    """
    Builds a pipeline with two queues (crop and raw), two UDPWorker threads,
    and a barrier to synchronize them.
    """
    crop_queue = multiprocessing.Queue(maxsize=10)  # for bounding‐box+coords
    raw_queue  = multiprocessing.Queue(maxsize=10)  # for raw frames

    # Create a barrier for two workers: crop and raw
    send_barrier = threading.Barrier(parties=2)

    crop_worker = UDPWorker('crop', crop_queue, send_barrier, debug=DEBUG)
    raw_worker  = UDPWorker('raw',  raw_queue,  send_barrier, debug=DEBUG)

    crop_worker.start()
    raw_worker.start()

    return {
        'queues': {
            'crop': crop_queue,
            'raw':  raw_queue
        },
        'workers': {
            'crop': crop_worker,
            'raw':  raw_worker
        },
        'barrier': send_barrier
    }

def main():
    pipeline = create_global_pipeline()

    capture_process = CaptureInferenceProcess(
        detection_callback=lambda frame, results, pipe: process_detections(frame, results, pipe),
        pipeline=pipeline
    )
    capture_process.start()

    print("System running. Latest frames are saved as 'crop_stream.jpg' and 'raw_stream.jpg'.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating system...")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
