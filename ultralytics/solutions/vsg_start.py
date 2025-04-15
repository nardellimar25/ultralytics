import multiprocessing
import threading
import time
from models.ultralytics_model.ultralytics.ultralytics.engine.vsg_capture_yolo_inference import CaptureInferenceProcess
from models.ultralytics_model.ultralytics.ultralytics.engine.vsg_detection_handler import process_detections
from models.ultralytics_model.ultralytics.ultralytics.solutions.vsg_udp_worker import UDPWorker

DEBUG = False  # Global flag to enable debugging mode (save latest frame to file)

def create_global_pipeline():
    """
    Create a global pipeline containing three queues (for crop, blurred, and raw streams),
    three UDPWorker threads, and a barrier to synchronize them.
    """
    # Create three queues.
    crop_queue = multiprocessing.Queue(maxsize=10)   # For crop (bounding boxes + frame)
    blur_queue = multiprocessing.Queue(maxsize=10)   # For blurred frame
    raw_queue = multiprocessing.Queue(maxsize=10)    # For raw frame

    # Create a barrier for three workers.
    send_barrier = threading.Barrier(parties=3)

    # Start UDP worker threads with debug mode enabled.
    crop_worker = UDPWorker('crop', crop_queue, send_barrier, debug=DEBUG)
    blur_worker = UDPWorker('blur', blur_queue, send_barrier, debug=DEBUG)
    raw_worker = UDPWorker('raw', raw_queue, send_barrier, debug=DEBUG)

    crop_worker.start()
    blur_worker.start()
    raw_worker.start()

    return {
        'queues': {
            'crop': crop_queue,
            'blur': blur_queue,
            'raw': raw_queue
        },
        'workers': {
            'crop': crop_worker,
            'blur': blur_worker,
            'raw': raw_worker
        },
        'barrier': send_barrier
    }

def main():
    # Create the global pipeline.
    pipeline = create_global_pipeline()

    # Start the capture process, passing the pipeline.
    capture_process = CaptureInferenceProcess(
        detection_callback=lambda frame, results, pipe: process_detections(frame, results, pipe),
        pipeline=pipeline
    )
    capture_process.start()

    print("System running. Latest frames are saved to disk as 'crop_stream.jpg', 'blur_stream.jpg', and 'raw_stream.jpg'.")
    print("Use your preferred image viewer to check the latest frame for each stream.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Terminating system...")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
