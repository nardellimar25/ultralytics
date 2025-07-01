import cv2
import multiprocessing as mp
import time
import os
import gi

from models.ultralytics_model.ultralytics.ultralytics.utils.vsg_config import (
    MODEL_PATH, IMG_SZ, CONF_THRESHOLD,
    UDP_IP, UDP_PORT_RAW, UDP_PORT_META,
    FRAME_WIDTH, FRAME_HEIGHT, FRAMERATE,
    DEBUG, DEBUG_DIR
)
from models.ultralytics_model.ultralytics.ultralytics.engine.vsg_capture_yolo_inference import YoloInference
from models.ultralytics_model.ultralytics.ultralytics.solutions.vsg_gstreamer import GstStreamer

gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

class MetaStreamer:
    """
    Simple class to send metadata over UDP using GStreamer.
    """
    def __init__(self, name: str, caps: str, sink_desc: str):
        self.name = name
        pipeline_desc = (
            f'appsrc name={name} is-live=true block=true format=TIME '
            f'caps={caps} '
            f'{sink_desc}'
        )
        self.pipeline = Gst.parse_launch(pipeline_desc)
        self.appsrc = self.pipeline.get_by_name(name)
        self.pipeline.set_state(Gst.State.PLAYING)

    def push(self, data_bytes: bytes) -> None:
        buf = Gst.Buffer.new_allocate(None, len(data_bytes), None)
        buf.fill(0, data_bytes)
        self.appsrc.emit('push-buffer', buf)

    def stop(self) -> None:
        self.pipeline.set_state(Gst.State.NULL)

def frame_producer(queue, stop_event, raw_streamer, latest_frame_path):
    """
    Reads frames from the latest_frame.jpg file (updated externally)
    and puts them in the queue, while streaming frames over UDP (raw video).
    """
    last_mtime = 0
    while not stop_event.is_set():
        try:
            mtime = os.path.getmtime(latest_frame_path)
            if mtime != last_mtime:
                frame = cv2.imread(latest_frame_path)
                if frame is None:
                    time.sleep(0.01)
                    continue
                last_mtime = mtime

                # Stream the frame via UDP
                raw_streamer.push(frame.tobytes())

                # Put frame in queue for YOLO process (drop if full)
                try:
                    queue.put_nowait(frame)
                except mp.queues.Full:
                    pass  # Drop frame if consumer is too slow
            else:
                time.sleep(0.01)
        except FileNotFoundError:
            time.sleep(0.1)
            continue

    raw_streamer.stop()

def yolo_consumer(queue, stop_event, model_path, img_sz, conf_threshold,
                  meta_caps, meta_sink, debug, debug_dir):
    """
    Process function: pops frames from the queue, runs YOLO inference,
    and streams metadata over UDP.
    """
    infer = YoloInference(model_path, img_sz, conf_threshold)
    meta_streamer = MetaStreamer('meta_src', meta_caps, meta_sink)

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    debug_path = os.path.join(project_root, debug_dir)

    while not stop_event.is_set():
        try:
            frame = queue.get(timeout=0.1)
        except Exception:
            continue

        detections = infer.run(frame)

        # Optional: save debug image with bounding boxes
        if debug and detections:
            debug_img = frame.copy()
            for score, (x1, y1, x2, y2) in detections:
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    debug_img,
                    f"{score:.2f}",
                    (x1, max(y1 - 5, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1
                )
            os.makedirs(debug_path, exist_ok=True)
            cv2.imwrite(os.path.join(debug_path, 'debug.jpg'), debug_img)

        # Stream YOLO metadata
        meta_streamer.push(infer.pack_metadata(detections))

    meta_streamer.stop()

def main():
    # Define pipeline and streamer params
    raw_caps = f'video/x-raw,format=BGR,width={FRAME_WIDTH},height={FRAME_HEIGHT}'
    raw_sink = (
        '! videoconvert '
        '! x264enc tune=zerolatency speed-preset=superfast bitrate=500 '
        '! rtph264pay config-interval=1 pt=96 '
        f'! udpsink host={UDP_IP} port={UDP_PORT_RAW} sync=false'
    )
    meta_caps = 'application/x-meta,media=(string)meta'
    meta_sink = f'! udpsink host={UDP_IP} port={UDP_PORT_META} sync=false'

    # Create raw video streamer in the main process for producer
    raw_streamer = GstStreamer('raw_src', raw_caps, raw_sink, FRAMERATE)

    # Path to the shared JPEG file (mounted into the container)
    latest_frame_path = "/home/latest_frame/latest_frame.jpg"

    frame_queue = mp.Queue(maxsize=4)
    stop_event = mp.Event()

    producer_process = mp.Process(
        target=frame_producer,
        args=(frame_queue, stop_event, raw_streamer, latest_frame_path),
        daemon=True
    )

    consumer_process = mp.Process(
        target=yolo_consumer,
        args=(frame_queue, stop_event, MODEL_PATH, IMG_SZ, CONF_THRESHOLD,
              meta_caps, meta_sink, DEBUG, DEBUG_DIR),
        daemon=True
    )

    producer_process.start()
    consumer_process.start()

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        producer_process.join()
        consumer_process.join()

if __name__ == '__main__':
    mp.set_start_method('spawn') 
    main()
