#vsg_sender.py
"""
vsg_sender.py - Main script to capture webcam frames,
perform YOLO inference, and stream raw frames and
bounding-box metadata via GStreamer pipelines.
"""
import cv2
import time
import multiprocessing

from models.ultralytics_model.ultralytics.ultralytics.utils.vsg_config import (
    MODEL_PATH, IMG_SZ, CONF_THRESHOLD,
    UDP_IP, UDP_PORT_RAW, UDP_PORT_META,
    FRAME_WIDTH, FRAME_HEIGHT, FRAMERATE
)
from  models.ultralytics_model.ultralytics.ultralytics.engine.inference import YoloInference
from  models.ultralytics_model.ultralytics.ultralytics.solutions.vsg_gstreamer import GstStreamer


def main():
    """
    Initialize inference engine and GStreamer streamers,
    then loop capturing and streaming frames and metadata.
    """
    infer = YoloInference(MODEL_PATH, IMG_SZ, CONF_THRESHOLD)

    raw_caps  = (
        f'video/x-raw,format=BGR,width={FRAME_WIDTH},'
        f'height={FRAME_HEIGHT}'
    )
    raw_sink  = (
        '! videoconvert '
        '! x264enc tune=zerolatency speed-preset=superfast bitrate=500 '
        '! rtph264pay config-interval=1 pt=96 '
        f'! udpsink host={UDP_IP} port={UDP_PORT_RAW} sync=false'
    )
    meta_caps = 'application/x-meta'
    meta_sink = f'! udpsink host={UDP_IP} port={UDP_PORT_META} sync=false'

    raw_streamer  = GstStreamer('raw_src', raw_caps, raw_sink, FRAMERATE)
    meta_streamer = GstStreamer('meta_src', meta_caps, meta_sink, FRAMERATE)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    time.sleep(0.5)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            dets = infer.run(frame)
            raw_streamer.push(frame.tobytes())
            meta_streamer.push(infer.pack_metadata(dets))

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        raw_streamer.stop()
        meta_streamer.stop()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
