import cv2
import time
import multiprocessing
import os
import gi
import numpy as np

# Load constants from vsg_config.ini
from models.ultralytics_model.ultralytics.ultralytics.utils.vsg_config import (
    MODEL_PATH, IMG_SZ, CONF_THRESHOLD,
    UDP_IP, UDP_PORT_RAW, UDP_PORT_META,
    FRAME_WIDTH, FRAME_HEIGHT, FRAMERATE,
    DEBUG, DEBUG_DIR
)
from models.ultralytics_model.ultralytics.ultralytics.engine.vsg_capture_yolo_inference import YoloInference
from models.ultralytics_model.ultralytics.ultralytics.solutions.vsg_gstreamer import GstStreamer

# Initialize GStreamer
gi.require_version('Gst', '1.0')
from gi.repository import Gst
Gst.init(None)

# --- RAW Bayer to BGR FAST PROCESSING ---
# Calibration
black_level = 64
white_level = 4095
gain_R, gain_G, gain_B = 2.2, 1.0, 1.9
gamma_exp = 1 / 2.2

# Demosaicing method (bilinear)
DEMO_METHOD = cv2.COLOR_BAYER_GB2BGR

# Precompute gamma LUT
gamma_lut = np.array([
    ((x / 255.0) ** gamma_exp) * 255 for x in range(256)
]).clip(0, 255).astype(np.uint8)

def process_fast(raw_frame):
    # 16-bit raw input
    gray16 = raw_frame.view(np.uint16).reshape((FRAME_HEIGHT, FRAME_WIDTH))
    norm = (gray16.astype(np.float32) - black_level) / (white_level - black_level)
    norm = np.clip(norm, 0.0, 1.0)

    # White balance
    wb = np.ones_like(norm)
    wb[0::2, 0::2] = gain_G
    wb[0::2, 1::2] = gain_R
    wb[1::2, 0::2] = gain_B
    wb[1::2, 1::2] = gain_G
    balanced = norm * wb

    # Convert to 8-bit
    raw8 = (balanced * 255).astype(np.uint8)

    # Demosaic
    bgr = cv2.cvtColor(raw8, DEMO_METHOD)

    # Apply gamma
    for i in range(3):
        bgr[:, :, i] = cv2.LUT(bgr[:, :, i], gamma_lut)
    return bgr
# ----------------------------------------

class MetaStreamer:
    """
    Simple class to send metadata over UDP using the expected caps
    (application/x-meta, media=meta) for the receiver.
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


def main():
    # Initialize inference and streamers
    infer = YoloInference(MODEL_PATH, IMG_SZ, CONF_THRESHOLD)
    raw_caps = (
        f'video/x-raw,format=BGR,width={FRAME_WIDTH},'
        f'height={FRAME_HEIGHT}'
    )
    raw_sink = (
        '! videoconvert '
        '! x264enc tune=zerolatency speed-preset=superfast bitrate=500 '
        '! rtph264pay config-interval=1 pt=96 '
        f'! udpsink host={UDP_IP} port={UDP_PORT_RAW} sync=false'
    )
    meta_caps = 'application/x-meta,media=(string)meta'
    meta_sink = f'! udpsink host={UDP_IP} port={UDP_PORT_META} sync=false'

    raw_streamer = GstStreamer('raw_src', raw_caps, raw_sink, FRAMERATE)
    meta_streamer = MetaStreamer('meta_src', meta_caps, meta_sink)

    # Open camera in raw mode
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    time.sleep(0.5)

    # Debug path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    debug_path = os.path.join(project_root, DEBUG_DIR)

    try:
        while True:
            ret, raw_frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            # Convert raw Bayer -> BGR8
            frame = process_fast(raw_frame)

            # Inference
            detections = infer.run(frame)

            # DEBUG draw boxes
            if DEBUG and detections:
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

            # Stream video and metadata
            raw_streamer.push(frame.tobytes())
            meta_streamer.push(infer.pack_metadata(detections))

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        raw_streamer.stop()
        meta_streamer.stop()

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()