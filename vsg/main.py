import signal
import numpy as np
import cv2

from config import config
from env_patch import ensure_ld_preload
from gst_pipeline import build_pipeline, gst_init_and_run
from yolo_inference import yolo_detect
from udp_sender import UDPMetaSender

# 1) LD_PRELOAD check
ensure_ld_preload()

# 2) Configurazione pipeline
WIDTH = int(config.pipeline.get("width"))
HEIGHT = int(config.pipeline.get("height"))
FPS = int(config.pipeline.get("fps"))
STREAM_IP = config.udp.get("stream_ip")
STREAM_PORT = int(config.udp.get("stream_port"))
PIPELINE = build_pipeline(WIDTH, HEIGHT, FPS, STREAM_IP, STREAM_PORT)

pipeline, appsink = gst_init_and_run(PIPELINE)
udp_sender = UDPMetaSender()

running = True
def signal_handler(sig, frame):
    global running
    print("\nShutting down...")
    running = False
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    while running:
        sample = appsink.emit("pull-sample")
        if not sample:
            continue
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_int('width').value, caps.get_int('height').value
        data = buf.extract_dup(0, buf.get_size())
        frame = np.frombuffer(data, dtype=np.uint8).reshape((h, w, 3))

        bboxes = yolo_detect(frame)
        udp_sender.send(bboxes)

        if not config.general.getboolean("debug"):
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
finally:
    print("Releasing resources...")
    pipeline.set_state(0)  # Gst.State.NULL
    udp_sender.close()
    cv2.destroyAllWindows()
    print("Exited cleanly.")
