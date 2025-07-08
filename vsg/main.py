import signal
import os,sys

LD_PRELOAD_VAL = "/usr/lib/aarch64-linux-gnu/libgomp.so.1:/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0"
if os.environ.get("LD_PRELOAD", "") != LD_PRELOAD_VAL:
    os.environ["LD_PRELOAD"] = LD_PRELOAD_VAL
    os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst
import numpy as np
import cv2

from config import config
from gst_pipeline import build_capture_pipeline, build_stream_pipeline
from yolo_inference import yolo_detect

# Load configuration parameters
WIDTH = int(config.pipeline.get("width"))
HEIGHT = int(config.pipeline.get("height"))
FPS = int(config.pipeline.get("fps"))
STREAM_IP = config.udp.get("stream_ip")
STREAM_PORT = int(config.udp.get("stream_port"))

# Initialize GStreamer
Gst.init(None)

# Build and start the capture pipeline: camera → appsink
capture_str = build_capture_pipeline(WIDTH, HEIGHT, FPS)
capture_pipe = Gst.parse_launch(capture_str)
src_sink = capture_pipe.get_by_name("src_sink")
capture_pipe.set_state(Gst.State.PLAYING)

# Build and start the streaming pipeline: appsrc → encoder → udpsink
stream_str = build_stream_pipeline(WIDTH, HEIGHT, FPS, STREAM_IP, STREAM_PORT)
stream_pipe = Gst.parse_launch(stream_str)
dst_src = stream_pipe.get_by_name("dst_src")
stream_pipe.set_state(Gst.State.PLAYING)

# Signal handler: stop pipelines, then exit
def immediate_exit(sig, frame):
    # Stop and release both pipelines
    capture_pipe.set_state(Gst.State.NULL)
    stream_pipe.set_state(Gst.State.NULL)
    # Destroy any OpenCV windows if open
    cv2.destroyAllWindows()
    # Exit the process immediately
    os._exit(0)

signal.signal(signal.SIGINT, immediate_exit)
signal.signal(signal.SIGTERM, immediate_exit)

# Main processing loop
while True:
    # Pull a sample from the capture pipeline (blocks until a frame is available)
    sample = src_sink.emit("pull-sample")
    if not sample:
        continue

    # Extract buffer and frame dimensions
    buf = sample.get_buffer()
    caps = sample.get_caps().get_structure(0)
    w, h = caps.get_int('width').value, caps.get_int('height').value

    # Convert buffer data to a NumPy array (BGR)
    raw_data = buf.extract_dup(0, buf.get_size())
    frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((h, w, 3))

    # Run YOLO inference and annotate the frame
    bboxes = yolo_detect(frame)
    # (Draw bounding boxes on `frame` here as needed)

    # Prepare output buffer from the annotated frame
    out_data = frame.tobytes()
    out_buf = Gst.Buffer.new_allocate(None, len(out_data), None)
    out_buf.fill(0, out_data)
    out_buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, FPS)

    # Push the buffer into the streaming pipeline
    result = dst_src.emit("push-buffer", out_buf)
    if result != Gst.FlowReturn.OK:
        print("Warning: push-buffer returned", result)