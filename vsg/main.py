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


# Load parameters from config
WIDTH = int(config.pipeline.get("width"))
HEIGHT = int(config.pipeline.get("height"))
FPS = int(config.pipeline.get("fps"))
STREAM_IP = config.udp.get("stream_ip")
STREAM_PORT = int(config.udp.get("stream_port"))

# Initialize GStreamer
Gst.init(None)

# Create capture pipeline: camera -> appsink
capture_str = build_capture_pipeline(WIDTH, HEIGHT, FPS)
capture_pipe = Gst.parse_launch(capture_str)
src_sink = capture_pipe.get_by_name("src_sink")
capture_pipe.set_state(Gst.State.PLAYING)

# Create streaming pipeline: appsrc -> encoder -> udpsink
stream_str = build_stream_pipeline(WIDTH, HEIGHT, FPS, STREAM_IP, STREAM_PORT)
stream_pipe = Gst.parse_launch(stream_str)
dst_src = stream_pipe.get_by_name("dst_src")
stream_pipe.set_state(Gst.State.PLAYING)

# Control flag for main loop
running = True

def signal_handler(sig, frame):
    """
    Handle SIGINT/SIGTERM: exit main loop
    """
    global running
    print("\nReceived termination signal, shutting down...")
    running = False

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

try:
    while running:
        # Pull sample from capture pipeline
        sample = src_sink.emit("pull-sample")
        if not sample:
            continue

        # Extract buffer and convert to numpy frame
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        w, h = caps.get_int('width').value, caps.get_int('height').value
        raw_data = buf.extract_dup(0, buf.get_size())
        frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((h, w, 3))

        # Perform YOLO inference and annotate frame
        bboxes = yolo_detect(frame)


        # Prepare Gst.Buffer from annotated frame
        out_data = frame.tobytes()
        out_buf = Gst.Buffer.new_allocate(None, len(out_data), None)
        out_buf.fill(0, out_data)
        # Set duration for correct framerate
        out_buf.duration = Gst.util_uint64_scale_int(1, Gst.SECOND, FPS)

        # Push buffer into streaming pipeline
        result = dst_src.emit("push-buffer", out_buf)
        if result != Gst.FlowReturn.OK:
            print("Warning: push-buffer returned", result)

    # Send EOS to pipelines for a clean shutdown
    capture_pipe.send_event(Gst.Event.new_eos())
    stream_pipe.send_event(Gst.Event.new_eos())
    # Wait for EOS messages on the bus (optional)
    capture_bus = capture_pipe.get_bus()
    stream_bus = stream_pipe.get_bus()
    capture_bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.EOS)
    stream_bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, Gst.MessageType.EOS)

finally:
    # Stop pipelines and release resources
    capture_pipe.set_state(Gst.State.NULL)
    stream_pipe.set_state(Gst.State.NULL)

    cv2.destroyAllWindows()
    print("Exited cleanly.")