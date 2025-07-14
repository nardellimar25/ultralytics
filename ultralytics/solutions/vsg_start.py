"""
vsg_sender.py - Main script to capture webcam frames,
perform YOLO inference, and stream raw frames with bounding-box metadata over UDP.
"""

import cv2
import time
import multiprocessing
import os
import gi
import statistics

# Load enum from test_bench.py
from models.ultralytics_model.ultralytics.ultralytics.solutions.test_bench import Runs

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


class MetaStreamer:
    """
    Simple class to send metadata over UDP using the expected caps
    (application/x-meta, media=meta) for the receiver.
    """
    def __init__(self, name: str, caps: str, sink_desc: str):
        self.name = name
        # Create an appsrc element without frame rate (metadata only)
        pipeline_desc = (
            f'appsrc name={name} is-live=true block=true format=TIME '
            f'caps={caps} '
            f'{sink_desc}'
        )
        self.pipeline = Gst.parse_launch(pipeline_desc)
        self.appsrc = self.pipeline.get_by_name(name)
        self.pipeline.set_state(Gst.State.PLAYING)

    def push(self, data_bytes: bytes) -> None:
        """
        Push raw bytes into the pipeline (no PTS/duration needed for metadata).
        """
        buf = Gst.Buffer.new_allocate(None, len(data_bytes), None)
        buf.fill(0, data_bytes)
        self.appsrc.emit('push-buffer', buf)

    def stop(self) -> None:
        """
        Cleanly stop the GStreamer pipeline.
        """
        self.pipeline.set_state(Gst.State.NULL)



def main():
    """
    Initialize the YOLO inference engine and GStreamer streamers,
    then continuously capture frames, perform inference, and send data.
    """
    # Create YOLO inference instance
    infer = YoloInference(MODEL_PATH, IMG_SZ, CONF_THRESHOLD)
    print("\nMODEL PATH: ",MODEL_PATH,"\n")
    print("IMG_SZ: ",IMG_SZ)
    # Define GStreamer caps and sink for raw video frames
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

    # Define caps and sink for metadata
    meta_caps = 'application/x-meta,media=(string)meta'
    meta_sink = f'! udpsink host={UDP_IP} port={UDP_PORT_META} sync=false'

    # Instantiate streamers
    raw_streamer = GstStreamer('raw_src', raw_caps, raw_sink, FRAMERATE)
    meta_streamer = MetaStreamer('meta_src', meta_caps, meta_sink)

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    time.sleep(0.5)  # Allow camera sensor to stabilize

    # Setup debug output path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    debug_path = os.path.join(project_root, DEBUG_DIR)

    # Init variables for test benching times and mem usage
    capread_times = []
    YOLOinfer_times = []
    count = 0
    runs = [Runs.SHORT.value, Runs.MED.value, Runs.LONG.value]
    

    try:
        while True:
            st_time = time.time()
            ret, frame = cap.read()
            end_time = time.time()
            if not ret:
                time.sleep(0.01)
                continue
            capread_time = end_time - st_time
            
            # Perform YOLO inference
            st_time = time.time()
            detections = infer.run(frame)
            end_time = time.time()
            YOLOinfer_time = end_time - st_time
            
            #print("\nTime for cap.read(): ", capread_time,"s")
            #print("\nTime for YOLO inference: ", YOLOinfer_time,"s")
            
            # DEBUG: draw bounding boxes and save an image
            # DEBUG: save timers for cap.read and yolo inference 
            if DEBUG and detections:
                count += 1
                if(count < runs[-1]):
                    capread_times.append(capread_time)
                    YOLOinfer_times.append(YOLOinfer_time)
                else:
                    print("\ndone!!\n")

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
                debug_file = os.path.join(debug_path, 'debug.jpg')
                cv2.imwrite(debug_file, debug_img)

            # Stream raw frame and metadata
            raw_streamer.push(frame.tobytes())
            meta_streamer.push(infer.pack_metadata(detections))

    except KeyboardInterrupt:
        # Exit cleanly on Ctrl+C
        pass
    finally:
        # DEBUG: print mean time and standard deviation for cap.read and yolo inference, in different run sizes
        if DEBUG:
            capread_times.pop(0)    #discard first time measured
            YOLOinfer_times.pop(0)    #discard first time measured
            for run in runs:
                print("[ RUNS:",run,"]\nMean time cap.read(): ", statistics.mean(capread_times[:run]),"s")
                print("[ RUNS:",run,"]\nStandard deviation for cap.read(): ", statistics.stdev(capread_times[:run]))
                print("[ RUNS:",run,"]\nMean time YOLO inference: ", statistics.mean(YOLOinfer_times[:run]),"s")
                print("[ RUNS:",run,"]\nStandard deviation for YOLO inference: ", statistics.stdev(YOLOinfer_times[:run]))
        cap.release()
        raw_streamer.stop()
        meta_streamer.stop()


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
