import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst

def build_pipeline(width, height, fps, stream_ip, stream_port):
    PIPELINE = (
        f"nvarguscamerasrc sensor-id=0 ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
        f"tee name=t "
        f"t. ! queue ! nvvidconv ! video/x-raw,format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR,width={width},height={height} ! "
        f"appsink name=sink emit-signals=false drop=true max-buffers=1 "
        f"t. ! queue ! nvvidconv ! nvv4l2h264enc insert-sps-pps=true ! "
        f"h264parse ! rtph264pay config-interval=1 pt=96 ! "
        f"udpsink host={stream_ip} port={stream_port}"
    )
    return PIPELINE

def gst_init_and_run(pipeline_str):
    Gst.init(None)
    pipeline = Gst.parse_launch(pipeline_str)
    appsink = pipeline.get_by_name("sink")
    pipeline.set_state(Gst.State.PLAYING)
    return pipeline, appsink
