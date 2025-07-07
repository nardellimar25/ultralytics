import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst


def build_capture_pipeline(width, height, fps):
    """
    Build capture pipeline: camera -> appsink
    """
    return (
        f"nvarguscamerasrc sensor-id=0 ! "
        f"video/x-raw(memory:NVMM),width={width},height={height},framerate={fps}/1 ! "
        f"nvvidconv ! video/x-raw,format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR,width={width},height={height} ! "
        f"appsink name=src_sink emit-signals=true drop=true max-buffers=1"
    )


def build_stream_pipeline(width, height, fps, stream_ip, stream_port):
    """
    Build low-latency streaming pipeline: appsrc -> x264enc -> udpsink
    Caps included inline for immediate negotiation
    """
    return (
        f"appsrc name=dst_src is-live=true format=time do-timestamp=true caps="
        f"\"video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1\" ! "
        f"videoconvert ! video/x-raw,format=I420 ! "  # Convert to I420 for H264 encoding
        f"x264enc tune=zerolatency byte-stream=true bitrate=2000 key-int-max={fps} ! "
        f"h264parse ! rtph264pay config-interval=1 pt=96 ! "
        f"udpsink host={stream_ip} port={stream_port} sync=false async=false"
    )