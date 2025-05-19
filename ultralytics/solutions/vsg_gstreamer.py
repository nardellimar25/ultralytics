#vsg_gstreamer.py
"""
vsg_gstreamer.py - Provides GstStreamer class to
build and manage GStreamer pipelines for raw frame
and metadata streaming over UDP.
"""
import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst

# Initialize GStreamer once
Gst.init(None)

class GstStreamer:
    """
    Wrapper for a GStreamer pipeline with an appsrc feeding an udpsink.
    """
    def __init__(self, name: str, caps: str, sink_desc: str, framerate: int):
        self.name = name
        pipeline_desc = (
            f'appsrc name={name} is-live=true block=true format=TIME '
            f'caps={caps},framerate={framerate}/1 '
            f'{sink_desc}'
        )
        self.pipeline = Gst.parse_launch(pipeline_desc)
        self.appsrc = self.pipeline.get_by_name(name)
        self.pipeline.set_state(Gst.State.PLAYING)
        self._frame_idx = 0
        self._framerate = framerate

    def push(self, data_bytes: bytes) -> None:
        """
        Push raw byte data into the appsrc element,
        attaching PTS and duration for proper streaming.
        """
        buf = Gst.Buffer.new_allocate(None, len(data_bytes), None)
        buf.fill(0, data_bytes)
        buf.pts = Gst.util_uint64_scale(self._frame_idx, Gst.SECOND, self._framerate)
        buf.duration = Gst.SECOND // self._framerate
        self.appsrc.emit('push-buffer', buf)
        self._frame_idx += 1

    def stop(self) -> None:
        """
        Stop the GStreamer pipeline cleanly.
        """
        self.pipeline.set_state(Gst.State.NULL)