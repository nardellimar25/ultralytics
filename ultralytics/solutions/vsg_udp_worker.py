# ultralytics/solutions/vsg_udp_worker.py

import cv2
import socket
import threading
import json

# UDP configuration constants
UDP_IP = "127.0.0.1"
UDP_PORT_RAW    = 5006
UDP_PORT_COORDS = 5007  # Port to send coordinates for the crop stream

class UDPWorker(threading.Thread):
    """
    Handles two modes: 'crop' (draw bounding boxes & send coords) and 'raw' (send JPEG frame).
    Synchronizes with other workers via a barrier and sends results over UDP.
    """
    def __init__(self, mode, queue, barrier, debug=False):
        super().__init__(daemon=True)
        self.mode = mode
        self.queue = queue
        self.barrier = barrier
        self.debug = debug
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.debug_filename = f"{self.mode.split('_')[0]}_stream.jpg"

    def _process_crop(self, packet):
        """
        Draws bounding boxes on the frame for the crop stream.
        """
        frame = packet.get('frame')
        bboxes = packet.get('bboxes', [])
        if frame is None:
            return None
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def run(self):
        while True:
            try:
                data_packet = self.queue.get(timeout=0.1)
            except Exception:
                continue

            # Synchronize with the other workers.
            try:
                self.barrier.wait()
            except threading.BrokenBarrierError:
                continue

            base_mode = self.mode.split('_')[0]

            if base_mode == 'crop':
                # Draw boxes and send coords as JSON
                processed = self._process_crop(data_packet)
                if processed is None:
                    continue
                coords_msg = json.dumps({'bboxes': data_packet.get('bboxes', [])}).encode('utf-8')
                try:
                    self.udp_socket.sendto(coords_msg, (UDP_IP, UDP_PORT_COORDS))
                    print(f"Coordinates sent via UDP to port {UDP_PORT_COORDS}.")
                except Exception as e:
                    print("UDP send error (crop):", e)
                ret = True  # we pretend we “sent” a frame for debug
                jpeg = None

            elif base_mode == 'raw':
                # Just send the unmodified frame as JPEG
                frame = data_packet.get('frame')
                if frame is None:
                    continue
                ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                try:
                    self.udp_socket.sendto(jpeg.tobytes(), (UDP_IP, UDP_PORT_RAW))
                    print(f"Raw frame sent via UDP to port {UDP_PORT_RAW}.")
                except Exception as e:
                    print("UDP send error (raw):", e)

            else:
                # Unknown mode (shouldn’t happen)
                continue

            # Save debug image if requested
            if self.debug and ret and jpeg is not None:
                try:
                    with open(self.debug_filename, 'wb') as f:
                        f.write(jpeg.tobytes())
                except Exception as e:
                    print("Error writing debug file:", e)
