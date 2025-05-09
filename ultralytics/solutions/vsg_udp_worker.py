# ultralytics/solutions/vsg_udp_worker.py

import cv2
import socket
import threading
import json
import queue  # added import for queue.Empty exception

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
            # Try to get a packet; if none, retry
            try:
                data_packet = self.queue.get(timeout=0.1)
            except queue.Empty:
                continue

            # Synchronize with other workers
            try:
                self.barrier.wait()
            except threading.BrokenBarrierError:
                # Barrier broken, skip this iteration
                continue

            base_mode = self.mode.split('_')[0]

            ret = False
            jpeg = None

            if base_mode == 'crop':
                # Draw bounding boxes and send only coords as JSON
                processed = self._process_crop(data_packet)
                if processed is None:
                    continue

                # Cast all numpy.int64 to native int for JSON serialization
                raw_bboxes = data_packet.get('bboxes', [])
                clean_bboxes = [[int(coord) for coord in bbox] for bbox in raw_bboxes]

                coords_msg = json.dumps({'bboxes': clean_bboxes}).encode('utf-8')
                try:
                    self.udp_socket.sendto(coords_msg, (UDP_IP, UDP_PORT_COORDS))
                    print(f"[UDPWorker:{base_mode}] Coordinates sent to port {UDP_PORT_COORDS}.")
                except Exception as e:
                    print(f"[UDPWorker:{base_mode}] UDP send error (coords): {e}")

                # Prepare a debug JPEG of the processed frame
                ret, jpeg = cv2.imencode('.jpg', processed, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            elif base_mode == 'raw':
                # Send the raw frame as JPEG
                frame = data_packet.get('frame')
                if frame is None:
                    continue

                ret, jpeg = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                if not ret:
                    print(f"[UDPWorker:{base_mode}] JPEG encoding failed.")
                else:
                    try:
                        self.udp_socket.sendto(jpeg.tobytes(), (UDP_IP, UDP_PORT_RAW))
                        print(f"[UDPWorker:{base_mode}] Raw frame sent to port {UDP_PORT_RAW}.")
                    except Exception as e:
                        print(f"[UDPWorker:{base_mode}] UDP send error (raw): {e}")

            else:
                # Unknown mode
                print(f"[UDPWorker] Unknown mode '{self.mode}'. Skipping.")
                continue

            # Save debug JPEG if requested
            if self.debug and ret and jpeg is not None:
                try:
                    with open(self.debug_filename, 'wb') as f:
                        f.write(jpeg.tobytes())
                    print(f"[UDPWorker:{base_mode}] Debug image saved as {self.debug_filename}")
                except Exception as e:
                    print(f"[UDPWorker:{base_mode}] Error writing debug file: {e}")
