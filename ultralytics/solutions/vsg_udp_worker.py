import cv2
import socket
import threading
import json

# UDP configuration constants
UDP_IP = "127.0.0.1"
UDP_PORT_BLURRED = 5005
UDP_PORT_RAW = 5006
UDP_PORT_COORDS = 5007  # Port to send coordinates for the crop stream

# Percentage of the bounding box height to determine the face region to blur.
FACE_REGION_PERCENT = 0.4

class UDPWorker(threading.Thread):
    """
    This class handles processing of video frames (or data packets in the case of coordinates)
    in three modes: 'crop' (coordinates only), 'blur', or 'raw'.
    It retrieves packets from a queue, synchronizes with the other workers via a barrier,
    processes the packet as required, and sends the result via UDP.
    In debug mode, it saves the processed JPEG image to disk, continuously overwriting
    the previous one so that the file always contains the latest frame.
    """
    def __init__(self, mode, queue, barrier, debug=False):
        """
        :param mode: 'crop', 'blur', or 'raw' (defines the packet processing).
        :param queue: a multithread/multiprocessing queue.
        :param barrier: a threading barrier to synchronize the workers.
        :param debug: if True, save the processed JPEG image to disk.
        """
        super().__init__(daemon=True)
        self.mode = mode
        self.queue = queue
        self.barrier = barrier
        self.debug = debug
        self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # Set the output file name for debugging.
        self.debug_filename = f"{self.mode.split('_')[0]}_stream.jpg"

    def _process_blur(self, frame, bboxes):
        """
        Process the frame in blur mode by applying a blur to each face region.
        :param frame: the full frame.
        :param bboxes: list of bounding boxes (x1, y1, x2, y2).
        :return: frame with blurred face regions.
        """
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # Calculate the height of the face region based on bounding box height.
            face_height = int((y2 - y1) * FACE_REGION_PERCENT)
            if y1 + face_height > frame.shape[0]:
                face_height = frame.shape[0] - y1
            # Extract the face region.
            face_region = frame[y1:y1 + face_height, x1:x2]
            if face_region.size != 0:
                # Apply Gaussian blur to the face region.
                blurred_face = cv2.GaussianBlur(face_region, (51, 51), 0)
                # Replace the face region with the blurred version.
                frame[y1:y1 + face_height, x1:x2] = blurred_face
        return frame

    def _process_crop(self, packet):
        """
        Process the crop packet by drawing bounding boxes on the frame.
        :param packet: dict containing 'frame' and 'bboxes'.
        :return: frame with drawn bounding boxes.
        """
        frame = packet.get('frame')
        bboxes = packet.get('bboxes', [])
        if frame is None:
            return None
        for bbox in bboxes:
            x1, y1, x2, y2 = bbox
            # Draw a green rectangle with thickness 2.
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def run(self):
        """
        Thread run method:
         - Waits for input from the queue.
         - Synchronizes with other workers using a barrier.
         - Processes the packet (coordinates or frames) and sends the result via UDP.
         - If debug mode is enabled, saves the resulting JPEG image to disk.
        """
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
                # Process the crop packet by drawing bounding boxes on the frame.
                processed_frame = self._process_crop(data_packet)
                if processed_frame is None:
                    continue
                ret, jpeg = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
                udp_port = UDP_PORT_COORDS
                # Also prepare and send JSON message with coordinates.
                coords_message = json.dumps({'bboxes': data_packet.get('bboxes', [])}).encode('utf-8')
                try:
                    self.udp_socket.sendto(coords_message, (UDP_IP, udp_port))
                    print(f"Coordinates sent via UDP to port {udp_port}.")
                except Exception as e:
                    print("UDP send error (crop):", e)
            elif base_mode == 'blur':
                # Process the blur stream.
                frame = data_packet.get('frame')
                bboxes = data_packet.get('bboxes', [])
                if frame is None:
                    continue
                processed_frame = self._process_blur(frame, bboxes)
                udp_port = UDP_PORT_BLURRED
                ret, jpeg = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
            else:
                # For the raw stream, just send the unmodified frame.
                processed_frame = data_packet.get('frame')
                if processed_frame is None:
                    continue
                udp_port = UDP_PORT_RAW
                ret, jpeg = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 50])

            if base_mode in ['blur', 'raw'] and ret:
                try:
                    self.udp_socket.sendto(jpeg.tobytes(), (UDP_IP, udp_port))
                    print(f"{base_mode.capitalize()} frame sent via UDP to port {udp_port}.")
                except Exception as e:
                    print(f"UDP send error ({base_mode}):", e)
            elif base_mode == 'crop':
                ret = True  # Already sent in crop mode.

            # Save the JPEG image to disk if in debug mode.
            if self.debug and ret:
                try:
                    with open(self.debug_filename, 'wb') as f:
                        f.write(jpeg.tobytes())
                except Exception as e:
                    print("Error writing debug file:", e)
