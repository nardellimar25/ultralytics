import socket
import json
from config import config

META_UDP_IP = config.udp.get("meta_ip")
META_UDP_PORT = int(config.udp.get("meta_port"))

class UDPMetaSender:
    def __init__(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    def send(self, bboxes):
        meta = json.dumps({"bboxes": bboxes})
        self.sock.sendto(meta.encode(), (META_UDP_IP, META_UDP_PORT))
    def close(self):
        self.sock.close()
