"""
Load configuration from INI file using configparser.
"""
import configparser
import os

# Path to this utils folder
BASE_DIR = os.path.dirname(__file__)
CONFIG_PATH = os.path.join(BASE_DIR, 'vsg_config.ini')

parser = configparser.ConfigParser()
parser.read(CONFIG_PATH)

# MODEL
MODEL_PATH     = parser.get('MODEL', 'path')
IMG_SZ         = parser.getint('MODEL', 'img_size')
CONF_THRESHOLD = parser.getfloat('MODEL', 'conf_threshold')

# NETWORK
UDP_IP        = parser.get('NETWORK', 'udp_ip')
UDP_PORT_RAW  = parser.getint('NETWORK', 'udp_port_raw')
UDP_PORT_META = parser.getint('NETWORK', 'udp_port_meta')

# VIDEO
FRAME_WIDTH   = parser.getint('VIDEO', 'frame_width')
FRAME_HEIGHT  = parser.getint('VIDEO', 'frame_height')
FRAMERATE     = parser.getint('VIDEO', 'framerate')

# DEBUG
DEBUG       = parser.getboolean('DEBUG', 'debug', fallback=False)
DEBUG_DIR   = parser.get('DEBUG', 'debug_dir', fallback='debug_frames')
if DEBUG:
     # Create debug directory if it doesn't exist (no error if it already exists)
     os.makedirs(
         os.path.join(BASE_DIR, os.pardir, DEBUG_DIR),
         exist_ok=True
    )