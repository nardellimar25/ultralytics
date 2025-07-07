import os, sys
from config import config

def ensure_ld_preload():
    val = config.get('Env', 'ld_preload')
    if os.environ.get("LD_PRELOAD", "") != val:
        os.environ["LD_PRELOAD"] = val
        os.execve(sys.executable, [sys.executable] + sys.argv, os.environ)
