import configparser

class Config:
    def __init__(self, filename="config.ini"):
        self.config = configparser.ConfigParser()
        self.config.read(filename)
        self.general = self.config['General']
        self.pipeline = self.config['Pipeline']
        self.yolo = self.config['YOLO']
        self.udp = self.config['UDP']
    
    def get(self, section, key, fallback=None):
        return self.config[section].get(key, fallback)

config = Config()
