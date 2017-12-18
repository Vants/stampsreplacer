from configparser import ConfigParser
import os
from pathlib import Path


class ConfigUtils:
    def __init__(self, resources_path):
        PROPERTIES_FILE = os.path.join(resources_path, 'properties.ini')

        self.config = ConfigParser()
        if not Path(PROPERTIES_FILE).exists():
            raise FileNotFoundError("No properties file. Path: " + PROPERTIES_FILE)

        self.config.read(PROPERTIES_FILE)

    def get_default_section(self, key: str):
        return self.config['DEFAULT'][key]