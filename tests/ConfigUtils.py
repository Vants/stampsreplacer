import configparser
import os
from pathlib import Path

from tests import TEST_RESOURCES_PATH


class ConfigUtils:

    _PROPERTIES_FILE = os.path.join(TEST_RESOURCES_PATH, 'properties.ini')

    def __init__(self):
        self.config = configparser.ConfigParser()
        if not Path(self._PROPERTIES_FILE).exists():
            raise FileNotFoundError("No properties file")

        self.config.read(self._PROPERTIES_FILE)

    def get_default_section(self, key: str):
        return self.config['DEFAULT'][key]