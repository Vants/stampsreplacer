import os
from pathlib import Path
import numpy as np


class ProcessDataSaver:
    def __init__(self, file_path: str, file_name: str):
        self.file_path_with_name = Path(file_path, file_name)

    def save_data(self, **data):
        np.savez(str(os.path.abspath(self.file_path_with_name)), **data)