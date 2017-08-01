from typing import Union
import os

import numpy as np
from numpy.lib.npyio import NpzFile

from scripts.utils.FolderConstants import FolderConstants
from scripts.utils.ProcessDataSaver import ProcessDataSaver


# Todo kas seda klassi saab teha dünaamilisemaks? Et tagastab kohe õiged asjad kui käss on ja kui ei ole
# vaatab funkstioonist (mis oleks callback parameetrina) dünaamiliselt mis tagastatakse
class ProcessCache:
    @staticmethod
    def get_from_cache(file_name: str, *params: str) -> NpzFile:
        save_path = os.path.join(FolderConstants.CACHE_PATH, file_name + ".npz")
        loaded = np.load(save_path)

        # Kontollime, et on ka samad asjad
        for param in params:
            if not loaded.files.__contains__(param):
                raise FileNotFoundError

        return loaded

    @staticmethod
    def save_to_cache(file_name: str, **cachable):
        ProcessDataSaver(FolderConstants.CACHE_PATH, file_name).save_data(**cachable)
