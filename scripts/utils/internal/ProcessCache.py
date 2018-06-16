import os

import numpy as np
from numpy.lib.npyio import NpzFile

from scripts.utils.internal.FolderConstants import FolderConstants
from scripts.utils.internal.ProcessDataSaver import ProcessDataSaver

# todo Is it possible to make this mode dynamical? Return correct things when there are cached
# parameters to be returned (callback maybe?)
class ProcessCache:
    @staticmethod
    def get_from_cache(file_name: str, *params: str) -> NpzFile:
        save_path = os.path.join(FolderConstants.CACHE_PATH, file_name + ".npz")
        loaded = np.load(save_path)

        # Check if it is same file by checking if parameters exist
        for param in params:
            if not loaded.files.__contains__(param):
                raise FileNotFoundError

        return loaded

    @staticmethod
    def save_to_cache(file_name: str, **cachable):
        ProcessDataSaver(FolderConstants.CACHE_PATH, file_name).save_data(**cachable)
