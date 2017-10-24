import os
from pathlib import Path


class FolderConstants:
    PATCH_FOLDER_NAME = "PATCH_1"

    SAVE_FOLDER = "process_saves"
    SAVE_PATH = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'tests', "resources",
                     SAVE_FOLDER))

    CACHE_PATH = os.path.join(SAVE_PATH, "tmp")