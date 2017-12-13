import os
from pathlib import Path

from scripts import RESOURCES_PATH
from scripts.utils.ConfigUtils import ConfigUtils


class FolderConstants:
    PATCH_FOLDER_NAME = "PATCH_1"

    __SAVE_PATH = ConfigUtils(RESOURCES_PATH).get_default_section('save_load_path')

    CACHE_PATH = os.path.join(__SAVE_PATH, "tmp")