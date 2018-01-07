import os
from unittest import TestCase
import numpy as np

from scripts.utils.internal.FolderConstants import FolderConstants
from scripts.utils.internal.ConfigUtils import ConfigUtils
from tests import TEST_RESOURCES_PATH


class MetaTestCase(TestCase):
    """See klass on justkui abstraktne klass testidele. Siin on üldmeetodi ja setup testidele"""
    _PLACES = 5

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls._config = ConfigUtils(TEST_RESOURCES_PATH)

        cls._PATH = cls._config.get_default_section('tests_files_path')

        PATCH_FOLDER = cls._config.get_default_section('patch_folder')

        cls._PATCH_1_FOLDER = os.path.join(cls._PATH, PATCH_FOLDER, FolderConstants.PATCH_FOLDER_NAME)

        cls._PATH_PATCH_FOLDER = os.path.join(cls._PATH, PATCH_FOLDER)

        cls._SAVE_LOAD_PATH = cls._config.get_default_section('save_load_path')

    def assert_array_not_empty(self, array: np.ndarray):
        self.assertNotEqual(array.size, 0)