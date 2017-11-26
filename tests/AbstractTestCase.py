import os
from unittest import TestCase

from scripts.utils.FolderConstants import FolderConstants
from tests.ConfigUtils import ConfigUtils


class AbstractTestCase(TestCase):
    """See klass on justkui abstraktne klass testidele. Siin on üldmeetodi ja setup testidele"""
    _PLACES = 5

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls._config = ConfigUtils()

        cls._PATH = cls._config.get_default_section('tests_files_path')

        PATCH_FOLDER = cls._config.get_default_section('patch_folder')

        cls._PATCH_1_FOLDER = os.path.join(cls._PATH, PATCH_FOLDER, FolderConstants.PATCH_FOLDER_NAME)

        cls._PATH_PATCH_FOLDER = os.path.join(cls._PATH, PATCH_FOLDER)