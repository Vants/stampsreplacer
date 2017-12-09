from unittest import TestCase

import os
import numpy as np

from scripts import RESOURCES_PATH
from scripts.Main import Main
from scripts.processes.CreateLonLat import CreateLonLat
from scripts.utils.ConfigUtils import ConfigUtils
from scripts.utils.FolderConstants import FolderConstants
from tests.AbstractTestCase import AbstractTestCase


class TestMain(AbstractTestCase):

    @classmethod
    def setUpClass(cls):
        """Laeme conf'i uuesti sest Main töötab päris propertes.ini peal, mitte
        testi omade peal"""
        super().setUpClass()

        cls._config = ConfigUtils(RESOURCES_PATH)

        cls._PATH = cls._config.get_default_section('path')

        PATCH_FOLDER = cls._config.get_default_section('patch_folder')

        cls._PATCH_1_FOLDER = os.path.join(cls._PATH, PATCH_FOLDER, FolderConstants.PATCH_FOLDER_NAME)

        cls._PATH_PATCH_FOLDER = os.path.join(cls._PATH, PATCH_FOLDER)

        cls._SAVE_LOAD_PATH = cls._config.get_default_section('save_load_path')

    def test_run(self):
        main = Main()
        main.run()

    def test_run_only_first(self):
        main = Main()
        main.run(0, 0)

        lonlat_process = CreateLonLat(self._PATH_PATCH_FOLDER, "")

        lonlat = lonlat_process.load_results(self._SAVE_LOAD_PATH)

        self.assertNotEqual(lonlat.size, 0)
        self.assertNotEqual(lonlat_process.pscands_ij.size, 0)
