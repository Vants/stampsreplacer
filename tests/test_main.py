import os
import unittest

import numpy as np

from scripts import RESOURCES_PATH
from Main import Main
from scripts.processes.CreateLonLat import CreateLonLat
from scripts.processes.PsEstGamma import PsEstGamma
from scripts.processes.PsFiles import PsFiles
from scripts.utils.internal.ConfigUtils import ConfigUtils
from scripts.utils.internal.FolderConstants import FolderConstants
from scripts.utils.internal.LoggerFactory import LoggerFactory
from tests.MetaTestCase import MetaTestCase


class TestMain(MetaTestCase):

    @classmethod
    def setUpClass(cls):
        """Load configuartion for testing."""
        super().setUpClass()

        cls._config = ConfigUtils(RESOURCES_PATH)

        cls._PATH = cls._config.get_default_section('path')

        PATCH_FOLDER = cls._config.get_default_section('patch_folder')

        cls._PATCH_1_FOLDER = os.path.join(cls._PATH, PATCH_FOLDER,
                                           FolderConstants.PATCH_FOLDER_NAME)

        cls._PATH_PATCH_FOLDER = os.path.join(cls._PATH, PATCH_FOLDER)

        cls._SAVE_LOAD_PATH = cls._config.get_default_section('save_load_path')

        cls.__logger = LoggerFactory.create("TestMain")

    @unittest.skip("Skiping whole process test")
    def test_run_whole_process(self):
        """Please be cautios with this test. This takes very long time and deletes all files in
        SAVE_LOAD_PATH directory."""

        main = Main()
        main.run()

    def test_run_only_first(self):
        self.__delete_saved_files(self._SAVE_LOAD_PATH, "lonlat_process")

        main = Main()
        main.run(0, 0)

        # geo_ref_product parameter may be empty cause we use saved data
        lonlat_process = CreateLonLat(self._PATH, "")
        lonlat_process.load_results(self._SAVE_LOAD_PATH)

        self.assert_array_not_empty(lonlat_process.lonlat)
        self.assert_array_not_empty(lonlat_process.pscands_ij)

    def test_run_start_second_stop_second(self):
        self.__delete_saved_files(self._SAVE_LOAD_PATH, "ps_files")

        main = Main()
        main.run(1, 1)

        # Let's put empty arrays here. These are not needed when loading data
        ps_files_loaded = PsFiles(self._PATH_PATCH_FOLDER, CreateLonLat(self._PATH, ""))
        ps_files_loaded.load_results(self._SAVE_LOAD_PATH)

        # Ühest kontrollist piisab küll. Täpsemad kontrollid on juba spetsiifilises klassis
        self.assert_array_not_empty(ps_files_loaded.bperp)

    def test_run_start_second_stop_third(self):
        self.__delete_saved_files(self._SAVE_LOAD_PATH, "ps_files")
        self.__delete_saved_files(self._SAVE_LOAD_PATH, "ps_est_gamma")

        main = Main()
        main.run(1, 2)

        # Let's put empty arrays here. These are not needed when loading data
        ps_files_loaded = PsFiles(self._PATH_PATCH_FOLDER, CreateLonLat(self._PATH, ""))
        ps_files_loaded.load_results(self._SAVE_LOAD_PATH)

        self.assert_array_not_empty(ps_files_loaded.bperp)

        ps_est_gamma_loaded = PsEstGamma(ps_files_loaded)
        ps_est_gamma_loaded.load_results(self._SAVE_LOAD_PATH)

        self.assert_array_not_empty(ps_est_gamma_loaded.low_pass)

    def __delete_saved_files(self, path, file_name=None):
        """This deletes all .npz files in path directory when file name is not showed."""

        self.__logger.info("Deleting .npz files in " + path)
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            file_extension = os.path.splitext(file)[-1].lower()
            try:
                if os.path.isfile(file_path) and file_extension == '.npz':
                    if file_name is not None and file == file_name:
                        os.remove(file_path)
                    elif file_name is None:
                        os.remove(file_path)
            except OSError as e:
                self.__logger.error("Error when deleting: " + str(e))
