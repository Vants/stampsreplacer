import os

from scripts.processes.CreateLonLat import CreateLonLat
from scripts.processes.PsEstGamma import PsEstGamma
from scripts.processes.PsFiles import PsFiles
from scripts.utils.FolderConstants import FolderConstants
from tests.AbstractTestCase import AbstractTestCase

import numpy as np


class TestPsEstGamma(AbstractTestCase):
    _TEST_RESOUCES_PATH = ''

    _GEO_DATA_FILE_NAME = 'subset_8_of_S1A_IW_SLC__1SDV_20160614T043402_20160614T043429_011702_011EEA_F130_Stack_deb_ifg_Geo.dim'
    _PLACES = 5

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls._PATCH_1_FOLDER = os.path.join(cls._PATH, FolderConstants.PATCH_FOLDER)

        lonlat_process = CreateLonLat(cls._PATH, cls._GEO_DATA_FILE_NAME)
        lonlat = lonlat_process.load_results()
        cls.ps_files = PsFiles(cls._PATH, lonlat_process.pscands_ij_array, lonlat)
        cls.ps_files.load_results()

        # Seda kasutame teistes testides ja None'i on lihtsam kontrollida ja see protsess on natuke pikk
        cls._est_gamma_process = None

    def test_start_process(self):
        self.__start_process()

        est_gamma_process_expected = np.load(os.path.join(self._PATH, 'ps_est_gamma_save.npz'))

        # TODO Tulevikus kontrollime juhuslikke arve. Hetkel aga kontrollime ühe eelmise vastu.
        np.testing.assert_array_equal(self._est_gamma_process.ph_patch,
                                      est_gamma_process_expected['ph_patch'])
        np.testing.assert_array_equal(self._est_gamma_process.k_ps,
                                      est_gamma_process_expected['k_ps'])
        np.testing.assert_array_equal(self._est_gamma_process.c_ps,
                                      est_gamma_process_expected['c_ps'])
        np.testing.assert_array_equal(self._est_gamma_process.n_opt,
                                      est_gamma_process_expected['n_opt'])
        np.testing.assert_array_equal(self._est_gamma_process.ph_res,
                                      est_gamma_process_expected['ph_res'])
        np.testing.assert_array_equal(self._est_gamma_process.ph_grid,
                                      est_gamma_process_expected['ph_grid'])
        np.testing.assert_array_equal(self._est_gamma_process.low_pass,
                                      est_gamma_process_expected['low_pass'])

    def test_save_and_load_results(self):
        self.__start_process()

        self._est_gamma_process.save_results()

        # Loome uue, mille sisse laeme salvestatud failid
        est_gamma_loaded = PsEstGamma(self.ps_files)
        est_gamma_loaded.load_results()

        np.testing.assert_array_equal(self._est_gamma_process.ph_patch, est_gamma_loaded.ph_patch)
        np.testing.assert_array_equal(self._est_gamma_process.k_ps, est_gamma_loaded.k_ps)
        np.testing.assert_array_equal(self._est_gamma_process.c_ps, est_gamma_loaded.c_ps)
        np.testing.assert_array_equal(self._est_gamma_process.n_opt, est_gamma_loaded.n_opt)
        np.testing.assert_array_equal(self._est_gamma_process.ph_res, est_gamma_loaded.ph_res)
        np.testing.assert_array_equal(self._est_gamma_process.ph_grid, est_gamma_loaded.ph_grid)
        np.testing.assert_array_equal(self._est_gamma_process.low_pass, est_gamma_loaded.low_pass)

    def __start_process(self):
        self._est_gamma_process = PsEstGamma(self.ps_files, True)
        self._est_gamma_process.start_process()
