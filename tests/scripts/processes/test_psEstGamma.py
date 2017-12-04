import os
import scipy.io

from scripts.processes.CreateLonLat import CreateLonLat
from scripts.processes.PsEstGamma import PsEstGamma
from scripts.processes.PsFiles import PsFiles
from scripts.utils.ArrayUtils import ArrayUtils
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

        lonlat_process = CreateLonLat(cls._PATH_PATCH_FOLDER, cls._GEO_DATA_FILE_NAME)
        lonlat = lonlat_process.load_results()
        cls.ps_files = PsFiles(cls._PATH_PATCH_FOLDER, lonlat_process.pscands_ij, lonlat)
        cls.ps_files.load_results()

        # Seda kasutame teistes testides ja None'i on lihtsam kontrollida ja see protsess on natuke pikk
        cls._est_gamma_process = None

    def test_start_process_rand_dist_cached_file(self):
        self.__start_process()

        self.assertIsNotNone(self._est_gamma_process.grid_ij)
        self.assertIsNotNone(self._est_gamma_process.coherence_bins)

        pm1_mat = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'pm1.mat'))

        np.testing.assert_array_almost_equal(
            self._est_gamma_process.coherence_bins,
            pm1_mat['coh_bins'][0])

        np.testing.assert_array_equal(self._est_gamma_process.grid_ij, pm1_mat['grid_ij'])
        # math.radians tekitab selle, et täpsus on natuke vähem kui tavaliselt
        np.testing.assert_array_almost_equal(
            self._est_gamma_process.nr_trial_wraps, pm1_mat['n_trial_wraps'], 4)

        # TODO Tulevikus kontrollime juhuslikke arve. Hetkel aga kontrollime ühe eelmise vastu.
        est_gamma_process_expected = np.load(os.path.join(self._PATH, 'ps_est_gamma_save.npz'))

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

    def test_start_process_outter_rand_dist(self):
        """Anname Stamps'is loodud juhuslike arvude massiivi ette"""
        org_Nr = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'org_Nr.mat'))['Nr'][0]

        self._est_gamma_process = PsEstGamma(self.ps_files, False, org_Nr)
        self._est_gamma_process.start_process()

        self.assertIsNotNone(self._est_gamma_process.grid_ij)
        self.assertIsNotNone(self._est_gamma_process.coherence_bins)

        pm1_mat = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'pm1.mat'))

        np.testing.assert_array_almost_equal(
            self._est_gamma_process.coherence_bins,
            pm1_mat['coh_bins'][0])

        np.testing.assert_array_equal(self._est_gamma_process.grid_ij, pm1_mat['grid_ij'])
        # math.radians tekitab selle, et täpsus on natuke vähem kui tavaliselt
        np.testing.assert_array_almost_equal(
            self._est_gamma_process.nr_trial_wraps, pm1_mat['n_trial_wraps'], 4)

        np.testing.assert_array_almost_equal(self._est_gamma_process.rand_dist,
                                             pm1_mat['Nr'][0])
        np.testing.assert_array_almost_equal(self._est_gamma_process.ph_patch,
                                             pm1_mat['ph_patch'])
        np.testing.assert_array_almost_equal(self._est_gamma_process.k_ps,
                                             pm1_mat['K_ps'])
        np.testing.assert_array_almost_equal(self._est_gamma_process.c_ps,
                                             pm1_mat['C_ps'])
        np.testing.assert_array_almost_equal(self._est_gamma_process.coh_ps,
                                             pm1_mat['coh_ps'])
        np.testing.assert_array_almost_equal(self._est_gamma_process.n_opt,
                                             pm1_mat['N_opt'])
        np.testing.assert_array_almost_equal(self._est_gamma_process.ph_res,
                                             pm1_mat['ph_res'])
        np.testing.assert_array_almost_equal(self._est_gamma_process.ph_grid,
                                             pm1_mat['ph_grid'])
        np.testing.assert_array_almost_equal(self._est_gamma_process.low_pass,
                                             pm1_mat['low_pass'])

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
        np.testing.assert_array_equal(self._est_gamma_process.coherence_bins, est_gamma_loaded.coherence_bins)
        np.testing.assert_array_equal(self._est_gamma_process.nr_trial_wraps, est_gamma_loaded.nr_trial_wraps)

    def __start_process(self):
        self._est_gamma_process = PsEstGamma(self.ps_files, True)
        self._est_gamma_process.start_process()