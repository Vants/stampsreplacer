from unittest import TestCase

import os
import scipy.io

from scripts.processes.CreateLonLat import CreateLonLat
from scripts.processes.PhaseCorrection import PhaseCorrection
from scripts.processes.PsEstGamma import PsEstGamma
from scripts.processes.PsFiles import PsFiles
from scripts.processes.PsSelect import PsSelect
from scripts.processes.PsWeed import PsWeed
from tests.AbstractTestCase import AbstractTestCase

import numpy as np


class TestPhaseCorrection(AbstractTestCase):
    _GEO_DATA_FILE_NAME = 'subset_8_of_S1A_IW_SLC__1SDV_20160614T043402_20160614T043429_011702_011EEA_F130_Stack_deb_ifg_Geo.dim'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        lonlat_process = CreateLonLat(cls._PATH, cls._GEO_DATA_FILE_NAME)
        lonlat = lonlat_process.load_results()
        cls.__ps_files = PsFiles(cls._PATH, lonlat_process.pscands_ij, lonlat)
        cls.__ps_files.load_results()

        cls.__ps_est_gamma = PsEstGamma(cls.__ps_files)

        self = TestPhaseCorrection() # Selleks, et saaks asju väljapool @classmethod kasutada
        self.__fill_est_gamma_with_matlab_data()

        # Siin võib ps_est_gamma olla none, sest me laeme ps_select'i eelnevalt salvestatult failist
        cls.__ps_select = PsSelect(cls.__ps_files, cls.__ps_est_gamma)
        cls.__ps_select.load_results()

        cls.__ps_weed = PsWeed(cls._PATH, cls.__ps_files, cls.__ps_est_gamma, cls.__ps_select)
        cls.__ps_weed.load_results()

        cls.__phase_correction = None

    def test_start_process(self):
        self.__start_process()

        rc_mat = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'rc2.mat'))

        np.testing.assert_allclose(self.__phase_correction.ph_rc, rc_mat['ph_rc'], atol=0.05)
        np.testing.assert_array_almost_equal(self.__phase_correction.ph_reref, rc_mat['ph_reref'])

    def test_save_and_load_results(self):
        self.__start_process()
        self.__phase_correction.save_results()

        phase_correction_loaded = PhaseCorrection(self.__ps_files, self.__ps_est_gamma,
                                                  self.__ps_weed, self.__ps_select)
        phase_correction_loaded.load_results()

        np.testing.assert_array_almost_equal(self.__phase_correction.ph_rc,
                                             phase_correction_loaded.ph_rc)
        np.testing.assert_array_almost_equal(self.__phase_correction.ph_reref,
                                             phase_correction_loaded.ph_reref)

    def __start_process(self):
        self.__phase_correction = PhaseCorrection(self.__ps_files, self.__ps_est_gamma,
                                                  self.__ps_weed, self.__ps_select)
        self.__phase_correction.start_process()

    # todo sama asi juba test_psSelect
    def __fill_est_gamma_with_matlab_data(self):
        pm1_mat = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'pm1.mat'))
        self.__ps_est_gamma.coherence_bins = pm1_mat['coh_bins'][0]
        self.__ps_est_gamma.grid_ij = pm1_mat['grid_ij']
        self.__ps_est_gamma.nr_trial_wraps = pm1_mat['n_trial_wraps']
        self.__ps_est_gamma.ph_patch = pm1_mat['ph_patch']
        self.__ps_est_gamma.k_ps = pm1_mat['K_ps']
        self.__ps_est_gamma.c_ps = pm1_mat['C_ps']
        self.__ps_est_gamma.coh_ps = pm1_mat['coh_ps']
        self.__ps_est_gamma.n_opt = pm1_mat['N_opt']
        self.__ps_est_gamma.ph_res = pm1_mat['ph_res']
        self.__ps_est_gamma.ph_grid = pm1_mat['ph_grid']
        self.__ps_est_gamma.low_pass = pm1_mat['low_pass']
        self.__ps_est_gamma.rand_dist = pm1_mat['Nr'][0]
