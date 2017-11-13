import os

import scipy.io
import numpy as np

from scripts.processes.CreateLonLat import CreateLonLat
from scripts.processes.PsEstGamma import PsEstGamma
from scripts.processes.PsFiles import PsFiles
from scripts.processes.PsSelect import PsSelect
from scripts.processes.PsWeed import PsWeed
from tests.AbstractTestCase import AbstractTestCase


class TestPsWeed(AbstractTestCase):
    _GEO_DATA_FILE_NAME = 'subset_8_of_S1A_IW_SLC__1SDV_20160614T043402_20160614T043429_011702_011EEA_F130_Stack_deb_ifg_Geo.dim'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        lonlat_process = CreateLonLat(cls._PATH, cls._GEO_DATA_FILE_NAME)
        lonlat = lonlat_process.load_results()
        cls.__ps_files = PsFiles(cls._PATH, lonlat_process.pscands_ij_array, lonlat)
        cls.__ps_files.load_results()

        cls.__est_gamma_process = None

        # Siin võib ps_est_gamma olla none, sest me laeme ps_select'i eelnevalt salvestatult failist
        cls.__ps_select = PsSelect(cls.__ps_files, cls.__est_gamma_process)
        cls.__ps_select.load_results()

    def test_start_process_with_matlab_data(self):
        self.__fill_est_gamma_with_matlab_data()
        self.__start_process()

    def __start_process(self):
        ps_weed_process = PsWeed(self._PATH, self.__ps_files, self.__est_gamma_process, self.__ps_select)
        ps_weed_process.start_process()

    # todo sama asi juba test_psSelect
    def __fill_est_gamma_with_matlab_data(self):
        pm1_mat = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'pm1.mat'))
        self.__est_gamma_process = PsEstGamma(self.__ps_files, False)
        self.__est_gamma_process.coherence_bins = pm1_mat['coh_bins'][0]
        self.__est_gamma_process.grid_ij = pm1_mat['grid_ij']
        self.__est_gamma_process.nr_trial_wraps = pm1_mat['n_trial_wraps']
        self.__est_gamma_process.ph_patch = pm1_mat['ph_patch']
        self.__est_gamma_process.k_ps = pm1_mat['K_ps']
        self.__est_gamma_process.c_ps = pm1_mat['C_ps']
        self.__est_gamma_process.coh_ps = pm1_mat['coh_ps']
        self.__est_gamma_process.n_opt = pm1_mat['N_opt']
        self.__est_gamma_process.ph_res = pm1_mat['ph_res']
        self.__est_gamma_process.ph_grid = pm1_mat['ph_grid']
        self.__est_gamma_process.low_pass = pm1_mat['low_pass']
        self.__est_gamma_process.rand_dist = pm1_mat['Nr'][0]