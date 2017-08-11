import os

import scipy.io
import numpy as np

from scripts.processes.CreateLonLat import CreateLonLat
from scripts.processes.PsEstGamma import PsEstGamma
from scripts.processes.PsFiles import PsFiles
from scripts.processes.PsSelect import PsSelect
from scripts.utils.ArrayUtils import ArrayUtils
from tests.AbstractTestCase import AbstractTestCase


class TestPsSelect(AbstractTestCase):
    _GEO_DATA_FILE_NAME = 'subset_8_of_S1A_IW_SLC__1SDV_20160614T043402_20160614T043429_011702_011EEA_F130_Stack_deb_ifg_Geo.dim'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        lonlat_process = CreateLonLat(cls._PATH, cls._GEO_DATA_FILE_NAME)
        lonlat = lonlat_process.load_results()
        cls._ps_files = PsFiles(cls._PATH, lonlat_process.pscands_ij_array, lonlat)
        cls._ps_files.load_results()

        cls._est_gamma_process = None

    def test_start_process_with_matlab_data(self):
        def fill_est_gamma_with_matlab_data():
            pm1_mat = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'pm1.mat'))
            self._est_gamma_process = PsEstGamma(self._ps_files, False)
            self._est_gamma_process.coherence_bins = pm1_mat['coh_bins'][0]
            self._est_gamma_process.grid_ij = pm1_mat['grid_ij']
            self._est_gamma_process.nr_trial_wraps = pm1_mat['n_trial_wraps']
            self._est_gamma_process.ph_patch = pm1_mat['ph_patch']
            self._est_gamma_process.k_ps = pm1_mat['K_ps']
            self._est_gamma_process.c_ps = pm1_mat['C_ps']
            self._est_gamma_process.coh_ps = pm1_mat['coh_ps']
            self._est_gamma_process.n_opt = pm1_mat['N_opt']
            self._est_gamma_process.ph_res = pm1_mat['ph_res']
            self._est_gamma_process.ph_grid = pm1_mat['ph_grid']
            self._est_gamma_process.low_pass = pm1_mat['low_pass']
            self._est_gamma_process.rand_dist = pm1_mat['Nr'][0]

        fill_est_gamma_with_matlab_data()

        self._ps_select = PsSelect(self._ps_files, self._est_gamma_process)
        self._ps_select.start_process()

        select1_mat = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'select1.mat'))

        # Kuna Matlab'i indeksid hakkavad ühest siis liidame siin juurde
        np.testing.assert_allclose(np.add(self._ps_select.ifg_ind, 1), select1_mat['ifg_index'][0])
        # Reshape, et saaks võrrelda omavahel. Matlab'i massiv on kahtlane veerumassiiv
        np.testing.assert_allclose(np.add(self._ps_select.coh_thresh_ind, 1),
                                   select1_mat['ix'].reshape(len(select1_mat['ix'])))
        np.testing.assert_allclose(self._ps_select.coh_thresh, select1_mat['coh_thresh'])
        np.testing.assert_allclose(self._ps_select.ph_patch, select1_mat['ph_patch2'])
        np.testing.assert_allclose(self._ps_select.coh_ps, select1_mat['coh_ps2'])
        np.testing.assert_allclose(self._ps_select.k_ps, select1_mat['K_ps'])
        np.testing.assert_allclose(self._ps_select.ph_res, select1_mat['ph_res2'])
