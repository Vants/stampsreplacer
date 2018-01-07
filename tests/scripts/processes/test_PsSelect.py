import os

import scipy.io
import numpy as np

from scripts.processes.CreateLonLat import CreateLonLat
from scripts.processes.PsEstGamma import PsEstGamma
from scripts.processes.PsFiles import PsFiles
from scripts.processes.PsSelect import PsSelect
from scripts.utils.ArrayUtils import ArrayUtils
from tests.MetaTestCase import MetaTestCase


class TestPsSelect(MetaTestCase):
    _GEO_DATA_FILE_NAME = 'subset_8_of_S1A_IW_SLC__1SDV_20160614T043402_20160614T043429_011702_011EEA_F130_Stack_deb_ifg_Geo.dim'

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        lonlat_process = CreateLonLat(cls._PATH, cls._GEO_DATA_FILE_NAME)
        lonlat_process.load_results(cls._SAVE_LOAD_PATH)
        cls._ps_files = PsFiles(cls._PATH_PATCH_FOLDER, lonlat_process)
        cls._ps_files.load_results(cls._SAVE_LOAD_PATH)

        cls._est_gamma_process = None

    def test_start_process_with_matlab_data(self):
        def get_keep_ix():
            """Selleks, et saaks Matlab'i tulemitest kätte keep_ix massiivi mida saaks kasutada
            ka Numpy array'de juures.
            Põhiline asi on, et Numpy's on olemas boolean indexing, aga Matlab'ist tulevad
            väärtused 1'de ja 0'dena. Selleks teeme hoopis massiivi indeksiteks mida selekeertida,
            kasutades np.where'i"""
            keep_ix = select1_mat['keep_ix']
            keep_ix = np.reshape(keep_ix, len(select1_mat['keep_ix']))
            keep_ix = np.where(keep_ix == 1)

            return keep_ix[0]

        self.__fill_est_gamma_with_matlab_data()

        self.__start_process()

        select1_mat = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'select1.mat'))

        # Kuna Matlab'i indeksid hakkavad ühest siis liidame siin juurde
        np.testing.assert_array_almost_equal(np.add(self._ps_select.ifg_ind, 1), select1_mat['ifg_index'][0])
        # Reshape, et saaks võrrelda omavahel. Matlab'i massiv on kahtlane veerumassiiv
        np.testing.assert_array_almost_equal(np.add(self._ps_select.coh_thresh_ind, 1),
                                   select1_mat['ix'].reshape(len(select1_mat['ix'])))
        np.testing.assert_array_almost_equal(self._ps_select.ph_patch, select1_mat['ph_patch2'],
                                             decimal=self._PLACES)
        np.testing.assert_array_almost_equal(self._ps_select.k_ps, select1_mat['K_ps2'])
        np.testing.assert_array_almost_equal(self._ps_select.ph_res, select1_mat['ph_res2'])
        np.testing.assert_array_almost_equal(self._ps_select.coh_ps2, select1_mat['coh_ps2'])
        coh_thresh = select1_mat['coh_thresh']
        np.testing.assert_array_almost_equal(ArrayUtils.to_col_matrix(self._ps_select.coh_thresh),
                                             coh_thresh)
        keep_ix = get_keep_ix()
        np.testing.assert_array_almost_equal(self._ps_select.keep_ind, keep_ix,
                                             decimal=self._PLACES)

    def test_save_and_load_results(self):
        self.__fill_est_gamma_with_matlab_data()
        self.__start_process()
        self._ps_select.save_results(self._SAVE_LOAD_PATH)

        ps_select_loaded = PsSelect(self._ps_files, self._est_gamma_process)
        ps_select_loaded.load_results(self._SAVE_LOAD_PATH)

        np.testing.assert_array_equal(self._ps_select.ifg_ind, ps_select_loaded.ifg_ind)
        np.testing.assert_array_equal(self._ps_select.coh_thresh_ind, ps_select_loaded.coh_thresh_ind)
        np.testing.assert_array_equal(self._ps_select.keep_ind, ps_select_loaded.keep_ind)
        np.testing.assert_array_equal(self._ps_select.ph_patch, ps_select_loaded.ph_patch)
        np.testing.assert_array_equal(self._ps_select.k_ps, ps_select_loaded.k_ps)
        np.testing.assert_array_equal(self._ps_select.ph_res, ps_select_loaded.ph_res)
        np.testing.assert_array_equal(self._ps_select.coh_ps2, ps_select_loaded.coh_ps2)
        np.testing.assert_array_equal(self._ps_select.coh_thresh, ps_select_loaded.coh_thresh)

    def __start_process(self):
        self._ps_select = PsSelect(self._ps_files, self._est_gamma_process)
        self._ps_select.start_process()

    def __fill_est_gamma_with_matlab_data(self):
        pm1_mat = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'pm1.mat'))
        self._est_gamma_process = PsEstGamma(self._ps_files, False)
        self._est_gamma_process.coherence_bins = pm1_mat['coh_bins'][0]
        self._est_gamma_process.grid_ij = pm1_mat['grid_ij']
        self._est_gamma_process.nr_trial_wraps = pm1_mat['n_trial_wraps'][0][0]
        self._est_gamma_process.ph_patch = pm1_mat['ph_patch']
        self._est_gamma_process.k_ps = pm1_mat['K_ps']
        self._est_gamma_process.c_ps = pm1_mat['C_ps']
        self._est_gamma_process.coh_ps = pm1_mat['coh_ps']
        self._est_gamma_process.n_opt = pm1_mat['N_opt']
        self._est_gamma_process.ph_res = pm1_mat['ph_res']
        self._est_gamma_process.ph_grid = pm1_mat['ph_grid']
        self._est_gamma_process.low_pass = pm1_mat['low_pass']
        self._est_gamma_process.rand_dist = pm1_mat['Nr'][0]