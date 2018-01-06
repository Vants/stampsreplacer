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

    # noinspection PyUnresolvedReferences
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        lonlat_process = CreateLonLat(cls._PATH, cls._GEO_DATA_FILE_NAME)
        lonlat_process.load_results(cls._SAVE_LOAD_PATH)
        cls.__ps_files = PsFiles(cls._PATH_PATCH_FOLDER, lonlat_process)
        cls.__ps_files.load_results(cls._SAVE_LOAD_PATH)

        cls.__est_gamma_process: PsEstGamma = None

        # Siin võib ps_est_gamma olla none, sest me laeme ps_select'i eelnevalt salvestatult failist
        cls.__ps_select = PsSelect(cls.__ps_files, cls.__est_gamma_process)
        cls.__ps_select.load_results(cls._SAVE_LOAD_PATH)

        cls.__ps_weed_process = None

    def test_start_process_with_matlab_data(self):
        def bool_to_int_array(bool_array: np.ndarray):
            return np.where(bool_array == 1)[0]

        self.__fill_est_gamma_with_matlab_data()
        self.__start_process()

        weed_mat = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'weed1.mat'))

        np.testing.assert_array_almost_equal(np.where(self.__ps_weed_process.selectable_ps)[0],
                                             bool_to_int_array(weed_mat['ix_weed']))
        np.testing.assert_array_almost_equal(np.where(self.__ps_weed_process.selectable_ps2)[0],
                                             bool_to_int_array(weed_mat['ix_weed2']))

        # Kuna 'drop_noisy' meetodis saadakse weighted_least_sqrt2 teisem kui Snap'is siis selle
        # tulemusena on ka need väärtused natuke teisemad. Küll aga see erinevus ei kandu edasi
        # selectable_ps ja selectable_ps2 massiividele
        PS_RTOL = 0.28
        PS_ATOL = 0.055
        np.testing.assert_allclose(self.__ps_weed_process.ps_std, np.squeeze(weed_mat['ps_std']),
                                   PS_RTOL, PS_ATOL)
        np.testing.assert_allclose(self.__ps_weed_process.ps_max, np.squeeze(weed_mat['ps_max']),
                                   PS_RTOL, PS_ATOL)

        np.testing.assert_array_almost_equal(np.add(self.__ps_weed_process.ifg_ind, 1),
                                             np.reshape(weed_mat['ifg_index'], len(self.__ps_weed_process.ifg_ind)))

    def test_save_and_load_results(self):
        self.__fill_est_gamma_with_matlab_data()
        self.__start_process()

        self.__ps_weed_process.save_results(self._SAVE_LOAD_PATH)

        ps_weed_loaded = PsWeed(self._PATH, self.__ps_files, self.__est_gamma_process, self.__ps_select)

        ps_weed_loaded.load_results(self._SAVE_LOAD_PATH)

        np.testing.assert_array_equal(self.__ps_weed_process.selectable_ps, ps_weed_loaded.selectable_ps)
        np.testing.assert_array_equal(self.__ps_weed_process.selectable_ps2, ps_weed_loaded.selectable_ps2)
        np.testing.assert_array_equal(self.__ps_weed_process.ps_std, ps_weed_loaded.ps_std)
        np.testing.assert_array_equal(self.__ps_weed_process.ps_max, ps_weed_loaded.ps_max)
        np.testing.assert_array_equal(self.__ps_weed_process.ifg_ind, ps_weed_loaded.ifg_ind)

    def test_get_filtered_results(self):
        self.__fill_est_gamma_with_matlab_data()
        self.__ps_weed_process = PsWeed(self._PATH_PATCH_FOLDER, self.__ps_files,
                                        self.__est_gamma_process, self.__ps_select)
        coh_ps, k_ps, c_ps, ph_patch, ph, xy, pscands_ij, lonlat, hgt, bperp, sort_ind =\
            self.__ps_weed_process.get_filtered_results(self._SAVE_LOAD_PATH)

        pm_mat = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'pm2.mat'))

        np.testing.assert_array_almost_equal(coh_ps, pm_mat['coh_ps'])
        np.testing.assert_array_almost_equal(k_ps, pm_mat['K_ps'])
        np.testing.assert_array_almost_equal(c_ps, pm_mat['C_ps'])
        np.testing.assert_array_almost_equal(ph_patch, pm_mat['ph_patch'])

        ph_mat = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'ph2.mat'))

        np.testing.assert_array_almost_equal(ph, ph_mat['ph'])

        ps = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'ps2.mat'))

        # Täpselt nagu PsFiles testis me ei vaata esimest veergu, sest meil pole seda lihtsalt
        np.testing.assert_array_almost_equal(xy, ps['xy'][:, 1:])
        np.testing.assert_array_almost_equal(pscands_ij, ps['ij'])
        np.testing.assert_array_almost_equal(lonlat.view(np.ndarray), ps['lonlat'], self._PLACES)
        np.testing.assert_array_almost_equal(len(ph), ps['n_ps'][0])

        hgt_mat= scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'hgt2.mat'))
        np.testing.assert_array_almost_equal(hgt, np.reshape(hgt_mat['hgt'], len(hgt)))

        bp_mat = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'bp2.mat'))

        np.testing.assert_array_almost_equal(bperp, bp_mat['bperp_mat'], self._PLACES)

        la_mat = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'la2.mat'))

        np.testing.assert_array_almost_equal(sort_ind, la_mat['la'])

    def __start_process(self):
        self.__ps_weed_process = PsWeed(self._PATH_PATCH_FOLDER, self.__ps_files, self.__est_gamma_process, self.__ps_select)
        self.__ps_weed_process.start_process()

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