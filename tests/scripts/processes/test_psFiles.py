import os
import scipy.io
import h5py
import numpy as np
from setuptools.command.test import test
from spyder_io_hdf5 import hdf5

from unittest import TestCase

from scripts.processes.CreateLonLat import CreateLonLat
from scripts.processes.PsFiles import PsFiles
from scripts.utils.FolderConstants import FolderConstants


class TestPsFiles(TestCase):
    _PATH = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'resources'))
    _GEO_DATA_FILE = os.path.join(_PATH,
                                  'subset_8_of_S1A_IW_SLC__1SDV_20160614T043402_20160614T043429_011702_011EEA_F130_Stack_deb_ifg_Geo.dim')
    _PATCH_1_FOLDER = os.path.join(_PATH, FolderConstants.PATCH_FOLDER)
    _PLACES = 5

    def setUp(self):
        super().setUp()
        self.lonlat_process = CreateLonLat(self._PATH, self._GEO_DATA_FILE)

    def test_load_files(self):
        lonlat = self.lonlat_process.load_results()

        ps_files = PsFiles(self._PATH, self.lonlat_process.pscands_ij_array, lonlat)
        ps_files.load_files()

        ps1_mat = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'ps1.mat'))
        # Need on salvestatud Matlab 7.3'es
        ph1 = self.__load_mat73(os.path.join(self._PATCH_1_FOLDER, 'ph1.mat'), 'ph')
        bp1 = self.__load_mat73(os.path.join(self._PATCH_1_FOLDER, 'bp1.mat'), 'bperp_mat')
        da1 = self.__load_mat73(os.path.join(self._PATCH_1_FOLDER, 'da1.mat'), 'D_A')

        self.assertEqual(len(ps_files.bperp), len(bp1))
        np.testing.assert_allclose(ps_files.bperp.view(np.ndarray), bp1)

        self.assertEqual(len(ps_files.da), len(da1))
        # Loadmat'is on muutujad omakorda ühemõõtmeliste massiivide sees
        np.testing.assert_allclose(np.reshape(ps_files.da, (len(ps_files.da), 1)), da1)

        np.testing.assert_allclose(ps_files.bperp_meaned, ps1_mat['bperp'])
        np.testing.assert_allclose(ps_files.pscands_ij.view(np.ndarray), ps1_mat['ij'])
        # Meie protsessis esimest veergu ei ole, seetõttu võtame viimased kaks algsest
        np.testing.assert_allclose(ps_files.xy, ps1_mat['xy'][:, 1:])

        self.assertAlmostEqual(ps_files.mean_range, ps1_mat['mean_range'])

        np.testing.assert_allclose(ps_files.mean_incidence.view(np.ndarray),
                                   ps1_mat['mean_incidence'])

        np.testing.assert_allclose(ps_files.ll, ps1_mat['ll0'])

        # Kuna neil pole mat'ides kontrollväärtuseid siis neid kontrollib kas on täidetud
        self.assertNotEquals(ps_files.wavelength, 0)
        self.assertIsNotNone(ps_files.heading)

        self.assertNotEquals(ps_files.master_date, ps1_mat['master_day'])
        self.assertEquals(ps_files.master_ix, ps1_mat['master_ix'])

        # Matlab'os hakkavad väärtused ühest, siis lisame lõppu ühe ja võtame algusest ühe ära
        sort_ix_numpy = np.insert(ps1_mat['sort_ix'], 0, [0])[: len(ps1_mat['sort_ix'])]
        np.testing.assert_array_equal(ps_files.sort_ind, sort_ix_numpy)

        self.assertEquals(len(ps_files.ifgs), ps1_mat['n_ifg'])

        self.assertEquals(len(ps_files.pscands_ij), ps1_mat['n_ps'])

        self.assertEqual(len(ps_files.ph), len(ph1))
        self.assert_ph(ps_files.ph, ph1)

    def assert_ph(self, ph_actual, ph_expected):
        """Matlab'i mat falides pole kompleksarvud definaaeritud nii nagu Numpy's.
        Seepärast peab tegama selliselt selle võrdluse"""
        for row_num in range(len(ph_expected) - 1):
            row_actual = ph_actual[row_num]
            row_expected = ph_expected[row_num]
            for col_num in range(len(row_actual) - 1):
                self.assertAlmostEqual(row_actual[col_num].real, row_expected[col_num]['real'],
                                       self._PLACES,
                                       "Error real row " + str(row_num) + " col " + str(col_num))
                self.assertAlmostEqual(row_actual[col_num].imag, row_expected[col_num]['imag'],
                                       self._PLACES,
                                       "Error imag row " + str(row_num) + " col " + str(col_num))

    def __load_mat73(self, path_with_file_name: str, dataset: str):
        with h5py.File(path_with_file_name) as hdf5_file:
            return hdf5_file[dataset][:].conjugate().transpose()

    def test_save_and_load_results(self):
        lonlat = self.lonlat_process.load_results()

        ps_files_save = PsFiles(self._PATH, self.lonlat_process.pscands_ij_array, lonlat)
        ps_files_save.load_files()

        ps_files_save.save_results()

        ps_files_load = PsFiles(self._PATH, self.lonlat_process.pscands_ij_array, lonlat)
        ps_files_load.load_results()

        self.assertIsNotNone(ps_files_load.heading)
        self.assertEquals(ps_files_load.mean_range, ps_files_save.mean_range)
        self.assertEquals(ps_files_load.wavelength, ps_files_save.wavelength)
        self.assertEquals(ps_files_load.mean_incidence, ps_files_save.mean_incidence)
        self.assertEquals(ps_files_load.master_ix, ps_files_save.master_ix)
        np.testing.assert_array_equal(ps_files_load.bperp_meaned, ps_files_save.bperp_meaned)
        np.testing.assert_array_equal(ps_files_load.bperp, ps_files_save.bperp)
        np.testing.assert_array_equal(ps_files_load.ph, ps_files_save.ph)
        np.testing.assert_array_equal(ps_files_load.ll, ps_files_save.ll)
        np.testing.assert_array_equal(ps_files_load.xy, ps_files_save.xy)
        np.testing.assert_array_equal(ps_files_load.da, ps_files_save.da)
        np.testing.assert_array_equal(ps_files_load.sort_ind, ps_files_save.sort_ind)
        self.assertEquals(ps_files_load.master_date, ps_files_save.master_date)
        np.testing.assert_array_equal(ps_files_load.ifgs, ps_files_save.ifgs)