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
        lonlat = self.lonlat_process.start_process()

        ps_files = PsFiles(self._PATH, self.lonlat_process.pscands_ij_array, lonlat)
        ps_files.load_files()

        ps1_mat = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'ps1.mat'))
        # Need on salvestatud Matlab 7.3'es
        ph1 = self.__load_mat73(os.path.join(self._PATCH_1_FOLDER, 'ph1.mat'), 'ph')
        bp1 = self.__load_mat73(os.path.join(self._PATCH_1_FOLDER, 'bp1.mat'), 'bperp_mat')

        self.assertEqual(len(ps_files.bperp), len(bp1))
        np.testing.assert_allclose(ps_files.bperp.view(np.ndarray), bp1)

        self.assertEqual(len(ps_files.ph), len(ph1))

        self.assert_ph(ph1, ps_files)

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
