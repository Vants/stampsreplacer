import os
import scipy.io
import h5py
import numpy as np

from scripts.processes.CreateLonLat import CreateLonLat
from scripts.processes.PsFiles import PsFiles
from scripts.utils.FolderConstants import FolderConstants
from tests.AbstractTestCase import AbstractTestCase


class TestPsFiles(AbstractTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls._GEO_DATA_FILE = os.path.join(cls._PATH,
                                           'subset_8_of_S1A_IW_SLC__1SDV_20160614T043402_20160614T043429_011702_011EEA_F130_Stack_deb_ifg_Geo.dim')
        cls._PATCH_1_FOLDER = os.path.join(cls._PATH, FolderConstants.PATCH_FOLDER)

        cls.lonlat_process = CreateLonLat(cls._PATH, cls._GEO_DATA_FILE)
        cls.lonlat = cls.lonlat_process.load_results()

        cls._ps_files = None

    def test_load_files(self):
        self.__start_process()

        ps1_mat = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'ps1.mat'))
        # Need on salvestatud Matlab 7.3'es
        ph1 = self.__load_mat73(os.path.join(self._PATCH_1_FOLDER, 'ph1.mat'), 'ph')
        bp1 = self.__load_mat73(os.path.join(self._PATCH_1_FOLDER, 'bp1.mat'), 'bperp_mat')
        da1 = self.__load_mat73(os.path.join(self._PATCH_1_FOLDER, 'da1.mat'), 'D_A')
        la1 = self.__load_mat73(os.path.join(self._PATCH_1_FOLDER, 'la1.mat'), 'la')

        self.assertEqual(len(self._ps_files.bperp), len(bp1))
        np.testing.assert_allclose(self._ps_files.bperp.view(np.ndarray), bp1)

        self.assertEqual(len(self._ps_files.da), len(da1))
        # Loadmat'is on muutujad omakorda ühemõõtmeliste massiivide sees
        np.testing.assert_allclose(np.reshape(self._ps_files.da, (len(self._ps_files.da), 1)), da1)

        np.testing.assert_allclose(self._ps_files.bperp_meaned, ps1_mat['bperp'])
        np.testing.assert_allclose(self._ps_files.pscands_ij.view(np.ndarray), ps1_mat['ij'])
        # Meie protsessis esimest veergu ei ole, seetõttu võtame viimased kaks algsest
        np.testing.assert_allclose(self._ps_files.xy, ps1_mat['xy'][:, 1:])

        self.assertAlmostEqual(self._ps_files.mean_range, ps1_mat['mean_range'])

        np.testing.assert_allclose(self._ps_files.mean_incidence.view(np.ndarray),
                                   ps1_mat['mean_incidence'])

        np.testing.assert_allclose(self._ps_files.ll, ps1_mat['ll0'])

        np.testing.assert_allclose(self._ps_files.lonlat, ps1_mat['lonlat'])

        # Kuna neil pole mat'ides kontrollväärtuseid siis neid kontrollib kas on täidetud
        self.assertNotEquals(self._ps_files.wavelength, 0)
        self.assertIsNotNone(self._ps_files.heading)

        self.assertNotEquals(self._ps_files.master_date, ps1_mat['master_day'])
        self.assertEquals(self._ps_files.master_ix, ps1_mat['master_ix'])

        # Matlab'os hakkavad väärtused ühest, siis lisame lõppu ühe ja võtame algusest ühe ära
        #
        # sort_ix_numpy = np.insert(ps1_mat['sort_ix'], 0, [0])[: len(ps1_mat['sort_ix'])]
        np.testing.assert_allclose(self._ps_files.sort_ind.view(np.ndarray), la1)

        self.assertEquals(len(self._ps_files.ifgs), ps1_mat['n_ifg'])

        self.assertEquals(len(self._ps_files.pscands_ij), ps1_mat['n_ps'])

        self.assertEqual(len(self._ps_files.ph), len(ph1))
        self.assert_ph(self._ps_files.ph, ph1)

    def assert_ph(self, ph_actual, ph_expected):
        """Matlab'i mat falides pole kompleksarvud defineeritud nii nagu Numpy's.
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
        self.__start_process()
        self._ps_files.save_results()

        ps_files_load = PsFiles(self._PATH, self.lonlat_process.pscands_ij_array, self.lonlat)
        ps_files_load.load_results()

        self.assertIsNotNone(ps_files_load.heading)
        self.assertEquals(ps_files_load.mean_range, self._ps_files.mean_range)
        self.assertEquals(ps_files_load.wavelength, self._ps_files.wavelength)
        self.assertEquals(ps_files_load.mean_incidence, self._ps_files.mean_incidence)
        self.assertEquals(ps_files_load.master_ix, self._ps_files.master_ix)
        np.testing.assert_array_equal(ps_files_load.bperp_meaned, self._ps_files.bperp_meaned)
        np.testing.assert_array_equal(ps_files_load.bperp, self._ps_files.bperp)
        np.testing.assert_array_equal(ps_files_load.ph, self._ps_files.ph)
        np.testing.assert_array_equal(ps_files_load.ll, self._ps_files.ll)
        np.testing.assert_array_equal(ps_files_load.xy, self._ps_files.xy)
        np.testing.assert_array_equal(ps_files_load.da, self._ps_files.da)
        np.testing.assert_array_equal(ps_files_load.sort_ind, self._ps_files.sort_ind)
        self.assertEquals(ps_files_load.master_date, self._ps_files.master_date)
        np.testing.assert_array_equal(ps_files_load.ifgs, self._ps_files.ifgs)

    def __start_process(self):
        self._ps_files = PsFiles(self._PATH, self.lonlat_process.pscands_ij_array, self.lonlat)
        self._ps_files.load_files()