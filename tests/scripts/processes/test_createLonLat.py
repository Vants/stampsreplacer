import os
import scipy.io

from scripts.processes.CreateLonLat import CreateLonLat
from tests.AbstractTestCase import AbstractTestCase


class TestCreateLonLat(AbstractTestCase):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls._GEO_DATA_FILE = os.path.join(cls._PATH,
                                           'subset_8_of_S1A_IW_SLC__1SDV_20160614T043402_20160614T043429_011702_011EEA_F130_Stack_deb_ifg_Geo.dim')

    def test_start_process(self):
        process = self.__start_process()

        pscands_actual = process.pscands_ij
        lonlat_actual = process.lonlat

        lonlat_expected = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'lonlat.mat')).get(
            'lonlat')

        self.assertNotEqual(len(lonlat_actual), 0, "made lonlat is empty")
        self.assertIsNotNone(lonlat_expected, "lonlat.mat not found (is None)")
        self.assertEqual(len(lonlat_expected), len(lonlat_actual),
                         "Made lonlat does not equal with .mat file")

        self.assertIsNotNone(pscands_actual)
        self.assertEqual(len(pscands_actual), len(lonlat_actual))

        for row_num in range(len(lonlat_actual) - 1):
            self.assertAlmostEqual(lonlat_expected[row_num, 0], lonlat_actual[row_num, 0],
                                   self._PLACES)
            self.assertAlmostEqual(lonlat_expected[row_num, 1], lonlat_actual[row_num, 1],
                                   self._PLACES)

    def test_save_and_load_results(self):
        process_save = self.__start_process()
        process_save.save_results(self._SAVE_LOAD_PATH)

        process_load = CreateLonLat(self._PATH_PATCH_FOLDER, self._GEO_DATA_FILE)
        process_load.load_results(self._SAVE_LOAD_PATH)

        self.assertIsNotNone(process_load.lonlat)
        self.assertNotEqual(len(process_load.lonlat), 0, "lonlat is empty")

        self.assertIsNotNone(process_load.pscands_ij)
        self.assertNotEqual(len(process_load.pscands_ij), 0, "pscands_ij_array is empty")

    def __start_process(self) -> CreateLonLat:
        process = CreateLonLat(self._PATH_PATCH_FOLDER, self._GEO_DATA_FILE)
        process.start_process()

        return process
