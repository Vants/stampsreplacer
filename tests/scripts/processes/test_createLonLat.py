import os
import scipy.io
from unittest import TestCase

from scripts.processes.CreateLonLat import CreateLonLat
from scripts.utils.FolderConstants import FolderConstants


class TestCreateLonLat(TestCase):
    # os.path.realpath on vajalik kui test käivitada käsurealt
    _PATH = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'resources'))
    _GEO_DATA_FILE = os.path.join(_PATH,
                                  'subset_8_of_S1A_IW_SLC__1SDV_20160614T043402_20160614T043429_011702_011EEA_F130_Stack_deb_ifg_Geo.dim')
    _PATCH_1_FOLDER = os.path.join(_PATH, FolderConstants.PATCH_FOLDER)
    _PLACES = 5

    def test_start_process(self):
        process = CreateLonLat(self._PATH, self._GEO_DATA_FILE)
        lonlat_actual = process.start_process()

        lonlat_expected = scipy.io.loadmat(os.path.join(self._PATCH_1_FOLDER, 'lonlat.mat')).get(
            'lonlat')

        self.assertNotEqual(len(lonlat_actual), 0, "made lonlat is empty")
        self.assertIsNotNone(lonlat_expected, "lonlat.mat not found (is None)")
        self.assertEqual(len(lonlat_expected), len(lonlat_actual),
                         "Made lonlat does not equal with .mat file")

        for row_num in range(len(lonlat_actual) - 1):
            self.assertAlmostEqual(lonlat_expected[row_num, 0], lonlat_actual[row_num, 0],
                                   self._PLACES)
            self.assertAlmostEqual(lonlat_expected[row_num, 1], lonlat_actual[row_num, 1],
                                   self._PLACES)
