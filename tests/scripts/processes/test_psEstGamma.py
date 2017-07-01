import os
import sys

from unittest import TestCase

from scripts.processes.CreateLonLat import CreateLonLat
from scripts.processes.PsEstGamma import PsEstGamma
from scripts.processes.PsFiles import PsFiles
from scripts.utils.FolderConstants import FolderConstants


class TestPsEstGamma(TestCase):

    _PATH = os.path.abspath(
        os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', '..', 'resources'))
    _GEO_DATA_FILE = os.path.join(_PATH,
                                  'subset_8_of_S1A_IW_SLC__1SDV_20160614T043402_20160614T043429_011702_011EEA_F130_Stack_deb_ifg_Geo.dim')
    _PATCH_1_FOLDER = os.path.join(_PATH, FolderConstants.PATCH_FOLDER)
    _PLACES = 5

    def setUp(self):
        super().setUp()

        lonlat_process = CreateLonLat(self._PATH, self._GEO_DATA_FILE)
        lonlat = lonlat_process.load_results()
        self.ps_files = PsFiles(self._PATH, lonlat_process.pscands_ij_array, lonlat)
        self.ps_files.load_results()

    def test_start_process(self):
        est_gamma_process = PsEstGamma(self.ps_files, True)
        est_gamma_process.start_process()
