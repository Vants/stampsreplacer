import os

from scripts.processes.CreateLonLat import CreateLonLat
from scripts.processes.PsEstGamma import PsEstGamma
from scripts.processes.PsFiles import PsFiles
from scripts.utils.FolderConstants import FolderConstants
from tests.AbstractTestCase import AbstractTestCase


class TestPsEstGamma(AbstractTestCase):
    _TEST_RESOUCES_PATH = ''

    _GEO_DATA_FILE_NAME = 'subset_8_of_S1A_IW_SLC__1SDV_20160614T043402_20160614T043429_011702_011EEA_F130_Stack_deb_ifg_Geo.dim'
    _PLACES = 5

    def setUp(self):
        super().setUp()

        self._PATCH_1_FOLDER = os.path.join(self._PATH, FolderConstants.PATCH_FOLDER)

        lonlat_process = CreateLonLat(self._PATH, self._GEO_DATA_FILE_NAME)
        lonlat = lonlat_process.load_results()
        self.ps_files = PsFiles(self._PATH, lonlat_process.pscands_ij_array, lonlat)
        self.ps_files.load_results()

    def test_start_process(self):
        est_gamma_process = PsEstGamma(self.ps_files, True)
        est_gamma_process.start_process()
