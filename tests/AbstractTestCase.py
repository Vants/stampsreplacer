from unittest import TestCase

from tests.ConfigUtils import ConfigUtils


class AbstractTestCase(TestCase):
    """See klass on justkui abstraktne klass testidele. Siin on üldmeetodi ja setup testidele"""
    _PLACES = 5

def setUp(self):
    super().setUp()

    self._config = ConfigUtils()

    self._PATH = self._config.get_default_section('tests_files_path')
