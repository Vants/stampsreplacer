from unittest import TestCase

from tests.ConfigUtils import ConfigUtils


class AbstractTestCase(TestCase):
    """See klass on justkui abstraktne klass testidele. Siin on üldmeetodi ja setup testidele"""
    _PLACES = 5

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls._config = ConfigUtils()

        cls._PATH = cls._config.get_default_section('tests_files_path')