import numpy as np

from unittest import TestCase

from scripts.utils.MatlabUtils import MatlabUtils


class TestMatlabUtils(TestCase):
    __multi_row_array = np.array([[1, 2, 3], [4, 5, 6]])
    __single_row_array = np.array([1, 2, 3])

    def test_max_multi(self):
        np.testing.assert_array_equal(MatlabUtils.max(self.__multi_row_array), np.array([4, 5, 6]))

    def test_max_single(self):
        self.assertEqual(MatlabUtils.max(self.__single_row_array), 3)

    def test_min_multi(self):
        np.testing.assert_array_equal(MatlabUtils.min(self.__multi_row_array), np.array([1, 2, 3]))

    def test_min_single(self):
        self.assertEqual(MatlabUtils.min(self.__single_row_array), 1)

    def test_sum_multi(self):
        np.testing.assert_array_equal(MatlabUtils.sum(self.__multi_row_array), np.array([5, 7, 9]))

    def test_sum_single(self):
        self.assertEqual(MatlabUtils.sum(self.__single_row_array), 6)
