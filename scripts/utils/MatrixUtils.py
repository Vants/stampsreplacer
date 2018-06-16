from builtins import staticmethod

import numpy as np


class MatrixUtils:
    @staticmethod
    def sort_matrix_with_sort_array(matrix: np.matrix, sort_ind: np.ndarray):
        """Numpy does not allow to sort matrixes with lexsort. For that we make view of that is
        array and sort with sort_ind.

        Source: http://stackoverflow.com/questions/13338110/python-matrix-sorting-via-one-column
        """

        tmp = matrix.view(np.ndarray)
        return np.asmatrix(tmp[sort_ind])

    @staticmethod
    def delete_master_col(matrix: np.matrix, master_ind: int):
        return np.delete(matrix, master_ind - 1, axis=1)
