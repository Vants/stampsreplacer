from builtins import staticmethod

import numpy as np


class MatrixUtils:
    @staticmethod
    def sort_matrix_with_sort_array(matrix: np.matrix, sort_ind: np.ndarray):
        """Numpy ei luba lexsortiga sorteerida maatrikseid. 
        Selle jaoks tehakse siin massiivi (np.ndarray) view maatriksist.
        Sorteeritud maatriks tagastatakse input parameertiga (mitte return'iga)
        
        Allikas http://stackoverflow.com/questions/13338110/python-matrix-sorting-via-one-column"""

        tmp = matrix.view(np.ndarray)
        matrix = np.asmatrix(tmp[sort_ind])

    @staticmethod
    def delete_master_col(matrix: np.matrix, master_ind: int):
         return np.delete(matrix, master_ind - 1, axis=1)

    @staticmethod
    def max(array: np.ndarray):
        return np.amax(array, axis=0)

    @staticmethod
    def min(array: np.ndarray):
        return np.amin(array, axis=0)

    @staticmethod
    def sum(array: np.ndarray):
        return np.sum(array, axis=0)