import numpy as np


class ArrayUtils:
    @staticmethod
    def arange_include_last(start: float, end: float, step: float = 1.0):
        """Fixes np.arange function. np.arange function does not guarentee that you have that
        last element in the end. Using this function you have that."""

        # You need to do it like that, otherwise you get return type float64
        if step == 1.0:
            aranged = np.arange(start, end)
        else:
            aranged = np.arange(start, end, step)

        if len(aranged) == 0 or aranged[len(aranged) - 1] != end:
            aranged = np.append(aranged, [end])

        return aranged

    @staticmethod
    def to_col_matrix(array: np.array):
        """Input array is transposed and made to column matrix.
        There isn't check if it is col - or row matrix."""

        return array[np.newaxis].transpose()

    @staticmethod
    def matrix_to_array(matrix: np.matrix) -> np.ndarray:
        return np.squeeze(np.asarray(matrix))