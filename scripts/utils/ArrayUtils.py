import numpy as np


class ArrayUtils:
    @staticmethod
    def arange_include_last(start: float, end: float, step: float = 1.0):
        """Funktsioon on mõeldud selleks, et kui on vaja maatriksit algusest lõpuni vastava sammuga
        ning on tähtis, et viimane element oleks olemas. np.arange ei kindlusta seda, et viimane
        element oleks olemas."""

        # Nii peab tegema, sest muidu on tagastusmassiiv tüübiga float64
        if step == 1.0:
            aranged = np.arange(start, end)
        else:
            aranged = np.arange(start, end, step)

        if len(aranged) == 0 or aranged[len(aranged) - 1] != end:
            aranged = np.append(aranged, [end])

        return aranged

    @staticmethod
    def to_col_matrix(array: np.array):
        """Sisse antud maatriks transponeeritakse ning tehakse veerumaatriksiks,
         mis on Matlab'is levinud viis.
         Maatriksi puhul ei kontrollita kas tegemist on veeru- või reamaatriksiga."""

        return array[np.newaxis].transpose()

    @staticmethod
    def matrix_to_array(matrix: np.matrix) -> np.ndarray:
        return np.squeeze(np.asarray(matrix))