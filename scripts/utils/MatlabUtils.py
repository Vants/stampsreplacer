import scipy.interpolate
import scipy.signal
from builtins import staticmethod

import numpy as np

from scripts.utils.ArrayUtils import ArrayUtils


class MatlabUtils:
    @staticmethod
    def max(array: np.ndarray):
        if len(array) > 1:
            return np.amax(array, axis=0)
        else:
            return np.amax(array)

    @staticmethod
    def min(array: np.ndarray):
        if len(array) > 1:
            return np.amin(array, axis=0)
        else:
            return np.amin(array)

    @staticmethod
    def sum(array: np.ndarray):
        if len(array.shape) > 1:
            return np.sum(array, axis=0)
        else:
            return np.sum(array)

    @staticmethod
    def gausswin(M: int, alpha=2.5):
        """
        This function works like Matlab's Gaussian function. In SciPy (scipy.signal.gaussian) it
        originally works bit differently.

        Idea: https://github.com/openworm/open-worm-analysis-toolbox/blob/master/open_worm_analysis_toolbox/utils.py
        """

        N = M - 1
        n = np.arange(start=0, stop=M) - N / 2
        w = np.exp(-0.5 * np.power((alpha * n / (N / 2)), 2))

        return w

    @staticmethod
    def hist(a: np.ndarray, bins: np.ndarray, density=False):
        """Adds np.Inf to the bins end to make Numpy histograms equal to Matlab's.
        Density helps with decimal values in response."""

        new_bins = np.r_[-np.Inf, 0.5 * (bins[:-1] + bins[1:]), np.Inf]
        return np.histogram(a, new_bins, density=density)

    @staticmethod
    def interp(vector: np.ndarray, interp_factor: int, kind: str = 'cubic'):
        vector_len = len(vector)

        arange = np.linspace(0, .1, vector_len)
        interp_fun = scipy.interpolate.interp1d(arange, vector, kind=kind)

        xnew = np.linspace(0, .1, vector_len * interp_factor)
        return interp_fun(xnew)

    @staticmethod
    def std(array: np.ndarray, axis=None):
        """https://stackoverflow.com/questions/27600207/why-does-numpy-std-give-a-different-result-to-matlab-std"""
        return np.std(array, axis, ddof=1)

    @staticmethod
    def polyfit_polyval(x: np.ndarray, y: np.ndarray, deg: int, max_desinty_or_percent_rand: float):
        """
        Function that works like polyfit and polyval where polyfit returns three values.

        https://stackoverflow.com/questions/45338872/matlab-polyval-function-with-three-outputs-equivalent-in-python-numpy/45339206#45339206
        """
        mu = np.mean(x)
        std = MatlabUtils.std(y)

        c_scaled = np.polyfit((x - mu) / std, y, deg)
        p_scaled = np.poly1d(c_scaled)

        polyval = p_scaled((max_desinty_or_percent_rand - mu) / std)

        return polyval

    @staticmethod
    def filter2(h, x, mode='same'):
        """https://stackoverflow.com/questions/43270274/equivalent-of-matlab-filter2filter-image-valid-in-python"""
        return scipy.signal.convolve2d(x, np.rot90(h, 2), mode)

    @staticmethod
    def lscov(A: np.ndarray, B: np.ndarray, weights: np.ndarray):
        """Least-squares solution in presence of known covariance
        https://stackoverflow.com/questions/27128688/how-to-use-least-squares-with-weight-matrix-in-python"""

        W_col_array = weights[:, np.newaxis]
        Aw = A * np.sqrt(W_col_array)
        Bw = B * np.sqrt(weights)[:, np.newaxis]

        return np.linalg.lstsq(Aw, Bw)[0]
