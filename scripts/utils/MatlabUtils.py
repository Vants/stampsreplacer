from scipy.interpolate import interp1d
import scipy.signal
from builtins import staticmethod

import numpy as np

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
        """Gaussi meetod töötab Maltlab'is natuke teisiti kui SciPy (scipy.signal.gaussian). 
        See funktioon töötab nagu Matlab'is.
        Idee: https://github.com/openworm/open-worm-analysis-toolbox/blob/master/open_worm_analysis_toolbox/utils.py"""

        N = M - 1
        n = np.arange(start=0, stop=M) - N / 2
        w = np.exp(-0.5 * np.power((alpha * n / (N / 2)), 2))

        return w

    @staticmethod
    def hist(a: np.ndarray, bins: np.ndarray):
        """Selleks, et Matlab'i ja Numpy hist funktsioonid oleksd võrdsed paneme bin'idele lõppu
        lõpmatuse. density=True aitab sellega, et vastustesse tekikisd murdarvud"""
        new_bins = np.r_[-np.Inf, 0.5 * (bins[:-1] + bins[1:]), np.Inf]
        return np.histogram(a, new_bins)

    @staticmethod
    def interp(y, m):
        """http://stackoverflow.com/questions/23024950/interp-function-in-python-like-matlab"""

        y = list(y)
        y.append(2 * y[-1] - y[-2])

        xs = np.arange(len(y))
        fn = interp1d(xs, y)

        new_xs = np.arange(len(y) - 1, step=1. / m)
        return fn(new_xs)

    @staticmethod
    def std(array: np.ndarray):
        """https://stackoverflow.com/questions/27600207/why-does-numpy-std-give-a-different-result-to-matlab-std"""
        return np.std(array, ddof=1)

    @staticmethod
    def polyfit_polyval(x: np.ndarray, y: np.ndarray, deg: int, max_desinty_or_percent_rand: float):
        """Funkstioon mis teeb seda mida Matlab'is teeb polyfit ja polyval'i, kus polyfit tagastab
        kolm muutujat S ja mu.

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
    def lscov(A: np.ndarray, B: np.ndarray, W: np.ndarray):
        """Least-squares solution in presence of known covariance
        https://stackoverflow.com/questions/27128688/how-to-use-least-squares-with-weight-matrix-in-python"""

        W_col_array = W[np.newaxis].transpose()
        Aw = A * np.sqrt(W_col_array)
        Bw = B * np.sqrt(W)

        np.linalg.lstsq(Aw, Bw)