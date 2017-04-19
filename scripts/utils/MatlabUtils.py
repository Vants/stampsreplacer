from scipy.interpolate import interp1d
from builtins import staticmethod

import numpy as np


class MatlabUtils:
    @staticmethod
    def max(array: np.ndarray):
        return np.amax(array, axis=0)

    @staticmethod
    def min(array: np.ndarray):
        return np.amin(array, axis=0)

    @staticmethod
    def sum(array: np.ndarray):
        return np.sum(array, axis=0)

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
    def histogram(a: np.ndarray, bins: np.ndarray):
        """Selleks, et Matlab'i ja Numpy histogrammid oleksd võrdsed paneme bin'idele lõppu 
        lõpmatuse"""

        new_bins = np.append(bins, np.inf)
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