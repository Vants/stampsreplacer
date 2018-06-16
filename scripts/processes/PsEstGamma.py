import math
import os

import pydsm.relab
import sys

import scipy.signal

from scripts.MetaSubProcess import MetaSubProcess
from scripts.funs.PsTopofit import PsTopofit
from scripts.processes.PsFiles import PsFiles

import numpy as np

from scripts.utils.ArrayUtils import ArrayUtils
from scripts.utils.internal.LoggerFactory import LoggerFactory
from scripts.utils.MatlabUtils import MatlabUtils
from scripts.utils.MatrixUtils import MatrixUtils
from scripts.utils.internal.ProcessCache import ProcessCache
from scripts.utils.internal.ProcessDataSaver import ProcessDataSaver


class PsEstGamma(MetaSubProcess):
    """In this process we analize potential phasenoise Analüüsitakse võimalike püsivpeegeldajate faasimüra"""

    low_pass = np.ndarray
    weights = np.ndarray
    weights_org = np.ndarray
    rand_dist = np.ndarray
    nr_max_nz_ind = -1

    __FILE_NAME = "ps_est_gamma"

    def __init__(self, ps_files: PsFiles, rand_dist_cached_file=False,
                 outter_rand_dist=np.array([])) -> None:
        """rand_dist_cached_file= when True loads array of random numbers 'tmp_rand_dist' from cache
        (see function 'self.__make_random_dist')
        outter_rand_dist = array of random numbers that are usually if found in function
        'self.__make_random_dist'. This is used for testing"""

        self.__logger = LoggerFactory.create("PsEstGamma")

        self.__ps_files = ps_files
        self.__set_internal_params()
        self.rand_dist_cached = rand_dist_cached_file
        self.outter_rand_dist = outter_rand_dist

        # In StaMPS this is called 'coh_bins'
        self.coherence_bins = ArrayUtils.arange_include_last(0.005, 0.995, 0.01)

    def __set_internal_params(self):
        """In StaMPS these where loaded to Matlab environment using setparam. All values are from
        ps_params_default file. Explantions are from Hooper's StaMPS/MTI Manual (Version 3.3b1)"""

        # Pixel size of the grid (in meters)
        self.__filter_grid_size = 50
        # The weighting scheme (PS probability squared)
        self.__filter_weighting = 'P-square'
        # CLAP (Combined Low-pass and Adaptive Phase).
        self.__clap_win = 32
        # The wavelenghts that are greater than that are passed through
        self.__clap_low_pass_wavelength = 800

        self.__clap_alpha = 1
        self.__clap_beta = 0.3
        # Maximum uncorrelated DEM error (in meters).
        # Any value greater than this is not accepted
        self.__max_topo_err = 5
        # Threshold for change in change in mean value of γ(coherence like value)
        self.__gamma_change_convergence = 0.005
        # mean range - need only be approximately correct
        self.__mean_range = 830000

        self.__low_coherence_thresh = 31  # Equal to 31/100

    def start_process(self):
        self.__logger.info("Started")

        self.low_pass = self.__get_low_pass()
        self.__logger.debug("low_pass.len: {0}".format(len(self.low_pass)))

        ph, bperp_meaned, bperp, nr_ifgs, nr_ps, xy, da, sort_ind_meaned = self.__load_ps_params()

        self.nr_trial_wraps = self.__get_nr_trial_wraps(bperp_meaned, sort_ind_meaned)
        self.__logger.debug("nr_trial_wraps: {0}".format(self.nr_trial_wraps))

        # self.rand_dist in Stamps is named 'Nr'
        self.rand_dist, self.nr_max_nz_ind = self.__make_random_dist(nr_ps, nr_ifgs, bperp_meaned,
                                                                     self.nr_trial_wraps)
        self.__logger.debug("rand_dist.len: {0}, self.nr_max_nz_ind: {1}"
                            .format(len(self.rand_dist), self.nr_max_nz_ind))

        self.grid_ij = self.__get_grid_ij(xy)
        self.__logger.debug("grid_ij.len: {0}".format(len(self.grid_ij)))

        self.weights_org = self.__get_weights(da)
        self.__logger.debug("weights_org.len: {0}".format(len(self.weights_org)))

        # Eelnev oli sisuliselt eelöö selleks mis nüüd hakkab.
        self.ph_patch, self.k_ps, self.c_ps, self.coh_ps, self.n_opt, \
        self.ph_res, self.ph_grid, self.low_pass = \
            self.__sw_loop(
                ph,
                self.weights_org.copy(),
                self.low_pass,
                bperp,
                nr_ifgs,
                nr_ps,
                self.nr_trial_wraps)

        self.__logger.info("End")

    def save_results(self, save_path: str):
        ProcessDataSaver(save_path, self.__FILE_NAME).save_data(
            ph_patch=self.ph_patch,
            k_ps=self.k_ps,
            c_ps=self.c_ps,
            coh_ps=self.coh_ps,
            n_opt=self.n_opt,
            ph_res=self.ph_res,
            ph_grid=self.ph_grid,
            low_pass=self.low_pass,
            coherence_bins=self.coherence_bins,
            grid_ij=self.grid_ij,
            nr_trial_wraps=self.nr_trial_wraps,
            rand_dist=self.rand_dist,
        )

    def load_results(self, load_path: str):
        file_with_path = os.path.join(load_path, self.__FILE_NAME + ".npz")
        data = np.load(file_with_path)

        self.ph_patch = data['ph_patch']
        self.k_ps = data['k_ps']
        self.c_ps = data['c_ps']
        self.coh_ps = data['coh_ps']
        self.n_opt = data['n_opt']
        self.ph_res = data['ph_res']
        self.ph_grid = data['ph_grid']
        self.low_pass = data['low_pass']
        self.coherence_bins = data['coherence_bins']
        self.grid_ij = data['grid_ij']
        self.nr_trial_wraps = data['nr_trial_wraps'].astype(np.float64)
        self.rand_dist = data['rand_dist']

    def __get_low_pass(self):
        start = -(self.__clap_win) / self.__filter_grid_size / self.__clap_win / 2
        stop = (self.__clap_win - 2) / self.__filter_grid_size / self.__clap_win / 2
        step = 1 / self.__filter_grid_size / self.__clap_win
        freg_i = ArrayUtils.arange_include_last(start, stop, step)

        freg_0 = 1 / self.__clap_low_pass_wavelength

        subtract = 1 + np.power(freg_i / freg_0, 10)
        butter_i = np.divide(1, subtract)

        return np.fft.fftshift(np.asmatrix(butter_i).conj().transpose() * butter_i)

    def __load_ps_params(self):
        """Loads needed parameters from ps_files object and takes what it needs"""

        ph, bperp, nr_ifgs, nr_ps, xy, da = self.__ps_files.get_ps_variables()
        # In StaMPS small_basline=n
        nr_ifgs -= 1 # This is only for this process. In other proecesses nr_ifgs must remain unchanged

        ph = MatrixUtils.delete_master_col(ph, self.__ps_files.master_nr)
        ph_abs = np.abs(ph)
        ph_abs[np.where(ph_abs == 0)] = 1 # Excluding the possibility of division by zero
        ph = np.divide(ph, ph_abs)

        # bprep_meaned is an array of rows (not columns), therefore usual
        # MatixUtils.delete_master_col function does not work
        bprep_meaned = np.delete(self.__ps_files.bperp_meaned, self.__ps_files.master_nr - 1)

        # In Stamps there is 0.052 instead of math.radians(3). Variable is named 'inc_mean'
        sort_ind_meaned = np.mean(self.__ps_files.sort_ind) + math.radians(3)

        return ph, bprep_meaned, bperp, nr_ifgs, nr_ps, xy, da, sort_ind_meaned

    def __get_nr_trial_wraps(self, bperp_meaned, sort_ind_meaned) -> np.float64:
        # todo what is k?
        k = self.__ps_files.wavelength * self.__mean_range * np.sin(sort_ind_meaned) / 4 / math.pi
        max_k = self.__max_topo_err / k

        bperp_range = MatlabUtils.max(bperp_meaned) - MatlabUtils.min(bperp_meaned)

        # todo why such formula?
        return bperp_range * max_k / (2 * math.pi)

    def __make_random_dist(self, nr_ps, nr_ifgs, bperp_meaned, nr_trial_wraps):
        CACHE_FILE_NAME = "tmp_rand_dist"

        def use_cached_from_file():
            try:
                loaded = ProcessCache.get_from_cache(CACHE_FILE_NAME, 'rand_dist', 'nr_max_nz_ind')
                rand_dist = loaded['rand_dist']
                nr_max_nz_ind = loaded['nr_max_nz_ind']

            except FileNotFoundError:
                self.__logger.info("No cache")

                rand_dist, nr_max_nz_ind = random_dist()
                cache(rand_dist, nr_max_nz_ind)

            return rand_dist, nr_max_nz_ind

        def cache(rand_dist: np.ndarray, nr_max_nz_ind: int):
            ProcessCache.save_to_cache(CACHE_FILE_NAME,
                                       rand_dist=rand_dist,
                                       nr_max_nz_ind=nr_max_nz_ind)

        def use_outter_array(outter_array: np.ndarray):
            rand_dist = outter_array
            nr_max_nz_ind = np.count_nonzero(rand_dist)

            return rand_dist, nr_max_nz_ind

        def random_dist():
            NR_RAND_IFGS = nr_ps  # In StaMPS it is 300000
            random = np.random.RandomState(2005)

            rnd_ifgs = 2 * math.pi * random.rand(NR_RAND_IFGS, nr_ifgs)

            random_coherence = np.zeros((NR_RAND_IFGS, 1))
            for i in range(NR_RAND_IFGS - 1, 0, -1):
                phase = np.exp(1j * rnd_ifgs[i])
                # We need only coherence here
                _, coherence_0, _, _ = PsTopofit.ps_topofit_fun(phase, bperp_meaned, nr_trial_wraps)
                random_coherence[i] = coherence_0[0]

            del rnd_ifgs

            hist, _ = MatlabUtils.hist(random_coherence, self.coherence_bins)

            rand_dist = hist

            return rand_dist, np.count_nonzero(hist)

        if self.rand_dist_cached:
            self.__logger.info("Using cache from file")
            return use_cached_from_file()
        elif len(self.outter_rand_dist) > 0:
            self.__logger.info("Using cache parameter. self.outter_rand_dist.len: {0}"
                               .format(len(self.outter_rand_dist)))
            return use_outter_array(self.outter_rand_dist)
        else:
            return random_dist()

    def __get_grid_ij(self, xy: np.ndarray):

        def fill_cols_with_xy_values(xy_col: np.ndarray):
            # Float32 is needed because with default type you get too many decimal places. But when
            # there is too many decimal places round up(ceil) makes 2224.00001 to 2225 which is the
            # wrong value
            col_formula = lambda x: np.ceil((x - np.amin(x) + 1e-6).astype(np.float32) / self.__filter_grid_size)

            grid_ij_col = col_formula(xy_col)
            max_ind = np.where(grid_ij_col == np.amax(grid_ij_col))
            grid_ij_col[max_ind] -= 1

            return grid_ij_col

        grid_ij = np.zeros((len(xy), 2), np.int32)

        grid_ij[:, 0] = fill_cols_with_xy_values(xy[:, 1])
        grid_ij[:, 1] = fill_cols_with_xy_values(xy[:, 0])

        return grid_ij

    def __get_weights(self, da: np.ndarray):
        return ArrayUtils.to_col_matrix(np.divide(1, da))

    def __sw_loop(self, ph: np.ndarray, weights: np.ndarray,
                  low_pass: np.ndarray, bprep: np.ndarray, nr_ifgs: int, nr_ps: int,
                  nr_trial_wraps: float):

        SW_ARRAY_SHAPE = (nr_ps, 1)

        def zero_ps_array_cont():
            """Konstruktor tühja pusivpeegeladajate info massiivi loomiseks"""
            return np.zeros(SW_ARRAY_SHAPE)

        def get_ph_weight(bprep, k_ps, nr_ifgs, ph, weights):
            exped = np.exp(np.multiply((-1j * bprep), np.tile(k_ps, (1, nr_ifgs))))
            exp_tiled_weight_multi = np.multiply(exped, np.tile(weights, (1, nr_ifgs)))
            return np.multiply(ph, exp_tiled_weight_multi)

        def is_gamma_in_change_delta():
            return abs(gamma_change_delta) < self.__gamma_change_convergence

        def make_ph_grid(ph_grid_shape: tuple, grid_ij: np.ndarray, weights: np.ndarray,
                         loop_nr: int) -> np.ndarray:
            # np.complex128 is needed because this is the type that pydsm.relab.shiftdim returns.
            # ph_grid and ph_filt are needed to make again. Otherwise there are old values in those
            # arrays
            ph_grid = np.zeros(ph_grid_shape, np.complex128)
            for id in range(loop_nr):
                x_ind = int(grid_ij[id, 0]) - 1
                y_ind = int(grid_ij[id, 1]) - 1
                ph_grid[x_ind, y_ind, :] += pydsm.relab.shiftdim(weights[id, :], -1, nargout=1)[0]

            return ph_grid

        def make_ph_filt(ph_grid_shape: tuple, ph_grid: np.ndarray, loop_nr: int,
                         low_pass: np.ndarray) -> np.ndarray:
            ph_filt = np.zeros(ph_grid_shape, np.complex128)
            for i in range(loop_nr):
                ph_filt[:, :, i] = self.__clap_filt(ph_grid[:, :, i], low_pass)

            return ph_filt

        def make_ph_path(ph_patch: np.ndarray, ph_filt: np.ndarray, grid_ij: np.ndarray,
                         loop_nr: int) -> np.ndarray:
            for i in range(loop_nr):
                x_ind = int(grid_ij[i, 0]) - 1
                y_ind = int(grid_ij[i, 1]) - 1
                ph_patch[i, :nr_ifgs] = np.squeeze(ph_filt[x_ind, y_ind, :])

            not_zero_patches_ind = np.nonzero(ph_patch)
            ph_patch[not_zero_patches_ind] = np.divide(ph_patch[not_zero_patches_ind],
                                                       np.abs(ph_patch[not_zero_patches_ind]))

            return ph_patch

        nr_i = int(np.max(self.grid_ij[:, 0]))
        nr_j = int(np.max(self.grid_ij[:, 1]))
        PH_GRID_SHAPE = (nr_i, nr_j, nr_ifgs)

        coh_ps_result = zero_ps_array_cont()
        gamma_change = 0
        gamma_change_delta = np.inf

        ph_patch = np.zeros(ph.shape, np.complex128)

        k_ps = zero_ps_array_cont()

        # Est topo error
        # Initializing variables that are returned in the end
        c_ps = zero_ps_array_cont()
        n_opt = zero_ps_array_cont()
        ph_res = np.zeros((nr_ps, nr_ifgs))

        log_i = 0 # Used for logging to see how many cycles we have done
        self.__logger.debug("is_gamma_in_change_delta loop begin")
        while not is_gamma_in_change_delta():
            log_i += 1
            self.__logger.debug("gamma change loop i " + str(log_i))
            ph_weight = get_ph_weight(bprep, k_ps, nr_ifgs, ph, weights)

            ph_grid = make_ph_grid(PH_GRID_SHAPE, self.grid_ij, ph_weight, nr_ps)
            ph_filt = make_ph_filt(PH_GRID_SHAPE, ph_grid, nr_ifgs, low_pass)

            self.__logger.debug("ph_filt found. first row: {0}, last row: {1}"
                                .format(ph_filt[0], ph_filt[len(ph_filt) - 1]))

            ph_patch = make_ph_path(ph_patch, ph_filt, self.grid_ij, nr_ps)

            self.__logger.debug("ph_patch found. first row: {0}, last row: {1}"
                                .format(ph_patch[0], ph_patch[len(ph_patch) - 1]))

            del ph_filt

            # This is the slowest part in this process
            topofit = PsTopofit(SW_ARRAY_SHAPE, nr_ps, nr_ifgs)
            topofit.ps_topofit_loop(ph, ph_patch, bprep, nr_trial_wraps)
            k_ps = topofit.k_ps.copy()
            c_ps = topofit.c_ps.copy()
            coh_ps = topofit.coh_ps.copy()
            n_opt = topofit.n_opt.copy()
            ph_res = topofit.ph_res.copy()

            del topofit

            self.__logger.debug("topofit found")

            gamma_change_rms = np.sqrt(np.sum(np.power(coh_ps - coh_ps_result, 2) / nr_ps))
            gamma_change_delta = gamma_change_rms - gamma_change
            # Saving gamma and coherence that are returned later to temp variables
            gamma_change = gamma_change_rms
            coh_ps_result = coh_ps

            self.__logger.debug("is_gamma_in_change_delta() and self.__filter_weighting: "
                                + str(not is_gamma_in_change_delta() and
                                      self.__filter_weighting == 'P-square'))
            if not is_gamma_in_change_delta() and self.__filter_weighting == 'P-square':
                # In Stamps it is named 'Na'
                hist, _ = MatlabUtils.hist(coh_ps, self.coherence_bins)
                self.__logger.debug("hist[0:3] " + str(hist[:3]))
                # The random values are transformed into real values here
                low_coh_thresh_ind = self.__low_coherence_thresh
                real_distr = np.sum(hist[:low_coh_thresh_ind]) / np.sum(
                    self.rand_dist[:low_coh_thresh_ind])
                self.rand_dist = self.rand_dist * real_distr

                hist[hist == 0] = 1
                p_rand = np.divide(self.rand_dist, hist)
                p_rand[:low_coh_thresh_ind] = 1
                p_rand[self.nr_max_nz_ind:] = 0 # In Stamps nr_max_nz_ind is incremented by one
                p_rand[p_rand > 1] = 1
                p_rand_added_ones = np.append(np.ones(7), p_rand)
                filtered = scipy.signal.lfilter(MatlabUtils.gausswin(7), [1], p_rand_added_ones)
                p_rand = filtered / np.sum(MatlabUtils.gausswin(7))
                p_rand = p_rand[7:]

                # Found that 'quadratic' is bit more accurate than 'cubic'
                p_rand = MatlabUtils.interp(np.append([1.0], p_rand), 10, 'quadratic')[:-9]

                # Here we covert coh_ps to indexes array. astype is needed because in Numpy all
                # indexes must be int type.
                # reshape is needed because coh_ps is array of arrays.
                coh_ps_as_ind = np.round(coh_ps * 1000).astype(np.int)
                if len(coh_ps_as_ind.shape) > 1:
                    coh_ps_as_ind = np.squeeze(coh_ps_as_ind)
                # In Stamps this is 'Prand_ps'
                ps_rand = p_rand[coh_ps_as_ind].conj().transpose()

                weights = np.reshape(np.power(1 - ps_rand, 2), SW_ARRAY_SHAPE)

        return ph_patch, k_ps, c_ps, coh_ps_result, n_opt, ph_res, ph_grid, low_pass

    def __clap_filt(self, ph: np.ndarray, low_pass: np.ndarray):
        """CLAP_FILT Combined Low-pass Adaptive Phase filtering.
        Variables nr_win, nr_pad where inputs in StaMPS but these were multiplied before inputing
        into function. Here it is done internally.
        Also clap_alpha ja clap_beta where inputs but in this case those are global class variables
        so we can use them and don't need for inputs"""

        def create_grid(nr_win: int):
            grid_array = ArrayUtils.arange_include_last(0, (nr_win / 2) - 1)
            grid_x, grid_y = np.meshgrid(grid_array, grid_array)
            grid = grid_x + grid_y

            return grid

        # todo What does wind_func mean? This isn't array of functions
        def make_wind_func(grid: np.ndarray):
            WIND_FUNC_TYPE = np.float64
            wind_func = np.array(np.append(grid, np.fliplr(grid), axis=1), WIND_FUNC_TYPE)
            wind_func = np.array(np.append(wind_func, np.flipud(wind_func), axis=0), WIND_FUNC_TYPE)
            # In order to prevent zeros in corners
            wind_func += 1e-6

            return wind_func

        def get_indexes(loop_index: int, inc: int, nr_win: int) -> (int, int):
            i1 = loop_index * inc
            # We don't do -1 because otherwise last element in array is not selected
            i2 = i1 + nr_win

            return i1, i2

        FILTERED_TYPE = np.complex128
        filtered = np.zeros(ph.shape, FILTERED_TYPE)

        ph = np.nan_to_num(ph)

        nr_win = int(self.__clap_win * 0.75)
        nr_pad = int(self.__clap_win * 0.25)

        ph_i_len = ph.shape[0] - 1
        ph_j_len = ph.shape[1] - 1
        nr_inc = int(np.floor(nr_win / 4))
        # Indices begin from zero on Python. That's why those values are greater than StaMPS
        nr_win_i = int(np.ceil(ph_i_len / nr_inc) - 3)
        nr_win_j = int(np.ceil(ph_j_len / nr_inc) - 3) + 1

        wind_func = make_wind_func(create_grid(nr_win))

        # To make transpose work like Matlab we need to convert those to matrix
        # todo: PsSelect has similar thing
        B = np.multiply(np.asmatrix(MatlabUtils.gausswin(7)),
                        np.asmatrix(MatlabUtils.gausswin(7)).transpose())

        nr_win_pad_sum = (nr_win + nr_pad)
        ph_bit = np.zeros((nr_win_pad_sum, nr_win_pad_sum), FILTERED_TYPE)
        # Todo: Refactor
        for i in range(nr_win_i):
            w_f = wind_func.copy()
            i1, i2 = get_indexes(i, nr_inc, nr_win)

            if i2 > ph_i_len:
                i_shift = i2 - ph_i_len - 1
                i2 = ph_i_len + 1
                i1 = ph_i_len - nr_win + 1
                w_f = np.append(np.zeros((i_shift, nr_win)), w_f[:nr_win - i_shift, :],
                                axis=0).astype(FILTERED_TYPE)

            for j in range(nr_win_j):
                w_f2 = w_f
                j1, j2 = get_indexes(j, nr_inc, nr_win)

                if j2 > ph_j_len:
                    j_shift = j2 - ph_j_len - 1
                    # Because array lenghts, ph_i_len and ph_j_len, are already smaller and Numpy
                    # does not take last index when selecting, then we need to add one more
                    j2 = ph_j_len + 1
                    j1 = ph_j_len - nr_win + 1
                    w_f2 = np.append(np.zeros((nr_win, j_shift)), w_f2[:, :nr_win - j_shift],
                                     axis=1).astype(FILTERED_TYPE)

                ph_bit[:nr_win, :nr_win] = ph[i1: i2, j1: j2]

                ph_fft = np.fft.fft2(ph_bit) # todo Todo from fifth decimal point the values are not equal to Stamps result
                # ph_fft = fftw.interfaces.numpy_fft.fft2(ph_bit) #  todo Todo from fifth decimalpoint the values are not equal to Stamps result
                smooth_resp = np.abs(ph_fft) # 'H' in Stamps
                smooth_resp = np.fft.ifftshift(
                    MatlabUtils.filter2(B, np.fft.ifftshift(smooth_resp)))
                # smooth_resp = fftw.interfaces.numpy_fft.ifftshift(
                #     MatlabUtils.filter2(B, fftw.interfaces.numpy_fft.ifftshift(smooth_resp)))
                mean_smooth_resp = np.median(smooth_resp)

                if mean_smooth_resp != 0:
                    smooth_resp /= mean_smooth_resp

                smooth_resp = np.power(smooth_resp, self.__clap_alpha)

                # todo Values under median to zero. Why that?
                smooth_resp -= 1
                smooth_resp[smooth_resp < 0] = 0

                # todo What is G?
                G = smooth_resp * self.__clap_beta + low_pass
                ph_filt = np.fft.ifft2(np.multiply(ph_fft, G))
                # ph_filt = fftw.interfaces.numpy_fft.ifft2(np.multiply(ph_fft, G))
                ph_filt = np.multiply(ph_filt[:nr_win, :nr_win], w_f2)

                filtered[i1:i2, j1:j2] += ph_filt

        return filtered
