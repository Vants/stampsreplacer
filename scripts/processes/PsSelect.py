import enum

import numpy as np
import os

from scripts.MetaSubProcess import MetaSubProcess
from scripts.funs.PsTopofit import PsTopofit
from scripts.processes.PsEstGamma import PsEstGamma
from scripts.processes.PsFiles import PsFiles
from scripts.utils.ArrayUtils import ArrayUtils
from scripts.utils.internal.LoggerFactory import LoggerFactory
from scripts.utils.MatlabUtils import MatlabUtils
from scripts.utils.internal.ProcessCache import ProcessCache
from scripts.utils.internal.ProcessDataSaver import ProcessDataSaver


class PsSelect(MetaSubProcess):
    """Select stabile pixels that become persistent scatterer"""


    __B = np.array([])

    __FILE_NAME = "ps_select"

    def __init__(self, ps_files: PsFiles, ps_est_gamma: PsEstGamma):
        self.__PH_PATCH_CACHE = True
        self.__ps_files = ps_files
        self.__ps_est_gamma = ps_est_gamma

        self.__logger = LoggerFactory.create("PsSelect")

        self.__set_internal_params()

    def __set_internal_params(self):
        """In StaMPS these where saved with setparam and getparam.
        All values are that small_baseline_flag = 'N'.

        In StaMPS max_desinty_rand ja max_percent_rand where two seperate varaibles, there we get
        them using function __get_max_rand.
        """
        self.__slc_osf = 1
        self.__clap_alpha = 1
        self.__clap_beta = 0.3
        self.__clap_win = 32
        self.__select_method = self._SelectMethod.DESINTY  # DESINITY or PERCENT
        # todo Why is this here
        self.__gamma_stdev_reject = 0
        # TODO This was [] in Stamps
        self.__drop_ifg_index = np.array([])
        self.__low_coh_tresh = 31  # 31/100

        self.__gaussian_window = np.multiply(np.asmatrix(MatlabUtils.gausswin(7)),
                                             np.asmatrix(MatlabUtils.gausswin(7)).conj().transpose())

    class __DataDTO(object):
        """This is inner data transfer object. It is because some functions take very many
        paramters, so we use this class. It is filled in load_ps_params function"""

        def __init__(self, ph: np.ndarray, nr_ifgs: int, xy: np.ndarray,
                     da: np.ndarray, ifg_ind: np.ndarray, da_max: np.ndarray,
                     rand_dist: np.ndarray):
            self.ph = ph
            self.nr_ifgs = nr_ifgs
            self.xy = xy
            self.da = da
            self.ifg_ind = ifg_ind
            self.da_max = da_max
            self.rand_dist = rand_dist

    @enum.unique
    class _SelectMethod(enum.Enum):
        """Internal varaible 'select_method' possible values"""
        DESINTY = 1
        PERCENT = 2

    def start_process(self):
        """Please note that min_coh, coh_thresh and coh_thresh_ind params must be precise as
        possible. Because 0.0001 offset may ruin coh_threh result"""

        self.__logger.info("Start")

        data = self.__load_ps_params()

        max_rand = self.__get_max_rand(data.da_max, data.xy)
        self.__logger.debug("max_rand: {0}".format(max_rand))

        min_coh, da_mean, is_min_coh_nan_array = self.__get_min_coh_and_da_mean(
            self.__ps_est_gamma.coh_ps, max_rand, data)
        self.__logger.debug("min_coh.len: {0} ; da_mean.len: {1}"
                            .format(len(min_coh), len(da_mean)))

        coh_thresh = self.__get_coh_thresh(min_coh, da_mean, is_min_coh_nan_array, data.da)
        self.__logger.debug("coh_thresh.len: {0}".format(len(coh_thresh)))

        coh_thresh_ind = self.__get_coh_thresh_ind(coh_thresh, data)
        self.__logger.debug("coh_thresh_ind.len: {0}".format(len(coh_thresh_ind)))

        ph_patch = self.__get_ph_patch(coh_thresh_ind, data)
        self.__logger.debug("ph_patch.shape: {0}".format(ph_patch.shape))

        coh_ps, topofit = self.__topofit(ph_patch, coh_thresh_ind, data)
        self.__logger.debug("coh_ps.len: {0}".format(len(coh_ps)))

        # And now we find coh_thresh again using new coh_os. For that we also need to find min_coh
        # and da_mean.

        min_coh, da_mean, is_min_coh_nan_array = self.__get_min_coh_and_da_mean(
            coh_ps, max_rand, data)
        self.__logger.debug("Second run min_coh.len: {0} ; da_mean.len: {1}"
                            .format(len(min_coh), len(da_mean)))

        # Please note that da array is filtered by coh_thresh_ind
        coh_thresh = self.__get_coh_thresh(min_coh, da_mean, is_min_coh_nan_array,
                                           data.da[coh_thresh_ind])
        self.__logger.debug("Second run coh_thresh.len: {0}".format(len(coh_thresh)))

        # todo Maybe filter when you find those results
        keep_ind = self.__get_keep_ind(topofit.coh_ps, coh_thresh, coh_thresh_ind,
                                                   topofit.k_ps)
        self.__logger.debug("keep_ind.len: {0}"
                            .format(len(keep_ind)))

        # Results to class variables
        self.coh_thresh = coh_thresh
        self.ph_patch = ph_patch
        self.coh_thresh_ind = coh_thresh_ind
        self.keep_ind = keep_ind
        self.coh_ps = coh_ps # In StaMPS this result was overriden from last process
        self.coh_ps2 = topofit.coh_ps # Find better name
        self.ph_res = topofit.ph_res
        self.k_ps = topofit.k_ps
        self.c_ps = topofit.c_ps
        self.ifg_ind = data.ifg_ind

        self.__logger.debug("End")

    def save_results(self, save_path: str):
        ProcessDataSaver(save_path, self.__FILE_NAME).save_data(
            coh_thresh=self.coh_thresh,
            ph_patch=self.ph_patch,
            coh_thresh_ind=self.coh_thresh_ind,
            keep_ind=self.keep_ind,
            coh_ps=self.coh_ps,
            coh_ps2=self.coh_ps2,
            ph_res=self.ph_res,
            k_ps=self.k_ps,
            c_ps=self.c_ps,
            ifg_ind=self.ifg_ind
        )

    def load_results(self, load_path: str):
        file_with_path = os.path.join(load_path, self.__FILE_NAME + ".npz")
        data = np.load(file_with_path)

        self.coh_thresh = data['coh_thresh']
        self.ph_patch = data['ph_patch']
        self.coh_thresh_ind = data['coh_thresh_ind']
        self.keep_ind = data['keep_ind']
        self.coh_ps = data['coh_ps']
        self.coh_ps2 = data['coh_ps2']
        self.ph_res = data['ph_res']
        self.k_ps = data['k_ps']
        self.c_ps = data['c_ps']
        self.ifg_ind = data['ifg_ind']

    def __load_ps_params(self) -> __DataDTO:
        """Finds values that are needed from ps_files and changes them a bit. It is similar to
        load_ps_params method in PsEstGamma function."""

        def get_da_max(da):
            # todo miks 10000?
            if da.size >= 10000:
                da_sorted = np.sort(da, axis=0)
                if da.size >= 50000:
                    bin_size = 10000
                else:
                    bin_size = 2000

                # bin_size - 1 is for that to take elements with correct indexes that are in Matlab
                da_max = np.concatenate(
                    (np.zeros(1), da_sorted[bin_size - 1: -bin_size - 1: bin_size],
                     np.array([da_sorted[-1]])))
            else:
                da_max = np.array([0], [1])
                da = np.ones(len(self.__ps_est_gamma.coh_ps))

            return da_max, da

        def filter_params_based_on_ifgs_and_master(ph: np.ndarray, bperp: np.ndarray, nr_ifgs: int):
            """Filter out master row form ph and bperp arrays"""

            comp_fun = lambda x, y: x < y

            no_master_ix = np.setdiff1d(np.arange(0, nr_ifgs),
                                        self.__ps_files.master_nr - 1)

            ifg_ind = np.setdiff1d(np.arange(0, nr_ifgs), self.__drop_ifg_index)
            ifg_ind = np.setdiff1d(ifg_ind, self.__ps_files.master_nr)
            master_ix = self.__ps_files.get_nr_ifgs_copared_to_master(comp_fun) - 1
            ifg_ind[ifg_ind > master_ix] -= 1

            ph = ph[:, no_master_ix]
            bperp = bperp[no_master_ix]
            nr_ifgs = len(no_master_ix)

            return ifg_ind, ph, bperp, nr_ifgs

        ph, bperp, nr_ifgs, _, xy, da = self.__ps_files.get_ps_variables()

        # In StaMPS this was done when small_base_line flag was not 'y'. Beacause this process is
        # made as small_baseline_flag value is 'n' we also make this always
        ifg_ind, ph, bperp, nr_ifgs = filter_params_based_on_ifgs_and_master(ph, bperp, nr_ifgs)

        da_max, da = get_da_max(da)

        # nr_dist in StaMPS
        rand_dist = self.__ps_est_gamma.rand_dist

        data_dto = self.__DataDTO(ph, nr_ifgs, xy, da, ifg_ind, da_max, rand_dist)
        return data_dto

    def __get_max_rand(self, da_max: np.ndarray, xy: np.ndarray):
        """This function finds variable that in StaMPS was called 'max_percent_rand'.

        In StaMPS this variable was read in parameters. But in this process we also change it a bit
        we calculate this here"""

        DEF_VAL = 20

        if self.__select_method is self._SelectMethod.DESINTY:
            # StaMPS'is tagastati min'ist ja max'ist massiivid milles oli üks element
            patch_area = np.prod(MatlabUtils.max(xy) - MatlabUtils.min(xy)) / 1e6  # km'ites
            max_rand = DEF_VAL * patch_area / (len(da_max) -1)
        else:
            max_rand = DEF_VAL

        return max_rand

    def __get_min_coh_and_da_mean(self, coh_ps: np.ndarray, max_rand: float, data: __DataDTO) -> (
            np.ndarray, np.ndarray, bool):

        # Internal parameters because full names are bad to write and read all the time
        coherence_bins = self.__ps_est_gamma.coherence_bins
        rand_dist = self.__ps_est_gamma.rand_dist

        array_size = data.da_max.size - 1

        min_coh = np.zeros(array_size)
        # In StaMPS this was size(da_max, 1) what is same as length(da_max)
        da_mean = np.zeros(array_size)
        for i in range(array_size):
            # You can use np.all or np.logical here too. Bitwize isn't must
            coh_chunk = coh_ps[(data.da > data.da_max[i]) & (data.da <= data.da_max[i + 1])]

            da_mean[i] = np.mean(
                data.da[(data.da > data.da_max[i]) & (data.da <= data.da_max[i + 1])])
            # Remove pixels that we could not find coherence
            coh_chunk = coh_chunk[coh_chunk != 0]
            # In StaMPS this was called 'Na'
            hist, _ = MatlabUtils.hist(coh_chunk, coherence_bins)

            hist_low_coh_sum = MatlabUtils.sum(hist[:self.__low_coh_tresh])
            rand_dist_low_coh_sum = MatlabUtils.sum(rand_dist[:self.__low_coh_tresh])
            nr = rand_dist * hist_low_coh_sum / rand_dist_low_coh_sum  # todo What does this 'nr' mean?

            # In StaMPS here was also possibility to make graph

            hist[hist == 0] = 1

            # Percent_rand calculate
            # np.flip allows to use one-dimencional arrays, thats why we don't use np.fliplr
            nr_cumsum = np.cumsum(np.flip(nr, axis=0), axis=0)
            if self.__select_method is self._SelectMethod.PERCENT:
                hist_cumsum = np.cumsum(np.flip(hist, axis=0), axis=0) * 100
                percent_rand = np.flip(np.divide(nr_cumsum, hist_cumsum), axis=0)
            else:
                percent_rand = np.flip(nr_cumsum, axis=0)

            ok_ind = np.where(percent_rand < max_rand)[0]

            if len(ok_ind) == 0:
                # When coherence is over limit
                min_coh[i] = 1
            else:
                # Here we don't need to add one to indexes because on 'ok_ind' array it is already
                # done. This means that all those 'magical constants' are that where in StaMPS

                min_fit_ind = MatlabUtils.min(ok_ind) - 3  # todo Why 3?

                if min_fit_ind <= 0:
                    min_coh[i] = np.nan
                else:
                    max_fit_ind = MatlabUtils.min(ok_ind) + 2  # todo Why 2?

                    # StaMPS'is oli suuruse asemel konstant 100
                    if max_fit_ind > len(percent_rand) - 1:
                        max_fit_ind = len(percent_rand) - 1

                    x_cordinates = percent_rand[min_fit_ind:max_fit_ind + 1]

                    y_cordinates = ArrayUtils.arange_include_last((min_fit_ind + 1) * 0.01,
                                                                  (max_fit_ind + 1) * 0.01, 0.01)
                    min_coh[i] = MatlabUtils.polyfit_polyval(x_cordinates, y_cordinates, 3,
                                                             max_rand)

        # Check if min_coh is unusable (full of nan's
        # This was bit different on StaMPS. I find min_coh'i ja da_mean in same method and in
        # same time
        not_nan_ind = np.where(min_coh != np.nan)[0]
        is_min_coh_nan_array = sum(not_nan_ind) == 0
        # When there isn't differences then we don't need to take subsets of arrays
        if not is_min_coh_nan_array or (not_nan_ind == array_size):
            min_coh = min_coh[not_nan_ind]
            da_mean = da_mean[not_nan_ind]

        return min_coh, da_mean, is_min_coh_nan_array

    def __get_coh_thresh(self, min_coh: np.ndarray, da_mean: np.ndarray,
                         is_min_coh_nan_array: bool, da: np.ndarray):
        """Here we don't return coh_tresh_coffs'i because it was used only for graphs"""
        DEF_COH_THRESH = 0.3

        if is_min_coh_nan_array:
            self.__logger.warn(
                'Not enough random phase pixels to set gamma threshold - using default threshold of '
                + str(DEF_COH_THRESH))
            # Default value is put into array for others to use. Other functions expect array
            # that's why we can't use just float
            coh_thresh = np.array([DEF_COH_THRESH])
        else:
            # Because we have already changed min_coh ja da_mean arrays
            if len(min_coh) > 1:
                coh_thresh_coffs = np.polyfit(da_mean, min_coh, 1)

                if coh_thresh_coffs[0] > 0:
                    # todo mida tähendab "positive slope"?
                    coh_thresh = np.polyval(coh_thresh_coffs, da)
                else:
                    # todo mida tähendab "unable to ascertain correct slope"
                    coh_thresh = np.polyval(coh_thresh_coffs, 0.35)
            else:
                coh_thresh = min_coh

        return coh_thresh

    def __is_select_method_percent(self):
        return self.__select_method is self._SelectMethod.PERCENT

    def __get_coh_thresh_ind(self, coh_thresh: np.ndarray, data: __DataDTO):

        def make_coh_thresh_ind_array(make_function):
            function_result = make_function()

            return function_result

        coh_ps = self.__ps_est_gamma.coh_ps
        ph_res = self.__ps_est_gamma.ph_res

        # We use reshape because this array is bit different
        # [0] is needed because where function returns tuple
        coh_thresh_ind_fun = lambda: np.where(coh_ps.reshape(len(coh_ps)) > coh_thresh)[0]
        # 'ix' in StaMPS
        coh_thresh_ind = make_coh_thresh_ind_array(coh_thresh_ind_fun)

        if self.__gamma_stdev_reject > 0:
            ph_res_cpx = np.exp(1j * ph_res[:, data.ifg_ind])

            coh_std = np.zeros(len(coh_thresh_ind))
            for i in range(len(coh_thresh_ind)):
                # todo Who to make bootstrp'i in Numpy?
                # bootstrap = np.boots
                # coh_std[i] = MatlabUtils.std()
                pass

            coh_thresh_filter_fun = lambda: coh_thresh_ind[coh_std < self.__gamma_stdev_reject]
            coh_thresh_ind = make_coh_thresh_ind_array(coh_thresh_filter_fun)

            # todo In StaMPS here was logic rest_flag. Is it needed?
            # for i in range(self.__drop_ifg_index):
            # Beacause this process is made as small_baseline_flag = 'N' we don't have more here.
            # But in StaMPS'is there was

        return coh_thresh_ind

    def __get_ph_patch(self, coh_thresh_ind: np.ndarray, data: __DataDTO):

        CACHE_FILE_NAME = "tmp_ph_patch"

        def get_max_min(ps_ij_col: np.ndarray, nr_ij: int):
            min_val = max(ps_ij_col - self.__clap_win / 2, 0)
            max_val = min_val + self.__clap_win - 1

            if max_val >= nr_ij:
                min_val = min_val - max_val + nr_ij - 1
                max_val = nr_ij - 1

            return int(min_val), int(max_val)

        def get_ph_bit_ind_array(ps_bit_col: int, ph_bit_len):
            slc_osf = self.__slc_osf - 1
            ind_array = ArrayUtils.arange_include_last(start=ps_bit_col - slc_osf,
                                                       end=ps_bit_col + slc_osf).astype(np.int32)
            ind_array = ind_array[(ind_array > 0) & (0 <= ph_bit_len)]

            # Python can't take anything from empty/ no values list
            if len(ind_array) == 0:
                ind_array = np.zeros(1).astype(np.int16)

            return ind_array

        def ph_path_loop():
            # In StaMPS this was the place where to delete 'ph_res' and 'ph_patch' that were found
            # from last process

            NR_PS = len(coh_thresh_ind)
            ph_patch = self.__zero_ph_array(NR_PS, data.nr_ifgs)

            # Similar logic with 'nr_i' ja 'nr_j' already exists in PsEstGamma process
            nr_i = MatlabUtils.max(self.__ps_est_gamma.grid_ij[:, 0])
            nr_j = MatlabUtils.max(self.__ps_est_gamma.grid_ij[:, 1])

            # In StaMPS this variable had '2' at the end of the name
            ph_filt = np.zeros((self.__clap_win, self.__clap_win, data.nr_ifgs), np.complex128)

            for i in range(ph_patch.shape[0]):
                ps_ij = self.__ps_est_gamma.grid_ij[coh_thresh_ind[i], :]

                i_min, i_max = get_max_min(ps_ij[0] - 1, nr_i)
                j_min, j_max = get_max_min(ps_ij[1] - 1, nr_j)

                # If you don't make copy then changes are made also in ph_grid variable
                ph_bit = np.copy(self.__ps_est_gamma.ph_grid[i_min:i_max + 1, j_min:j_max + 1, :])

                ps_bit_i = int(ps_ij[0] - i_min - 1)
                ps_bit_j = int(ps_ij[1] - j_min - 1)
                ph_bit[ps_bit_i, ps_bit_j, :] = 0

                # todo Some kind of JJS oversample update
                ph_bit_len = len(ph_bit) + 1
                ph_bit_ind_i = get_ph_bit_ind_array(ps_bit_i, ph_bit_len)
                ph_bit_ind_j = get_ph_bit_ind_array(ps_bit_j, ph_bit_len)
                ph_bit[ph_bit_ind_i, ph_bit_ind_j, 0] = 0

                # It is similar with PsEstGammas ph_flit process but still not the same
                for j in range(ph_patch.shape[1]):
                    ph_filt[:, :, j] = self.__clap_filt_for_patch(ph_bit[:, :, j],
                                                                  self.__ps_est_gamma.low_pass)

                ph_patch[i, :] = np.squeeze(ph_filt[ps_bit_i, ps_bit_j, :])

            return ph_patch

        if self.__PH_PATCH_CACHE:
            try:
                self.__logger.debug("Trying to use cache")
                loaded = ProcessCache.get_from_cache(CACHE_FILE_NAME, 'ph_patch', 'coh_thresh_ind')
                if np.array_equal(coh_thresh_ind, loaded['coh_thresh_ind']):
                    self.__logger.debug("Using cache")
                    ph_patch = loaded['ph_patch']
                else:
                    self.__logger.debug("No usable cache")
                    ph_patch = ph_path_loop()
                    ProcessCache.save_to_cache(CACHE_FILE_NAME,
                                               ph_patch=ph_patch,
                                               coh_thresh_ind=coh_thresh_ind)
            except FileNotFoundError:
                self.__logger.debug("No cache")
                ph_patch = ph_path_loop()
                ProcessCache.save_to_cache(CACHE_FILE_NAME,
                                           ph_patch=ph_patch,
                                           coh_thresh_ind=coh_thresh_ind)
        else:
            self.__logger.debug("Not using cache")
            ph_patch = ph_path_loop()

        return ph_patch

    def __clap_filt_for_patch(self, ph, low_pass):
        """Combined Low-pass Adaptive Phase filtering on 1 patch.
        In StaMPS this was in separate function clap_filt_patch"""

        alpha = self.__clap_alpha
        beta = self.__clap_beta

        if len(low_pass) == 0:
            low_pass = np.zeros(len(ph))

        ph = np.nan_to_num(ph)

        # todo This ph_fft its very similar with PhEstGamma function clap_filt
        # todo There where problems (incorrect value) with this part when calling third time
        ph_fft = np.fft.fft2(ph)
        smooth_resp = np.abs(ph_fft)
        smooth_resp = np.fft.ifftshift(
            MatlabUtils.filter2(self.__gaussian_window, np.fft.fftshift(smooth_resp)))
        smooth_resp_mean = np.median(smooth_resp.flatten())

        if smooth_resp_mean != 0:
            smooth_resp /= smooth_resp_mean

        smooth_resp = np.power(smooth_resp, alpha)

        smooth_resp -= 1
        smooth_resp[smooth_resp < 0] = 0

        G = smooth_resp * beta + low_pass
        ph_filt = np.fft.ifft2(np.multiply(ph_fft, G))

        return ph_filt

    def __topofit(self, ph_patch, coh_thresh_ind, data) -> (np.ndarray, PsTopofit):
        NR_PS = len(coh_thresh_ind)
        SW_ARRAY_SHAPE = (NR_PS, 1)

        ph = data.ph[coh_thresh_ind, :]
        bperp = self.__ps_files.bperp[coh_thresh_ind]

        topofit = PsTopofit(SW_ARRAY_SHAPE, NR_PS, data.nr_ifgs)
        topofit.ps_topofit_loop(ph, ph_patch, bperp, self.__ps_est_gamma.nr_trial_wraps,
                                data.ifg_ind)

        # In StaMPS old value was overridden here
        coh_ps = self.__ps_est_gamma.coh_ps.copy()
        coh_ps[coh_thresh_ind] = topofit.coh_ps

        return coh_ps, topofit

    def __get_keep_ind(self, coh_ps : np.ndarray, coh_thresh : np.ndarray,
                       coh_thresh_ind : np.ndarray, k_ps2: np.ndarray) -> np.ndarray:
        """In Stamps variable was named 'keep_ix'"""

        bperp_meaned = self.__ps_files.bperp_meaned
        k_ps = self.__ps_est_gamma.k_ps[coh_thresh_ind]
        bperp_delta = np.max(bperp_meaned) - np.min(bperp_meaned)

        # Reshape is needed because otherwise we get array in array
        coh_ps_len = len(coh_ps)
        coh_ps_reshaped = coh_ps.reshape(coh_ps_len)
        delta = (np.abs(k_ps - k_ps2) < 2 * np.pi / bperp_delta).reshape(coh_ps_len) #todo parem nimi
        keep_ind = np.where((coh_ps_reshaped > coh_thresh) & delta)[0]

        return keep_ind

    # todo Maybe some other solution here? PsEstGamma has already this kind of logic.
    def __zero_ps_array(self, shape):
        """Constuctor for making empty array for persistent scatterers data"""
        return np.zeros(shape)

    # TODO: Why not the same as in PsEstGamma?
    def __zero_ph_array(self, nr_ps, nr_ifgs):
        return np.zeros((nr_ps, nr_ifgs), np.complex128)
