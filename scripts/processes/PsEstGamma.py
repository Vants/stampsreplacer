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
from scripts.utils.LoggerFactory import LoggerFactory
from scripts.utils.MatlabUtils import MatlabUtils
from scripts.utils.MatrixUtils import MatrixUtils
from scripts.utils.ProcessCache import ProcessCache
from scripts.utils.ProcessDataSaver import ProcessDataSaver


class PsEstGamma(MetaSubProcess):
    """Analüüsitakse võimalike püsivpeegeldajate faasimüra"""

    low_pass = np.ndarray
    weights = np.ndarray
    weights_org = np.ndarray
    rand_dist = np.ndarray
    nr_max_nz_ind = -1

    __FILE_NAME = "ps_est_gamma"

    def __init__(self, ps_files: PsFiles, rand_dist_cached_file=False,
                 outter_rand_dist=np.array([])) -> None:
        """rand_dist_cached_file=True laeb eelnevalt leitud massiivi juhuslikkest arvudest failist
        'tmp_rand_dist'. Täpsem loogika meetodis 'self.__make_random_dist'.
        outter_rand_dist on massiiv juhuslikkest arvudest, mis muidu leitakse meetodiga
        'self.__make_random_dist'. Eelkõige kasutatakse testimisekss"""

        self.__logger = LoggerFactory.create("PsEstGamma")

        self.ps_files = ps_files
        self.__set_internal_params()
        self.rand_dist_cached = rand_dist_cached_file
        self.outter_rand_dist = outter_rand_dist

        # StaMPS'is oli see 'coh_bins'
        self.coherence_bins = ArrayUtils.arange_include_last(0.005, 0.995, 0.01)

    def __set_internal_params(self):
        """StaMPS'is loeti need setparam'iga süsteemi. Väärtused on saadud ps_params_default failist"""

        # Todo Pixel size of grid (meetrites)
        self.__filter_grid_size = 50
        # Todo Weighting scheme (PS probability squared)
        self.__filter_weighting = 'P-square'
        # Todo CLAP (Combined Low-pass and Adaptive Phase) libiseva akna suurus.
        self.__clap_win = 32
        # Todo Selllest suuremad lainepikkused pääsevad läbi
        self.__clap_low_pass_wavelength = 800

        self.__clap_alpha = 1
        self.__clap_beta = 0.3
        # todo Maksimaalne DEM'i (digitaalse kõrgusmudeli) viga (meetrites).
        # Sellest suurema väärtusega ei arvestata
        self.__max_topo_err = 5
        # Todo Lubatud muutuste keskmine väärtus. Koherentsuse sarnane väärtus.
        self.__gamma_change_convergence = 0.005
        # todo mean range - need only be approximately correct
        self.__mean_range = 830000

        self.__low_coherence_thresh = 31  # Võrdne 31/100'jaga

    def start_process(self):
        self.__logger.info("Started")

        self.low_pass = self.__get_low_pass()
        self.__logger.debug("low_pass.len: {0}".format(len(self.low_pass)))

        ph, bperp_meaned, bperp, nr_ifgs, nr_ps, xy, da, sort_ind_meaned = self.__load_ps_params()

        self.nr_trial_wraps = self.__get_nr_trial_wraps(bperp_meaned, sort_ind_meaned)
        self.__logger.debug("nr_trial_wraps: {0}".format(self.nr_trial_wraps))

        # StaMPS'is oli self.rand_dist nimetatud 'Nr'
        self.rand_dist, self.nr_max_nz_ind = self.__make_random_dist(nr_ps, nr_ifgs, bperp_meaned,
                                                                     self.nr_trial_wraps)
        self.__logger.debug("rand_dist.len: {0}, self.nr_max_nz_ind: {1}"
                            .format(len(self.rand_dist), self.nr_max_nz_ind))


        self.grid_ij = self.__get_grid_ij(xy)
        self.__logger.debug("grid_ij.len: {0}".format(len(self.grid_ij)))

        self.weights_org = self.__get_weights(da)
        self.__logger.debug("weights_org.len: {0}".format(len(self.weights_org)))

        weights = np.array(self.weights_org, copy=True) #todo miks kaks korda copy? sw_loop'i pannakse ka copy sellest

        # Eelnev oli sisuliselt eelöö selleks mis nüüd hakkab.
        self.ph_patch, self.k_ps, self.c_ps, self.coh_ps, self.n_opt, \
        self.ph_res, self.ph_grid, self.low_pass = \
            self.__sw_loop(
                ph,
                weights.copy(),
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
            rand_dist = self.rand_dist,
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
        """Loeb sisse muutujad ps_files'ist ja muudab neid vastavalt"""

        ph, bperp, nr_ifgs, nr_ps, xy, da = self.ps_files.get_ps_variables()
        nr_ifgs -= 1 # Teistes protsessides kus seda muutjat kasutatakse seda teha ei tohi. StaMPS'is small_basline=n

        ph = MatrixUtils.delete_master_col(ph, self.ps_files.master_nr)
        ph_abs = np.abs(ph)
        ph_abs[np.where(ph_abs == 0)] = 1 # Selleks, et nulliga jagamine välistada
        ph = np.divide(ph, ph_abs)

        # bprep_meaned on massiiv ridadest, mitte veergudest
        # siis tavaline delete_master_col ei tööta
        bprep_meaned = np.delete(self.ps_files.bperp_meaned, self.ps_files.master_nr - 1)

        # Matlab'is oli 0.052 selle math.radians(3) asemel
        sort_ind_meaned = np.mean(self.ps_files.sort_ind) + math.radians(3)

        return ph, bprep_meaned, bperp, nr_ifgs, nr_ps, xy, da, sort_ind_meaned

    def __get_nr_trial_wraps(self, bperp_meaned, sort_ind_meaned):
        # todo mis on k?
        k = self.ps_files.wavelength * self.__mean_range * np.sin(sort_ind_meaned) / 4 / math.pi
        max_k = self.__max_topo_err / k

        bperp_range = MatlabUtils.max(bperp_meaned) - MatlabUtils.min(bperp_meaned)

        # todo kas miks see max_k / (2 * math.pi)?
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
            NR_RAND_IFGS = nr_ps  # StaMPS'is oli see 300000
            random = np.random.RandomState(2005)

            rnd_ifgs = np.array(2 * math.pi * random.rand(NR_RAND_IFGS, nr_ifgs), np.complex64)

            random_coherence = np.zeros((NR_RAND_IFGS, 1))
            for i in range(NR_RAND_IFGS - 1, 0, -1):
                phase = np.exp(1j * rnd_ifgs[i])
                # Siin juhul kasutame ainult esimest parameetrit
                phase_residual, _, _, _ = PsTopofit.ps_topofit_fun(phase, bperp_meaned, nr_trial_wraps)
                random_coherence[i] = phase_residual[0]

            # todo
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

        def fill_cols_with_xy_values(xy_col):
            # Float32 on vajalik selleks, et komakohti ei tekiks liiga palju. Kui komakohti on
            # liiga palju siis võib juhtuda, et arvud on näiteks 2224.000001, mis ümardatakse 2225'eni
            col_formula = lambda x: np.ceil((x - np.amin(x) + 1e-6).astype(np.float32) / self.__filter_grid_size)

            grid_ij_col = col_formula(xy_col)
            max_ind = np.where(grid_ij_col == np.amax(grid_ij_col))
            grid_ij_col[max_ind] -= 1

            return grid_ij_col

        grid_ij = np.zeros((len(xy), 2), np.int32)

        grid_ij[:, 0] = fill_cols_with_xy_values(xy[:, 1])
        grid_ij[:, 1] = fill_cols_with_xy_values(xy[:, 0])

        return grid_ij

    def __get_weights(self, da):
        return ArrayUtils.to_col_matrix(np.divide(1, da))

    def __sw_loop(self, ph: np.ndarray, weights: np.ndarray,
                  low_pass: np.ndarray, bprep: np.ndarray, nr_ifgs: int, nr_ps: int,
                  nr_trial_wraps: float):

        SW_ARRAY_SHAPE = (nr_ps, 1)

        # Konstruktor tühja pusivpeegeladajate info massiivi loomiseks
        # TODO: on siin vaja sellist suurust?
        def zero_ps_array_cont():
            return np.zeros(SW_ARRAY_SHAPE)

        def get_ph_weight(bprep, k_ps, nr_ifgs, ph, weights):
            exped = np.exp(np.multiply((-1j * bprep), np.tile(k_ps, (1, nr_ifgs))))
            exp_tiled_weight_multi = np.multiply(exped, np.tile(weights, (1, nr_ifgs)))
            return np.multiply(ph, exp_tiled_weight_multi)

        def is_gamma_in_change_delta():
            return abs(gamma_change_delta) < self.__gamma_change_convergence

        nr_i = int(np.max(self.grid_ij[:, 0]))
        nr_j = int(np.max(self.grid_ij[:, 1]))

        coh_ps_result = zero_ps_array_cont()
        gamma_change = 0
        gamma_change_delta = sys.maxsize

        ph_patch = np.zeros(ph.shape, np.complex128)

        k_ps = zero_ps_array_cont()

        # Est topo error
        # todo: siin me loome muutujad, et returnida, refacto
        c_ps = zero_ps_array_cont()
        coh_ps = zero_ps_array_cont()
        n_opt = zero_ps_array_cont()
        ph_res = np.zeros((nr_ps, nr_ifgs))

        log_i = -1 # Logimiseks int, et näha mitmendat tiiru tehakse
        self.__logger.debug("is_gamma_in_change_delta loop begin")
        while not is_gamma_in_change_delta():
            log_i += 1
            self.__logger.debug("gamma change loop i " + str(log_i))
            ph_weight = get_ph_weight(bprep, k_ps, nr_ifgs, ph, weights)

            # Tüüp peab olema np.complex128, sest pydsm.relab.shiftdim tagastab selles tüübis
            # ph_grid'i ja ph_filt peab uuesti looma. Vastasel juhul jäävad vanad tulemused sisse
            # ja väärtused on valed
            ph_grid = np.zeros((nr_i, nr_j, nr_ifgs), np.complex128)

            for i in range(nr_ps):
                x_ind = int(self.grid_ij[i, 0]) - 1
                y_ind = int(self.grid_ij[i, 1]) - 1
                ph_grid[x_ind, y_ind, :] += pydsm.relab.shiftdim(ph_weight[i, :], -1, nargout=1)[0]

            ph_filt = np.zeros((nr_i, nr_j, nr_ifgs), np.complex128)
            for i in range(nr_ifgs):
                ph_filt[:, :, i] = self.__clap_filt(ph_grid[:, :, i], low_pass)

            self.__logger.debug("ph_filt found. first row: {0}, last row: {1}"
                                .format(ph_filt[0], ph_filt[len(ph_filt) - 1]))

            for i in range(nr_ps):
                x_ind = int(self.grid_ij[i, 0]) - 1
                y_ind = int(self.grid_ij[i, 1]) - 1
                ph_patch[i, :nr_ifgs] = np.squeeze(ph_filt[x_ind, y_ind, :])

            not_zero_patches_ind = np.nonzero(ph_patch)
            ph_patch[not_zero_patches_ind] = np.divide(ph_patch[not_zero_patches_ind],
                                                       np.abs(ph_patch[not_zero_patches_ind]))

            self.__logger.debug("ph_patch found. first row: {0}, last row: {1}"
                                .format(ph_patch[0], ph_patch[len(ph_patch) - 1]))

            topofit = PsTopofit(SW_ARRAY_SHAPE, nr_ps, nr_ifgs)
            topofit.ps_topofit_loop(ph, ph_patch, bprep, nr_trial_wraps)

            k_ps = topofit.k_ps
            c_ps = topofit.c_ps
            coh_ps = topofit.coh_ps
            n_opt = topofit.n_opt
            ph_res = topofit.ph_res

            self.__logger.debug("topofit found")

            gamma_change_rms = np.sqrt(np.sum(np.power(coh_ps - coh_ps_result, 2) / nr_ps))
            gamma_change_delta = gamma_change_rms - gamma_change
            # Salvestame ajutistesse muutujatesse gamma ja koherentsuse
            gamma_change = gamma_change_rms
            coh_ps_result = coh_ps

            self.__logger.debug("is_gamma_in_change_delta() and self.__filter_weighting: "
                                + str(not is_gamma_in_change_delta() and self.__filter_weighting == 'P-square'))
            if not is_gamma_in_change_delta() and self.__filter_weighting == 'P-square':
                hist, _ = MatlabUtils.hist(coh_ps, self.coherence_bins) # Stamps'is oli see 'Na'
                self.__logger.debug("hist[0:3] " + str(hist[:3]))
                # Juhuslikud sagedused tehakse reaalseteks
                low_coh_thresh_ind = self.__low_coherence_thresh
                real_distr = np.sum(hist[:low_coh_thresh_ind]) / np.sum(
                    self.rand_dist[:low_coh_thresh_ind])
                self.rand_dist = self.rand_dist * real_distr

                hist[hist == 0] = 1
                p_rand = np.divide(self.rand_dist, hist)
                p_rand[:low_coh_thresh_ind] = 1
                p_rand[self.nr_max_nz_ind:] = 0 #Stampsis liideti nr_max_nz_ind'le üks juurde
                p_rand[p_rand > 1] = 1
                p_rand_added_ones = np.append(np.ones(7), p_rand)
                filtered = scipy.signal.lfilter(MatlabUtils.gausswin(7), [1], p_rand_added_ones)
                p_rand = filtered / np.sum(MatlabUtils.gausswin(7))
                p_rand = p_rand[7:]

                # Leidsin, et quadratic on natuke täpsem kui tavaline cubic
                p_rand = MatlabUtils.interp(np.append([1.0], p_rand), 10, 'quadratic')[:-9]

                # coh_ps'ist teeme indeksite massiivi. astype on vajalik sest Numpy jaoks peavad
                # indeksid olema int'id. reshape on vajalik seepärast, et coh_ps on massiiv
                # massiividest
                coh_ps_as_ind = np.round(coh_ps * 1000).astype(np.int)
                if len(coh_ps_as_ind.shape) > 1:
                    coh_ps_as_ind = coh_ps_as_ind.reshape(len(coh_ps))
                # Stampsis oli see 'Prand_ps'
                ps_rand = p_rand[coh_ps_as_ind].conj().transpose()

                weights = np.reshape(np.power(1 - ps_rand, 2), SW_ARRAY_SHAPE)

        return ph_patch, k_ps, c_ps, coh_ps_result, n_opt, ph_res, ph_grid, low_pass

    def __clap_filt(self, ph: np.ndarray, low_pass: np.ndarray):
        """CLAP_FILT Combined Low-pass Adaptive Phase filtering. 
        Muutujad nr_win, nr_pad olid StaMPS'is sisendmuutujad ning need korrutati läbi
        enne funktsiooni poole pöördumist sama moodi nagu tehakse siin. 
        Samuti olid sisendmuutujad clap_alpha ja clap_beta, aga kuna need on globaalsed muutujad 
        siis ei teinud neid funkstiooni sisenditeks"""

        def create_grid(nr_win: int):
            # todo Kuidas siin see -1 asju katki ei tee?
            grid_array = ArrayUtils.arange_include_last(0, (nr_win / 2) - 1)
            grid_x, grid_y = np.meshgrid(grid_array, grid_array)
            grid = grid_x + grid_y

            return grid

        # todo Mida tähendab wind_func. See pole ju massiiv funkstioonidest
        def make_wind_func(grid: np.ndarray):
            WIND_FUNC_TYPE = np.float64
            wind_func = np.array(np.append(grid, np.fliplr(grid), axis=1), WIND_FUNC_TYPE)
            wind_func = np.array(np.append(wind_func, np.flipud(wind_func), axis=0), WIND_FUNC_TYPE)
            # Selleks, et äärtes ei läheks nulli
            wind_func += 1e-6

            return wind_func

        def get_indexes(loop_index: int, inc: int, nr_win: int) -> (int, int):
            i1 = loop_index * inc
            # Siin ei tee -1, sest kui me selekteerime massiivist siis viimast ei võeta
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
        # Kuna indeksid hakkavad 0'ist siis need väärtused on ühe võrra suuremad võrreldes Stamps'iga
        nr_win_i = int(np.ceil(ph_i_len / nr_inc) - 3)
        nr_win_j = int(np.ceil(ph_j_len / nr_inc) - 3) + 1

        wind_func = make_wind_func(create_grid(nr_win))

        # Selleks, et transponeerimine töötaks nagu Matlab'is teeme need maatriksiteks
        # todo: samasugune asi on juba PsSelect'is
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
                    # Kuna massiivi pikkus (ph_i_len ja ph_j_len) on niigi lühem ja numpy ei võta
                    # viimast numbrit selekteerimisel arvesse siis liidame ühe juurde indeksitele
                    # juurde
                    j2 = ph_j_len + 1
                    j1 = ph_j_len - nr_win + 1
                    w_f2 = np.append(np.zeros((nr_win, j_shift)), w_f2[:, :nr_win - j_shift],
                                     axis=1).astype(FILTERED_TYPE)

                ph_bit[:nr_win, :nr_win] = ph[i1: i2, j1: j2]

                ph_fft = np.fft.fft2(ph_bit) # todo viiendast komakohast lähevad tulemused vääraks
                # ph_fft = fftw.interfaces.numpy_fft.fft2(ph_bit) # todo viiendast komakohast lähevad tulemused vääraks
                smooth_resp = np.abs(ph_fft) # Stamps*is oli see 'H'
                smooth_resp = np.fft.ifftshift(
                    MatlabUtils.filter2(B, np.fft.ifftshift(smooth_resp)))
                # smooth_resp = fftw.interfaces.numpy_fft.ifftshift(
                #     MatlabUtils.filter2(B, fftw.interfaces.numpy_fft.ifftshift(smooth_resp)))
                mean_smooth_resp = np.median(smooth_resp)

                if mean_smooth_resp != 0:
                    smooth_resp /= mean_smooth_resp

                smooth_resp = np.power(smooth_resp, self.__clap_alpha)

                # todo Väärtused alla mediaani nulliks. Milleks see osa?
                smooth_resp -= 1
                smooth_resp[smooth_resp < 0] = 0

                # todo mida tähistab G?
                G = smooth_resp * self.__clap_beta + low_pass
                ph_filt = np.fft.ifft2(np.multiply(ph_fft, G))
                # ph_filt = fftw.interfaces.numpy_fft.ifft2(np.multiply(ph_fft, G))
                ph_filt = np.multiply(ph_filt[:nr_win, :nr_win], w_f2)

                filtered[i1:i2, j1:j2] += ph_filt

        return filtered
