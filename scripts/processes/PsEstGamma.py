import math
import os

import pydsm.relab
import sys
from statsmodels.compat import scipy
import scipy.signal

from scripts.MetaSubProcess import MetaSubProcess
from scripts.processes.PsFiles import PsFiles

import numpy as np

from scripts.utils.ArrayUtils import ArrayUtils
from scripts.utils.FolderConstants import FolderConstants
from scripts.utils.LoggerFactory import LoggerFactory
from scripts.utils.MatlabUtils import MatlabUtils
from scripts.utils.MatrixUtils import MatrixUtils
from scripts.utils.ProcessDataSaver import ProcessDataSaver


class PsEstGamma(MetaSubProcess):
    """Analüüsitakse võimalike püsivpeegeldajate faasimüra"""

    low_pass = np.ndarray
    weights = np.ndarray
    weights_org = np.ndarray
    rand_dist = np.ndarray
    nr_max_nz_ind = -1

    __FILE_NAME = "ps_est_gamma"

    def __init__(self, ps_files: PsFiles, rand_dist_cached=False) -> None:
        self.__logger = LoggerFactory.create("PsEstGamma")

        self.ps_files = ps_files
        self.__set_internal_params()
        self.rand_dist_cached = rand_dist_cached

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

        self.__low_coherence_tresh = 31  # Võrdne 31/100'jaga

    def start_process(self):
        self.__logger.debug("Started")

        self.low_pass = self.__get_low_pass()

        ph, bperp_meaned, bperp, nr_ifgs, nr_ps, xy, da, sort_ind_meaned = self.__load_ps_params()

        nr_trial_waps = self.__get_nr_trial_wraps(bperp_meaned, sort_ind_meaned)

        # StaMPS'is oli self.rand_dist nimetatud 'Nr'
        self.rand_dist, self.nr_max_nz_ind = self.__make_random_dist(nr_ps, nr_ifgs, bperp_meaned,
                                                                     nr_trial_waps)

        grid_ij = self.__get_grid_ij(xy)

        self.weights_org = self.__get_weights(da)

        weights = np.array(self.weights_org, copy=True)

        # Eelnev oli sisuliselt eelöö selleks mis nüüd hakkab.
        self.ph_patch, self.k_ps, self.c_ps, self.coh_ps, self.n_opt, \
        self.ph_res, self.ph_grid, self.low_pass = \
            self.__sw_loop(
                grid_ij, ph,
                weights.copy(),
                self.low_pass,
                bperp,
                nr_ifgs,
                nr_ps,
                nr_trial_waps)

        self.__logger.debug("End")

    def save_results(self):
        ProcessDataSaver(FolderConstants.SAVE_PATH, self.__FILE_NAME).save_data(
            ph_patch=self.ph_patch,
            k_ps=self.k_ps,
            c_ps=self.c_ps,
            coh_ps=self.coh_ps,
            n_opt=self.n_opt,
            ph_res=self.ph_res,
            ph_grid=self.ph_grid,
            low_pass=self.low_pass
        )

    def load_results(self):
        file_with_path = os.path.join(FolderConstants.SAVE_PATH, self.__FILE_NAME + ".npz")
        data = np.load(file_with_path)

        self.ph_patch = data['ph_patch']
        self.k_ps = data['k_ps']
        self.c_ps = data['c_ps']
        self.coh_ps = data['coh_ps']
        self.n_opt = data['n_opt']
        self.ph_res = data['ph_res']
        self.ph_grid = data['ph_grid']
        self.low_pass = data['low_pass']

    def __get_low_pass(self):
        start = -(self.__clap_win) / self.__filter_grid_size / self.__clap_win / 2
        stop = (self.__clap_win - 2) / self.__filter_grid_size / self.__clap_win / 2
        step = 1 / self.__filter_grid_size / self.__clap_win
        freg_i = ArrayUtils.arange_include_last(start, stop, step)

        freg_0 = 1 / self.__clap_low_pass_wavelength
        subtract = np.power(1 + (freg_i / freg_0), 10)
        butter_i = np.asmatrix(np.divide(1, subtract))
        return np.fft.fftshift(butter_i.transpose() * butter_i)

    def __load_ps_params(self):
        """Loeb sisse muutujad ps_files'ist ja muudab neid vastavalt"""

        ph, bperp, nr_ifgs, nr_ps, xy, da = self.ps_files.get_ps_variables()

        ph = MatrixUtils.delete_master_col(ph, self.ps_files.master_ix)
        ph_abs = np.abs(ph)
        ph = np.divide(ph_abs, ph)

        # bprep_meaned on massiiv ridadest, mitte veergudest
        # siis tavaline delete_master_col ei tööta
        bprep_meaned = np.delete(self.ps_files.bperp_meaned, self.ps_files.master_ix - 1)


        sort_ind_meaned = np.mean(self.ps_files.sort_ind + 1) + math.radians(3)

        return ph, bprep_meaned, bperp, nr_ifgs, nr_ps, xy, da, sort_ind_meaned

    def __get_nr_trial_wraps(self, bperp_meaned, sort_ind_meaned):
        # todo mis on k?
        k = self.ps_files.wavelength * self.__mean_range * np.sin(sort_ind_meaned) / 4 / math.pi
        max_k = self.__max_topo_err / k

        bperp_range = MatlabUtils.max(bperp_meaned) - MatlabUtils.min(bperp_meaned)

        # todo kas miks see max_k / (2 * math.pi)?
        return bperp_range * max_k / (2 * math.pi)

    def __make_random_dist(self, nr_ps, nr_ifgs, bperp_meaned, nr_trial_wraps):

        def use_cached():
            try:
                loaded = np.load(FolderConstants.SAVE_PATH + "/tmp/tmp_rand_dist.npz")
                rand_dist = loaded['rand_dist']
                nr_max_nz_ind = loaded['nr_max_nz_ind']
            except FileNotFoundError:
                self.__logger.info("No cache")

                rand_dist, nr_max_nz_ind = random_dist()
                cache(rand_dist, nr_max_nz_ind)

            return rand_dist, nr_max_nz_ind

        def cache(rand_dist: np.ndarray, nr_max_nz_ind: int):
            ProcessDataSaver(FolderConstants.SAVE_PATH + "/tmp", "tmp_rand_dist").save_data(
                rand_dist=rand_dist,
                nr_max_nz_ind=nr_max_nz_ind)

        def random_dist():
            NR_RAND_IFGS = nr_ps  # StaMPS'is oli see 300000
            random = np.random.RandomState(2005)

            rnd_ifgs = np.array(2 * math.pi * random.rand(NR_RAND_IFGS, nr_ifgs), np.complex64)

            random_coherence = np.zeros((NR_RAND_IFGS, 1))
            for i in range(NR_RAND_IFGS - 1, 0, -1):
                phase = np.exp(1j * rnd_ifgs[i])
                # Siin juhul kasutame ainult esimest parameetrit
                phase_residual, _, _, _ = self.__ps_topofit(phase,
                                                                                    bperp_meaned,
                                                                                    nr_trial_wraps)
                random_coherence[i] = phase_residual[0]

            # todo
            del rnd_ifgs

            hist, _ = MatlabUtils.hist(random_coherence, self.coherence_bins)

            rand_dist = hist

            return rand_dist, np.count_nonzero(hist)

        if (self.rand_dist_cached):
            self.__logger.info("Using cache")
            return use_cached()
        else:
            return random_dist()

    def __ps_topofit(self, phase: np.ndarray, bperp_meaned: np.ndarray, nr_trial_wraps: float):
        def fiter_zeros(phase, bperp):
            not_zeros_ind = np.nonzero(np.nan_to_num(phase))

            phase = phase[not_zeros_ind]
            bperp = bperp[not_zeros_ind]

            return phase, bperp

        # phase, bperp_meaned = fiter_zeros(phase, bperp_meaned)

        # Et edasipidi oleksid tulemid õiged teeme bperp'i veerumaatriksiks
        bperp_meaned = ArrayUtils.to_col_matrix(bperp_meaned)

        # Siin ei saa get_nr_trial_wraps leitut kasutada, kuna seal oli see üldisem
        bperp_range = np.amax(bperp_meaned) - np.amin(bperp_meaned)

        CONST = 8 * nr_trial_wraps  # todo aga mis const? Miks see 8 on?
        trial_multi_start = -np.ceil(CONST)
        trial_multi_end = np.ceil(CONST)
        trial_multi = ArrayUtils.arange_include_last(trial_multi_start, trial_multi_end, 1)

        trial_phase = bperp_meaned / bperp_range * math.pi / 4

        # Tavaline korrutamine võib anda vigu, seepärast on siin np.outer
        trial_phase = np.exp(np.outer(-1j * trial_phase, trial_multi))

        # Selleks, et korrutamine õnnestuks teeme ta veeruvektoriks
        phase = ArrayUtils.to_col_matrix(phase)
        phase_tile = np.tile(phase, (1, len(trial_multi)))
        phaser = np.multiply(trial_phase, phase_tile)

        phaser_sum = MatlabUtils.sum(phaser)

        phase_abs_sum = MatlabUtils.sum(np.abs(phase))
        trial_coherence = np.abs(phaser_sum) / phase_abs_sum
        trial_coherence_max_ind = np.where(trial_coherence == MatlabUtils.max(trial_coherence))

        # todo: kas siin on viga? trial_coherence_max_ind on tuple ju
        k_0 = (math.pi / 4 / bperp_range) * trial_multi[trial_coherence_max_ind][0]

        re_phase = np.multiply(phase, np.exp(-1j * (k_0 * bperp_meaned)))
        phase_offset = MatlabUtils.sum(re_phase)
        re_phase = np.angle(re_phase * phase_offset.conjugate())
        weigth = np.abs(phase)
        mopt = np.multiply(weigth, bperp_meaned) / np.multiply(weigth, re_phase)
        k_0 = k_0 + mopt

        phase_residual = np.multiply(phase, np.exp(-1j * (k_0 * bperp_meaned)))
        phase_residual_sum = MatlabUtils.sum(phase_residual)
        static_offset = np.angle(phase_residual_sum)
        coherence_0 = np.abs(phase_residual_sum) / MatlabUtils.sum(np.abs(phase_residual))

        return phase_residual, coherence_0, static_offset, k_0

    def __get_grid_ij(self, xy: np.ndarray):

        def fill_cols_with_xy_values(xy_col):
            col_formula = lambda x: np.ceil((x - np.amin(x) + 1e-6) / self.__filter_grid_size)

            grid_ij_col = col_formula(xy_col)
            max_ind = np.where(grid_ij_col == np.amax(grid_ij_col))
            grid_ij_col[max_ind] -= 1

            return grid_ij_col

        grid_ij = np.zeros((len(xy), 2))

        grid_ij[:, 0] = fill_cols_with_xy_values(xy[:, 0])
        grid_ij[:, 1] = fill_cols_with_xy_values(xy[:, 1])

        return grid_ij

    def __get_weights(self, da):
        return ArrayUtils.to_col_matrix(np.divide(1, da))

    def __sw_loop(self, grid_ij: np.ndarray, ph: np.ndarray, weights: np.ndarray,
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
            return abs(gamma_change_delta) > self.__gamma_change_convergence

        nr_i = int(np.max(grid_ij[:, 0]))
        nr_j = int(np.max(grid_ij[:, 1]))
        # Liidame siin ühe juurde, sest indeksid hakkavad nullist
        # Tüüp peab olema np.complex64, sest pydsm.relab.shiftdim tagastab selles tüübis
        ph_grid = np.zeros((nr_j + 1, nr_i + 1, nr_ifgs), np.complex128)
        ph_filt = ph_grid.copy()

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

        while is_gamma_in_change_delta():
            ph_weight = get_ph_weight(bprep, k_ps, nr_ifgs, ph, weights)

            for i in range(nr_ps):
                ph_grid[int(grid_ij[i, 1]), int(grid_ij[i, 0]), :] += pydsm.relab.shiftdim(
                    ph_weight[i, :], -1, nargout=1)[0]

            for i in range(nr_ifgs):
                ph_filt[:, :, i] = self.__clap_filt(ph_grid[:, :, 1], low_pass)

            for i in range(nr_ps):
                ph_patch[i, :nr_ifgs] = np.squeeze(
                    ph_filt[int(grid_ij[i, 1]), int(grid_ij[i, 0]), :])

            not_zero_patches_ind = np.nonzero(ph_patch)
            ph_patch[not_zero_patches_ind] = np.divide(ph_patch[not_zero_patches_ind],
                                                       np.abs(ph_patch[not_zero_patches_ind]))

            # Est topo error
            # todo siin me puhastame need array'd, refacto
            c_ps = zero_ps_array_cont()
            coh_ps = zero_ps_array_cont()
            n_opt = zero_ps_array_cont()
            ph_res = np.zeros((nr_ps, nr_ifgs))
            for i in range(nr_ps):
                psdph = np.multiply(ph[i, :], np.conjugate(ph_patch[i, :]))

                # todo refactor
                if np.sum(np.isnan(psdph)) == 0 and np.sum(psdph == 0) == 0:
                    phase_residual, coh_0, static_offset, k_0 = self.__ps_topofit(
                        psdph, bprep[i, :].transpose(), nr_trial_wraps)
                    k_ps[i] = k_0[0]
                    c_ps[i] = static_offset[0]
                    coh_ps[i] = coh_0[0]
                    n_opt[i] = len(k_0)
                    ph_res[i, :] = np.angle(phase_residual).transpose()
                else:
                    k_ps[i] = np.nan
                    coh_ps[i] = 0

            gamma_change_rms = np.sqrt(np.sum(np.power(coh_ps - coh_ps_result, 2) / nr_ps))
            gamma_change_delta = gamma_change_rms - gamma_change
            # Salvestame ajutistesse muutujatesse gamma ja koherentsuse
            gamma_change = gamma_change_rms
            coh_ps_result = coh_ps

            if is_gamma_in_change_delta() or self.__filter_weighting == 'P-square':
                hist, _ = MatlabUtils.hist(coh_ps, self.coherence_bins)
                # todo Juhuslikud sagedused tehakse reaalseteks. Mida iganes see ka ei tähenda
                self.rand_dist = self.rand_dist * np.sum(
                    hist[1:self.__low_coherence_tresh]) / np.sum(
                    self.rand_dist[1:self.__low_coherence_tresh])

                hist[hist == 0] = 1
                p_rand = np.divide(self.rand_dist, hist)
                p_rand[1:self.__low_coherence_tresh] = 1
                p_rand[self.nr_max_nz_ind + 1:] = 0
                p_rand[p_rand > 1] = 1
                n_dimension = np.append(np.ones((1, 7)), p_rand) / np.sum(MatlabUtils.gausswin(7))
                p_rand = scipy.signal.lfilter(MatlabUtils.gausswin(7), 1, n_dimension)
                p_rand = p_rand[8:]

                p_rand = MatlabUtils.interp(np.append(1, p_rand), 10)[:-9]

                #todo kas siin on vaja seda reshape'de asja?

                # Reshape sest zero_ps_array_cont, mis on massiiv masiividest, ei sobi
                coh_ps_int = np.reshape(coh_ps, (coh_ps.size)).astype(np.int)
                ps_rand = p_rand[np.round(coh_ps_int * 1000)].conj().transpose()

                weights = np.reshape(np.power(1 - ps_rand, 2), SW_ARRAY_SHAPE)

        #todo k_ps üle vaadata
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

        FILTERED_TYPE = np.complex128
        filtered = np.zeros(ph.shape, FILTERED_TYPE)

        ph = np.nan_to_num(ph)

        nr_win = int(self.__clap_win * 0.75)
        nr_pad = int(self.__clap_win * 0.25)

        ph_i_len = ph.shape[0] - 1
        ph_j_len = ph.shape[1] - 1
        nr_inc = int(np.floor(nr_win / 4))
        nr_win_i = int(np.ceil(ph_i_len / nr_inc) - 3)
        nr_win_j = int(np.ceil(ph_j_len / nr_inc) - 3)

        wind_func = make_wind_func(create_grid(nr_win))

        # Selleks, et transponeerimine töötaks nagu Matlab'is teeme need maatriksiteks
        B = np.asmatrix(MatlabUtils.gausswin(7)) * np.asmatrix(MatlabUtils.gausswin(7)).transpose()

        nr_win_pad_sum = (nr_win + nr_pad)
        ph_bit = np.zeros((nr_win_pad_sum, nr_win_pad_sum), FILTERED_TYPE)
        # Todo: Refactor
        for i in range(nr_win_i):
            w_f = wind_func.copy()
            i1 = i * nr_inc
            i2 = i1 + nr_win

            if i2 > ph_i_len:
                i_shift = i2 - ph_i_len
                i2 = ph_i_len
                i1 = ph_i_len - nr_win
                w_f = np.append(np.zeros((i_shift, nr_win)), w_f[:nr_win - i_shift, :],
                                axis=0).astype(FILTERED_TYPE)

            for j in range(nr_win_j):
                w_f2 = w_f
                j1 = j * nr_inc
                j2 = j1 + nr_win

                if j2 > ph_j_len:
                    j_shift = j2 - ph_j_len
                    j2 = ph_j_len
                    j1 = ph_j_len - nr_win
                    w_f2 = np.append(np.zeros((nr_win, j_shift)), w_f2[:, :nr_win - j_shift],
                                     axis=1).astype(FILTERED_TYPE)

                ph_bit[:nr_win, :nr_win] = ph[i1: i2, j1: j2]

                ph_fft = np.fft.fft2(ph_bit)
                smooth_resp = np.abs(ph_fft)
                smooth_resp = np.fft.ifftshift(
                    scipy.signal.convolve2d(B, np.fft.ifftshift(smooth_resp)))
                mean_smooth_resp = np.median(smooth_resp)

                if mean_smooth_resp != 0:
                    smooth_resp /= mean_smooth_resp

                smooth_resp = np.power(smooth_resp, self.__clap_alpha)

                # todo Väärtused alla mediaani nulliks. Milleks see osa?
                smooth_resp -= 1
                smooth_resp[smooth_resp < 0] = 0

                # todo mida tähistab G?
                G = smooth_resp * self.__clap_beta + low_pass
                ph_filt = np.fft.fft2(np.multiply(ph_fft, G))
                ph_filt = np.multiply(ph_filt[:nr_win, :nr_win], w_f2)

                filtered[i1:i2, j1:j2] += ph_filt

        return filtered
