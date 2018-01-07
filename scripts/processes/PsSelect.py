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
    """Stabiilsete pikslite valimine, et neist teha püsivpeegeldajad"""


    __B = np.array([])

    __FILE_NAME = "ps_select"

    def __init__(self, ps_files: PsFiles, ps_est_gamma: PsEstGamma):
        self.__PH_PATCH_CACHE = True
        self.__ps_files = ps_files
        self.__ps_est_gamma = ps_est_gamma

        self.__logger = LoggerFactory.create("PsSelect")

        self.__set_internal_params()

    def __set_internal_params(self):
        """StaMPS'is loeti need setparami'iga süteemi sisse ja pärast getparam'iga välja.
        Kõik väärtused on võetud, et small_baseline_flag on 'N'

        StaMPS'is oli max_desinty_rand ja max_percent_rand muutujad eraldi ja sarnaselt neile siin.
        Siin aga saadakse see funktsioonist __get_max_rand.
        """
        self.__slc_osf = 1
        self.__clap_alpha = 1
        self.__clap_beta = 0.3
        self.__clap_win = 32
        self.__select_method = self._SelectMethod.DESINTY  # DESINITY või PERCENT
        # todo mis see on?
        self.__gamma_stdev_reject = 0
        # TODO StaMPS'is oli []
        self.__drop_ifg_index = np.array([])
        self.__low_coh_tresh = 31  # 31/100

        self.__gaussian_window = np.multiply(np.asmatrix(MatlabUtils.gausswin(7)),
                                             np.asmatrix(MatlabUtils.gausswin(7)).conj().transpose())

    class __DataDTO(object):
        """Klass millega vahetada funkstioonide vahel muutujaid.
        Loodud eelõige seepärast, et näiteks self.ps_files.nr_ifgs on natuke liiga pikk kirjutada ja
        vahel on meil vaja enne nende kasutamist veel muuta (funkstioon load_ps_params).
        Siin objektis ei toimu töötlust, see on vaid andmete kapseldamiseks"""

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
        """Sisemise muutuja 'select_method' võimalikud väärtused"""
        DESINTY = 1
        PERCENT = 2

    def start_process(self):
        """Siin on tähtis, et min_coh, coh_thresh ja coh_thresh_ind oleksid leitud võimalikult täpselt.
        Seda seepärast, et ka 0.001'ne täpsus võib rikkuda coh_threh tulemust"""
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

        # Ja nüüd leitakse uue coh_ps'iga uuesti min_coh, da_mean ja coh_thresh (viimase jaoks on
        # tegelikult tarvis kahte esimest) uuesti

        min_coh, da_mean, is_min_coh_nan_array = self.__get_min_coh_and_da_mean(
            coh_ps, max_rand, data)
        self.__logger.debug("Second run min_coh.len: {0} ; da_mean.len: {1}"
                            .format(len(min_coh), len(da_mean)))

        # Tähele tuleb panna, et da massiiv on filtreeritud coh_thresh_ind alusel
        coh_thresh = self.__get_coh_thresh(min_coh, da_mean, is_min_coh_nan_array,
                                           data.da[coh_thresh_ind])
        self.__logger.debug("Second run coh_thresh.len: {0}".format(len(coh_thresh)))

        # todo äkki filteeriks kohe saadud tulemid?
        keep_ind = self.__get_keep_ind(topofit.coh_ps, coh_thresh, coh_thresh_ind,
                                                   topofit.k_ps)
        self.__logger.debug("keep_ind.len: {0}"
                            .format(len(keep_ind)))

        # Leitud tulemused klassimuutujatesse
        self.coh_thresh = coh_thresh
        self.ph_patch = ph_patch
        self.coh_thresh_ind = coh_thresh_ind
        self.keep_ind = keep_ind
        self.coh_ps = coh_ps # StaMPS'is salvestati see muutuja eelmisest protsessist üle
        self.coh_ps2 = topofit.coh_ps #todo parem nimi
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
        """Leiab parameetritest ps_files väärtused mida on hiljem vaja ning vajadusel muudab neid.
        Natuke kattub __load_ps_params meetodiga PsEstGamma funkstioonis"""

        def get_da_max(da):
            # todo miks 10000?
            if da.size >= 10000:
                da_sorted = np.sort(da, axis=0)
                if da.size >= 50000:
                    bin_size = 10000
                else:
                    bin_size = 2000

                # bin_size - 1 on selleks, et võtta täpselt need elemendid nende indeksitega
                # mis Matlab'is
                da_max = np.concatenate(
                    (np.zeros(1), da_sorted[bin_size - 1: -bin_size - 1: bin_size],
                     np.array([da_sorted[-1]])))
            else:
                da_max = np.array([0], [1])
                da = np.ones(len(self.__ps_est_gamma.coh_ps))

            return da_max, da

        def filter_params_based_on_ifgs_and_master(ph: np.ndarray, bperp: np.ndarray, nr_ifgs: int):
            """Filteerime ph ja bperp massiividest välja masteri veeru"""

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

        # StaMPS'is tehti see siis kui small_baseline_flag ei olnud 'y'. Siin protsessis on see alati sedasi
        ifg_ind, ph, bperp, nr_ifgs = filter_params_based_on_ifgs_and_master(ph, bperp, nr_ifgs)

        da_max, da = get_da_max(da)

        # StaMPS'is oli see nimetatud nr_dist
        rand_dist = self.__ps_est_gamma.rand_dist

        data_dto = self.__DataDTO(ph, nr_ifgs, xy, da, ifg_ind, da_max, rand_dist)
        return data_dto

    def __get_max_rand(self, da_max: np.ndarray, xy: np.ndarray):
        """Funkstioon leidmaks muutuja mis oli StaMPS'is nimega 'max_percent_rand'.

        StaMPS'is loeti see parameetritest sisse, aga kuna seda ka muudetakse vajadusel siis
        selle leidmine siin eraldi toodud"""

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

        # Paneme kohalikesse muutujatesse eelmistest protsessidest saadud tulemused,
        # kuna täisnimetusi on paha kirjutada
        coherence_bins = self.__ps_est_gamma.coherence_bins
        rand_dist = self.__ps_est_gamma.rand_dist

        array_size = data.da_max.size - 1

        min_coh = np.zeros(array_size)
        # StaMPS'is tehti see size(da_max, 1), mis on sama mis length(da_max)
        da_mean = np.zeros(array_size)
        for i in range(array_size):
            # Bitwize selleks, et katsetada. Võib kasutada ka np.all ja np.logical_and'i
            coh_chunk = coh_ps[(data.da > data.da_max[i]) & (data.da <= data.da_max[i + 1])]

            da_mean[i] = np.mean(
                data.da[(data.da > data.da_max[i]) & (data.da <= data.da_max[i + 1])])
            # Eelmaldame pikslid millel koherentsust ei leitud
            coh_chunk = coh_chunk[coh_chunk != 0]
            # StaMSP'is oli see muutuja 'Na'
            hist, _ = MatlabUtils.hist(coh_chunk, coherence_bins)

            hist_low_coh_sum = MatlabUtils.sum(hist[:self.__low_coh_tresh])
            rand_dist_low_coh_sum = MatlabUtils.sum(rand_dist[:self.__low_coh_tresh])
            nr = rand_dist * hist_low_coh_sum / rand_dist_low_coh_sum  # todo Mis on muutuja nimi 'nr'

            # Siin oli vahel ka mingi joonise tegemine

            hist[hist == 0] = 1

            # Percent_rand'i leidmine
            # np.flip võimaldab vastu võtta ühemõõtmelisi massive, seepärast ei kasuta np.fliplr
            nr_cumsum = np.cumsum(np.flip(nr, axis=0), axis=0)
            if self.__select_method is self._SelectMethod.PERCENT:
                hist_cumsum = np.cumsum(np.flip(hist, axis=0), axis=0) * 100
                percent_rand = np.flip(np.divide(nr_cumsum, hist_cumsum), axis=0)
            else:
                percent_rand = np.flip(nr_cumsum, axis=0)

            ok_ind = np.where(percent_rand < max_rand)[0]

            if len(ok_ind) == 0:
                # Kui koherentsuse väärtused ületavad lubatu väärtuse
                min_coh[i] = 1
            else:
                # Järgnevatele leitavatele indeksitele liidetavad maagilised konstandid on identsed
                # StaMPS'iga hoolimata asjaolust, et Matlab'is on indeksid ühe võrra väiksemad. Seda
                # seepärast et selle ühega arvestamine tehakse juba 'ok_ind' massiivi loomisel ära

                min_fit_ind = MatlabUtils.min(ok_ind) - 3  # todo miks 3?

                if min_fit_ind <= 0:
                    min_coh[i] = np.nan
                else:
                    max_fit_ind = MatlabUtils.min(ok_ind) + 2  # todo miks 2?

                    # StaMPS'is oli suuruse asemel konstant 100
                    if max_fit_ind > len(percent_rand) - 1:
                        max_fit_ind = len(percent_rand) - 1

                    x_cordinates = percent_rand[min_fit_ind:max_fit_ind + 1]

                    y_cordinates = ArrayUtils.arange_include_last((min_fit_ind + 1) * 0.01,
                                                                  (max_fit_ind + 1) * 0.01, 0.01)
                    min_coh[i] = MatlabUtils.polyfit_polyval(x_cordinates, y_cordinates, 3,
                                                             max_rand)

        # Leiame kas min_coh on täis nan'e ja on täiesti kasutamatu.
        # See osa oli natuke teisem StaMPS'is, Siin olen ma toonud kogu min_coh'i ja da_mean'iga tegevused ühte meetodi
        not_nan_ind = np.where(min_coh != np.nan)[0]
        is_min_coh_nan_array = sum(not_nan_ind) == 0
        # Kui erinevusi ei olnud siis pole ka mõtet võtta array'idest osasid
        if not is_min_coh_nan_array or (not_nan_ind == array_size):
            min_coh = min_coh[not_nan_ind]
            da_mean = da_mean[not_nan_ind]

        return min_coh, da_mean, is_min_coh_nan_array

    def __get_coh_thresh(self, min_coh: np.ndarray, da_mean: np.ndarray,
                         is_min_coh_nan_array: bool, da: np.ndarray):
        """Siin ei tagatsata coh_tresh_coffs'i kuna seda seda kasutati StaMPS'is vaid joonistamiseks"""
        DEF_COH_THRESH = 0.3

        if is_min_coh_nan_array:
            self.__logger.warn(
                'Not enough random phase pixels to set gamma threshold - using default threshold of '
                + str(DEF_COH_THRESH))
            # Tavaväärtus pannakse üksikult massiivi, et pärast kontollida selle jägi
            coh_thresh = np.array([DEF_COH_THRESH])
        else:
            # Kuna eelmises funktsioonis muutsime juba vastavalt min_coh ja da_mean parameetreid
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

        # reshape on vajalik, et where saaks hakkama, see massiiv on natuke imelik
        # [0] on vajalik where pärast, mis tagastab tuple
        coh_thresh_ind_fun = lambda: np.where(coh_ps.reshape(len(coh_ps)) > coh_thresh)[0]
        # StaMPS'is oli see nimetatud 'ix'
        coh_thresh_ind = make_coh_thresh_ind_array(coh_thresh_ind_fun)

        if self.__gamma_stdev_reject > 0:
            ph_res_cpx = np.exp(1j * ph_res[:, data.ifg_ind])

            coh_std = np.zeros(len(coh_thresh_ind))
            for i in range(len(coh_thresh_ind)):
                # todo kuidas teha bootstrp'i Numpys?
                # bootstrap = np.boots
                # coh_std[i] = MatlabUtils.std()
                pass

            coh_thresh_filter_fun = lambda: coh_thresh_ind[coh_std < self.__gamma_stdev_reject]
            coh_thresh_ind = make_coh_thresh_ind_array(coh_thresh_filter_fun)

            # todo siin oli StaMPS'is loogika reest_flag'iga. kas seda on vaja?
            # for i in range(self.__drop_ifg_index):
            # todo siin oli StaMPS'is loogika koos small baseline flag'iga, mis minu puhul on võetud N'ina

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

            # Tühjast list'ist ei oska Python midagi võtta
            if len(ind_array) == 0:
                ind_array = np.zeros(1).astype(np.int16)

            return ind_array

        def ph_path_loop():
            # StaMPS'is siin hetkel kusutati eelmisest protsessist saadud 'ph_res' ja 'ph_patch'

            NR_PS = len(coh_thresh_ind)
            ph_patch = self.__zero_ph_array(NR_PS, data.nr_ifgs)

            # Sarnane 'nr_i' ja 'nr_j' oli juba PsEstGammas
            nr_i = MatlabUtils.max(self.__ps_est_gamma.grid_ij[:, 0])
            nr_j = MatlabUtils.max(self.__ps_est_gamma.grid_ij[:, 1])

            # StaMPS'is oli sel muutujal nimes taga '2'
            ph_filt = np.zeros((self.__clap_win, self.__clap_win, data.nr_ifgs), np.complex128)

            for i in range(ph_patch.shape[0]):
                ps_ij = self.__ps_est_gamma.grid_ij[coh_thresh_ind[i], :]

                i_min, i_max = get_max_min(ps_ij[0] - 1, nr_i)
                j_min, j_max = get_max_min(ps_ij[1] - 1, nr_j)

                # Kui siin ei tee copy't siis muudatused tehakse ka ph_grid'is
                ph_bit = np.copy(self.__ps_est_gamma.ph_grid[i_min:i_max + 1, j_min:j_max + 1, :])

                ps_bit_i = int(ps_ij[0] - i_min - 1)
                ps_bit_j = int(ps_ij[1] - j_min - 1)
                ph_bit[ps_bit_i, ps_bit_j, :] = 0

                # todo mingi JJS oversample update
                ph_bit_len = len(ph_bit) + 1
                ph_bit_ind_i = get_ph_bit_ind_array(ps_bit_i, ph_bit_len)
                ph_bit_ind_j = get_ph_bit_ind_array(ps_bit_j, ph_bit_len)
                ph_bit[ph_bit_ind_i, ph_bit_ind_j, 0] = 0

                # Sarnane küll PsEstGammas oleva ph_flit'iga, aga siisiki erinev
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
        StaMPS'is eraldi funktsioon clap_filt_patch"""

        alpha = self.__clap_alpha
        beta = self.__clap_beta

        if len(low_pass) == 0:
            low_pass = np.zeros(len(ph))

        ph = np.nan_to_num(ph)

        # todo see ph_fft jne on väga sarnane PhEstGammas oleva clap_filt'iga
        ph_fft = np.fft.fft2(ph) #fixme mingil põhjusel erineb kolmanda tiiruga
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

        # StaMPS'is muudeti eelmisena saadud tulemust. Siin nii ei tee
        coh_ps = self.__ps_est_gamma.coh_ps.copy()
        coh_ps[coh_thresh_ind] = topofit.coh_ps

        return coh_ps, topofit

    def __get_keep_ind(self, coh_ps : np.ndarray, coh_thresh : np.ndarray,
                       coh_thresh_ind : np.ndarray, k_ps2: np.ndarray) -> np.ndarray:
        """Stamps'is oli samasugune asi nimetatud 'keep_ix'"""

        bperp_meaned = self.__ps_files.bperp_meaned
        k_ps = self.__ps_est_gamma.k_ps[coh_thresh_ind]
        bperp_delta = np.max(bperp_meaned) - np.min(bperp_meaned)

        # Reshape on vajalik selleks, et ei tekiks massiivi massiivis
        coh_ps_len = len(coh_ps)
        coh_ps_reshaped = coh_ps.reshape(coh_ps_len)
        delta = (np.abs(k_ps - k_ps2) < 2 * np.pi / bperp_delta).reshape(coh_ps_len) #todo parem nimi
        keep_ind = np.where((coh_ps_reshaped > coh_thresh) & delta)[0]

        return keep_ind

    # todo mingi parem lahendus siia ehk?
    # Konstruktor tühja pusivpeegeladajate info massiivi loomiseks
    # TODO: Samasugune asi oli juba PsEstGammas, Refacto?
    def __zero_ps_array(self, shape):
        return np.zeros(shape)

    # TODO: Miks see on erinev PsEstGammast?
    def __zero_ph_array(self, nr_ps, nr_ifgs):
        return np.zeros((nr_ps, nr_ifgs), np.complex128)
