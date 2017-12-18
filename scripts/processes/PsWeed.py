import os

from datetime import datetime
import numpy as np
import numpy.matlib
import sys
import math
from pathlib import Path

from scripts import RESOURCES_PATH
from scripts.MetaSubProcess import MetaSubProcess
from scripts.processes.PsEstGamma import PsEstGamma
from scripts.processes.PsFiles import PsFiles
from scripts.processes.PsSelect import PsSelect
from scripts.utils.ArrayUtils import ArrayUtils
from scripts.utils.internal.ConfigUtils import ConfigUtils
from scripts.utils.internal.FolderConstants import FolderConstants
from scripts.utils.internal.LoggerFactory import LoggerFactory
from scripts.utils.MatlabUtils import MatlabUtils
from scripts.utils.internal.ProcessDataSaver import ProcessDataSaver


class PsWeed(MetaSubProcess):
    """Pikslite filtreerimine teiste naabrusest. Valitakse hulgast vaid selgemad"""

    __IND_ARRAY_TYPE = np.int32
    __DEF_NEIGHBOUR_VAL = -1
    __FILE_NAME = "ps_weed"

    selectable_ps = np.array([])

    def __init__(self, path_to_patch: str, ps_files: PsFiles, ps_est_gamma: PsEstGamma,
                 ps_select: PsSelect):
        self.ps_files = ps_files
        self.ps_select = ps_select
        self.ps_est_gamma = ps_est_gamma

        self.__logger = LoggerFactory.create("PsWeed")

        self.__time_win = 730
        self.__weed_standard_dev = 1
        self.__weed_max_noise = sys.maxsize  # Stampsis oli tavaväärtus inf
        self.__weed_zero_elevation = False
        self.__weed_neighbours = True
        # todo drop_ifg_index on juba PsSelect'is
        self.__drop_ifg_index = np.array([])

        self.__ps_weed_edge_data = self.__load_psweed_edge_file(path_to_patch)
        self.__logger.debug("self.__ps_weed_edge_data.len: " + str(len(self.__ps_weed_edge_data)))

    def __load_psweed_edge_file(self, path: str) -> (int, np.ndarray):
        """Põhjus miks me ei loe seda faili sisse juba PsFiles'ides on see, et me ei pruugi
        PsWeed protsessi jõuda enda töötluses ja seda läheb ainult siin vaja.

        Stamps'is võeti päisest ka number, kui suur massiiv on, aga ma ei näe mõtet sellel"""
        # todo selle võib teha @lazy'ga PsFiles'idesse

        file_name = "psweed.2.edge"
        path = Path(path, FolderConstants.PATCH_FOLDER_NAME, file_name)
        self.__logger.debug("Path to psweed edge file: " + str(path))
        if path.exists():
            data = np.genfromtxt(path, skip_header=True, dtype=self.__IND_ARRAY_TYPE)
            return data
        else:
            raise FileNotFoundError("File named '{1}' not found. AbsPath '{0}'".format(
                str(path.absolute()), file_name))

    class __DataDTO(object):

        def __init__(self, ind: np.ndarray, ph_res: np.ndarray, coh_thresh_ind: np.ndarray,
                     k_ps: np.ndarray, c_ps: np.ndarray, coh_ps: np.ndarray, pscands_ij: np.matrix,
                     xy: np.ndarray, lonlat: np.matrix, hgt: np.ndarray, ph: np.ndarray,
                     ph_patch_org: np.ndarray, bperp_meaned: np.ndarray, nr_ifgs: int,
                     nr_ps: int, master_date: datetime, master_nr: int, ifg_dates: []):
            self.ind = ind
            self.ph_res = ph_res
            self.coh_thresh_ind = coh_thresh_ind
            self.k_ps = k_ps
            self.c_ps = c_ps
            self.coh_ps = coh_ps
            self.pscands_ij = pscands_ij
            self.xy = xy
            self.lonlat = lonlat
            self.hgt = hgt
            self.ph_patch_org = ph_patch_org
            self.ph = ph
            self.bperp_meaned = bperp_meaned
            self.nr_ifgs = nr_ifgs
            self.nr_ps = nr_ps
            self.master_date = master_date
            self.master_nr = master_nr
            self.ifg_dates = ifg_dates

    def start_process(self):
        self.__logger.info("Start")

        data = self.__load_ps_params()
        # Stamps*is oli see nimetatud kui nr_ps, aga see on meil juba olemas
        coh_thresh_ind_len = len(data.coh_thresh_ind)
        self.__logger.debug("Loaded data. coh_thresh_ind.len: {0}, coh_thresh_ind_len: {1}"
                            .format(coh_thresh_ind_len, data.nr_ps))

        ij_shift = self.__get_ij_shift(data.pscands_ij, coh_thresh_ind_len)
        self.__logger.debug("ij_shift.len: {0}".format(len(ij_shift)))

        neighbour_ind = self.__init_neighbours(ij_shift, coh_thresh_ind_len)
        self.__logger.debug("neighbours.len: {0}".format(len(neighbour_ind)))

        neighbour_ps = self.__find_neighbours(ij_shift, coh_thresh_ind_len, neighbour_ind)
        # todo kas saab logida ka tühjade arvu?
        self.__logger.debug("neighbour_ps.len: {0}".format(len(neighbour_ps)))

        # Stamps'is oli see 'ix_weed'
        selectable_ps = self.__select_best(neighbour_ps, coh_thresh_ind_len, data.coh_ps, data.hgt)
        self.__logger.debug("selectable_ps.len: {0}, true vals: {1}"
                            .format(len(selectable_ps), np.count_nonzero(selectable_ps)))
        del neighbour_ps

        xy, selectable_ps = self.__filter_xy(data.xy, selectable_ps, data.coh_ps)

        # PsWeed'is tehakse oma inteferogrammide massiiv. Stamps'is oli muutuja nimi 'ifg_index'
        ifg_ind = np.arange(0, data.nr_ifgs, dtype=self.__IND_ARRAY_TYPE)
        if len(self.__drop_ifg_index) > 0:
            self.__logger.debug("Dropping indexes {0}".format(self.__drop_ifg_index))

        # Stamps'is oli selle asemel 'no_weed_noisy'
        if not (self.__weed_standard_dev >= math.pi and self.__weed_max_noise >= math.pi):
            edge_std, edge_max = self.__drop_noisy(data, selectable_ps, ifg_ind,
                                                   self.__ps_weed_edge_data)
            self.__logger.debug("edge_std.len: {0}, edge_std.len: {1}"
                                .format(len(edge_std), len(edge_std)))
            ps_std, ps_max = self.__get_ps_arrays(edge_std, edge_max,
                                                  np.count_nonzero(selectable_ps),
                                                  self.__ps_weed_edge_data)
            self.__logger.debug("ps_std.len: {0}, ps_max.len: {1}"
                                .format(len(ps_std), len(ps_max)))
            selectable_ps, selectable_ps2 = self.__estimate_max_noise(ps_std, ps_max, selectable_ps)
            self.__logger.debug("selectable_ps.len: {0}, selectable_ps2.len: {1}"
                                .format(len(selectable_ps), len(selectable_ps2)))
        else:
            self.__logger.error("weed_standard_dev or weed_max_noise where bigger than pi")
            raise NotImplemented("weed_standard_dev or weed_max_noise where bigger than pi")

        # Leitud tulemused klassimuutujatesse
        self.selectable_ps = selectable_ps
        self.selectable_ps2 = selectable_ps2 # todo parem nimi
        self.ifg_ind = ifg_ind
        self.ps_max = ps_max
        self.ps_std = ps_std

        self.__logger.info("End")

    def save_results(self, save_path: str):
        ProcessDataSaver(save_path, self.__FILE_NAME).save_data(
            selectable_ps = self.selectable_ps,
            selectable_ps2 = self.selectable_ps2,
            ifg_ind = self.ifg_ind,
            ps_max = self.ps_max,
            ps_std = self.ps_std
        )

    def load_results(self, load_path: str):
        file_with_path = os.path.join(load_path, self.__FILE_NAME + ".npz")
        data = np.load(file_with_path)

        self.selectable_ps = data["selectable_ps"]
        self.selectable_ps2 = data["selectable_ps2"]
        self.ifg_ind = data["ifg_ind"]
        self.ps_max = data["ps_max"]
        self.ps_std = data["ps_std"]

    def get_filtered_results(self, load_path: str = None):
        """Kuna filteerimine selectable_ps'iga alusel tehakse muutjate põhjal mis on siin juba
        leitud sisendmuutujate leidmisel ja filteeritud näiteks 'ps_select.keep_ind' siis on siin
        klassis seda oluliselt lihtsam teha.

        Stamps'is tehti uued .mat failid nende salvestamiseks"""

        self.__logger.info("Finding filtered results")

        if len(self.selectable_ps) == 0:
            self.__logger.debug("Load results")
            if load_path is None:
                load_path = ConfigUtils(RESOURCES_PATH).get_default_section("save_load_path")
                self.__logger.info("Using default load/save path from config: '{0}'".format(
                    load_path))

            self.load_results(load_path)

        data = self.__load_ps_params()

        coh_ps = data.coh_ps[self.selectable_ps]
        k_ps = data.k_ps[self.selectable_ps]
        c_ps = data.c_ps[self.selectable_ps]
        ph_patch = data.ph_patch_org[self.selectable_ps]
        ph = data.ph[self.selectable_ps]
        xy = data.xy[self.selectable_ps]
        pscands_ij = data.pscands_ij[self.selectable_ps]
        lonlat = data.lonlat[self.selectable_ps]
        hgt = data.hgt[self.selectable_ps]

        bperp = self.ps_files.bperp[data.coh_thresh_ind]
        bperp = bperp[self.selectable_ps]

        return coh_ps, k_ps, c_ps, ph_patch, ph, xy, pscands_ij, lonlat, hgt, bperp

    def __load_ps_params(self):

        def get_from_ps_select(ps_select: PsSelect):
            ind = ps_select.keep_ind
            ph_res = ps_select.ph_res[ind]

            if len(ind) > 0:
                coh_thresh_ind = ps_select.coh_thresh_ind[ind]
                c_ps = ps_select.c_ps[ind]
                k_ps = ps_select.k_ps[ind]
                coh_ps = ps_select.coh_ps2[ind]
            else:
                coh_thresh_ind = ps_select.coh_thresh_ind
                c_ps = ps_select.c_ps
                k_ps = ps_select.k_ps
                coh_ps = ps_select.coh_ps2

            return ind, ph_res, coh_thresh_ind, k_ps, c_ps, coh_ps

        def get_from_ps_files(ps_files: PsFiles, coh_thresh_ind: np.ndarray):
            # Kasutame väärtuste saamiseks tavapärast funksiooni
            ph, _, nr_ifgs, nr_ps, xy, _ = ps_files.get_ps_variables()

            # Ja siis filterdame coh_thresh alusel
            pscands_ij = ps_files.pscands_ij[coh_thresh_ind]
            xy = xy[coh_thresh_ind]
            ph = ph[coh_thresh_ind]
            lonlat = ps_files.lonlat[coh_thresh_ind]
            hgt = ps_files.hgt[coh_thresh_ind]

            # Ja siis on mõned asjad mida me ei filterda
            master_nr = ps_files.master_nr
            ifg_dates = ps_files.ifg_dates
            bperp_meaned = ps_files.bperp_meaned
            master_date = ps_files.master_date

            return pscands_ij, xy, ph, lonlat, hgt, nr_ifgs, nr_ps, master_nr, ifg_dates, \
                   bperp_meaned, master_date

        def get_from_ps_est_gamma(ps_est_gamma: PsEstGamma):
            ph_patch = ps_est_gamma.ph_patch[coh_thresh_ind, :]

            return ph_patch

        ind, ph_res, coh_thresh_ind, k_ps, c_ps, coh_ps = get_from_ps_select(self.ps_select)

        pscands_ij, xy, ph, lonlat, hgt, nr_ifgs, nr_ps, master_nr, ifg_dates, bperp_meaned,\
        master_date = get_from_ps_files(self.ps_files, coh_thresh_ind)

        ph_patch_org = get_from_ps_est_gamma(self.ps_est_gamma)

        # Stamps'is oli siin oli ka lisaks 'all_da_flag' ja leiti teised väärtused muutujatele k_ps,
        # c_ps, coh_ps, ph_patch_org, ph_res

        return self.__DataDTO(ind, ph_res, coh_thresh_ind, k_ps, c_ps, coh_ps, pscands_ij, xy,
                              lonlat, hgt, ph, ph_patch_org, bperp_meaned, nr_ifgs, nr_ps,
                              master_date, master_nr, ifg_dates)

    def __get_ij_shift(self, pscands_ij: np.matrix, coh_ps_len: int) -> np.ndarray:
        ij = np.asarray(pscands_ij[:, 1:3])
        repmated = np.matlib.repmat(np.array([2, 2]) - ij.min(axis=0), coh_ps_len, 1)
        ij_shift = ij + repmated

        return ij_shift

    def __init_neighbours(self, ij_shift: np.ndarray, coh_ps_len: int) -> np.ndarray:
        """Stamps'is täideti massiiv nullidega siis mina täidan siin -1 'ega.
        Kuna täidetakse massiiv indeksitest ja Numpy's/ Python'is hakkavad indeksid nullist siis
        täidame -1'ega ja siis uute väärtustega"""

        def arange_neighbours_select_arr(i, ind):
            #todo repmat?
            return ArrayUtils.arange_include_last(ij_shift[i, ind] - 2, ij_shift[i, ind])

        def make_miss_middle_mask():
            miss_middle = np.ones((3, 3), dtype=bool)
            miss_middle[1, 1] = False

            return miss_middle

        neighbour_ind = np.ones((MatlabUtils.max(ij_shift[:, 0]) + 1,
                                 MatlabUtils.max(ij_shift[:, 1]) + 1),
                                self.__IND_ARRAY_TYPE) * self.__DEF_NEIGHBOUR_VAL
        miss_middle = make_miss_middle_mask()

        for i in range(coh_ps_len):
            start = arange_neighbours_select_arr(i, 0)
            end = arange_neighbours_select_arr(i, 1)

            # Selleks, et saada len(start) * len(end) massiivi tuleb numpy's sedasi selekteerida
            # Võib kasutada ka neighbour_ind[start, :][:, end], aga see ei luba pärast sama moodi
            # väärtustada
            neighbours_val = neighbour_ind[np.ix_(start, end)]
            neighbours_val[(neighbours_val == self.__DEF_NEIGHBOUR_VAL) & (miss_middle == True)] = i

            neighbour_ind[np.ix_(start, end)] = neighbours_val

        return neighbour_ind

    def __find_neighbours(self, ij_shift: np.ndarray, coh_thresh_ind_len: int,
                          neighbour_ind: np.ndarray) -> np.ndarray:
        # Loome tühja listi, kus on sees tühjad numpy massivid
        neighbour_ps = [np.array([], self.__IND_ARRAY_TYPE)] * (coh_thresh_ind_len + 1)
        for i in range(coh_thresh_ind_len):
            ind = neighbour_ind[ij_shift[i, 0] - 1, ij_shift[i, 1] - 1]
            if ind != self.__DEF_NEIGHBOUR_VAL:
                neighbour_ps[ind] = np.append(neighbour_ps[ind], [i])

        return np.array(neighbour_ps)

    def __select_best(self, neighbour_ps: np.ndarray, coh_thresh_ind_len: int,
                      coh_ps: np.ndarray, htg: np.ndarray) -> np.ndarray:
        """Tagastab boolean'idest array, et pärast selle järgi filteerida ülejäänud massiivid.
        Stamps'is oli tegemist massiiv int'intidest"""
        selectable_ps = np.ones(coh_thresh_ind_len, dtype=bool)

        for i in range(coh_thresh_ind_len):
            ps_ind = neighbour_ps[i]
            if len(ps_ind) != 0:
                j = 0
                while j < len(ps_ind):
                    ps_i = ps_ind[j]
                    ps_ind = np.append(ps_ind, neighbour_ps[ps_i]).astype(self.__IND_ARRAY_TYPE)
                    neighbour_ps[ps_i] = np.array([]) # Mis on loetud tühjendame
                    j += 1

                ps_ind = np.unique(ps_ind)
                highest_coh_ind = coh_ps[ps_ind].argmax()

                low_coh_ind = np.ones(len(ps_ind), dtype=bool)
                low_coh_ind[highest_coh_ind] = False

                ps_ind = ps_ind[low_coh_ind]
                selectable_ps[ps_ind] = False

        self.__logger.debug("self.__weed_zero_elevation: {0}, len(htg): {1}".format(
            self.__weed_zero_elevation, len(htg)))
        if self.__weed_zero_elevation and len(htg) > 0:
            self.__logger.debug("Fiding sea evel")
            sea_ind = htg < 1e-6
            selectable_ps[sea_ind] = False

        return selectable_ps

    def __filter_xy(self, xy: np.ndarray, selectable_ps: np.ndarray, coh_ps: np.ndarray):
        """Leiame xy massiiv filteeritult
        Siin oli veel lisaks kas tehtud dublikaatide massiv on tühi, aga selle peale leti weeded_xy
        uuesti, aga mina sellisel tegevusel mõtet ei näinud"""

        weeded_xy = xy[selectable_ps] # Stamps'is oli see 'xy_weed'

        weed_ind = np.flatnonzero(selectable_ps) # Stamsp*is oli see 'ix_weed_num'
        unique_rows = np.unique(weeded_xy, return_index=True, axis=0)[1].astype(self.__IND_ARRAY_TYPE)
        # Stamps'is transponeeriti ka veel seda järgmist, aga siin ei tee see midagi
        last = np.arange(0, len(weed_ind))
        # Stamps'is oli see 'dps'. Pikslid topelt lon/ lat'iga
        duplicates = np.setxor1d(unique_rows, last)

        for duplicate in duplicates:
            weeded_duplicates_ind = np.where((weeded_xy[:, 0] == weeded_xy[duplicate, 0]) &
                                      ((weeded_xy[:, 1]) == weeded_xy[duplicate, 1])) # 'dups_ix_weed' oli originaalis
            duplicates_ind = weed_ind[weeded_duplicates_ind] #
            high_coh_ind = coh_ps[duplicates_ind].argmax()
            selectable_ps[duplicates_ind != high_coh_ind] = False

        return xy, selectable_ps

    def __drop_noisy(self, data: __DataDTO, selectable_ps: np.ndarray, ifg_ind: np.ndarray,
                     edges: np.ndarray) -> (np.ndarray, np.ndarray):

        def get_ph_weed(bperp: np.ndarray, k_ps: np.ndarray, ph: np.ndarray, c_ps: np.ndarray,
                        master_nr: int):
            exped = np.exp(-1j * (k_ps * bperp.conj().transpose()))
            ph_weed = np.multiply(ph, exped)
            ph_weed = np.divide(ph_weed, np.abs(ph_weed))
            # Masteri müra lisamine. Tehti juhul kui Stamps'is oli small_baseline_flag != 'y'
            # rehsape on vajalik seepärast, et c_ps on massiiv kus sees on massiivid
            ph_weed[:, (master_nr - 1)] = np.exp(1j * c_ps).reshape(len(ph_weed))

            return ph_weed

        def get_time_deltas_in_days(index: int) -> np.ndarray:
            """Selleks, et saaks date objektst päevade vahemiku int'ides teeme järgnevalt"""
            return np.array([(ifg_dates[index] - ifg_dates[x]).days for x in np.nditer(ifg_ind)])

        def get_dph_mean(dph_space, edges_len, weight_factor):
            repmat = np.matlib.repmat(weight_factor, edges_len, 1)
            dph_mean = np.sum(np.multiply(dph_space, repmat), axis=1)

            return dph_mean

        ph_filtered = data.ph[selectable_ps]
        k_ps_filtered = data.k_ps[selectable_ps]
        c_ps_filtered = data.c_ps[selectable_ps]
        bperp_meaned = data.bperp_meaned
        master_nr = data.master_nr
        ifg_dates = data.ifg_dates

        ph_weed = get_ph_weed(bperp_meaned, k_ps_filtered, ph_filtered, c_ps_filtered, master_nr)

        dph_space = np.multiply(ph_weed[edges[:, 2] - 1], ph_weed[edges[:, 1] - 1].conj())
        dph_space = dph_space[:, ifg_ind]

        #todo drop_ifg_index loogika

        # Järgnev tehti ainult siis kui small_baseline_flag != 'y'

        dph_shape = (len(edges), len(ifg_ind))
        dph_smooth = np.zeros(dph_shape).astype(np.complex128)
        dph_smooth2 = np.zeros(dph_shape).astype(np.complex128)
        for i in range(len(ifg_ind)):
            time_delta = get_time_deltas_in_days(i)
            weight_factor = np.exp(-(np.power(time_delta, 2)) / 2 / math.pow(self.__time_win, 2))
            weight_factor = weight_factor / np.sum(weight_factor)

            dph_mean = get_dph_mean(dph_space, len(edges), weight_factor)

            repmat = np.matlib.repmat(ArrayUtils.to_col_matrix(dph_mean).conj(), 1, len(ifg_ind))
            dph_mean_adj = np.angle(np.multiply(dph_space, repmat))

            G = np.array([np.ones(len(ifg_ind)), time_delta]).transpose()
            # Stamps'is oli 'm'
            weighted_least_sqrt = MatlabUtils.lscov(G, dph_mean_adj.conj().transpose(),
                                                    weight_factor)
            #todo parem muutja nimi
            least_sqrt_G = np.asarray((np.asmatrix(G) * np.asmatrix(weighted_least_sqrt))
                                      .conj().transpose())
            dph_mean_adj = np.angle(np.exp(1j * (dph_mean_adj - least_sqrt_G)))
            # Stamps'is oli 'm2'
            weighted_least_sqrt2 = MatlabUtils.lscov(G, dph_mean_adj.conj().transpose(),
                                                     weight_factor)

            # weighted_least_sqrt'te juures jätame transponeerimise tegemata sest see ei mõjuta midagi
            dph_smooth_val_exp = np.exp(1j * (weighted_least_sqrt[0, :] + weighted_least_sqrt2[0, :]))
            dph_smooth[:, i] = np.multiply(dph_mean, dph_smooth_val_exp)
            weight_factor[i] = 0 # Jätame ennast välja

            dph_smooth2[:, i] = get_dph_mean(dph_space, len(edges), weight_factor)

        dph_noise = np.angle(np.multiply(dph_space, dph_smooth2.conj()))
        ifg_var = np.var(dph_noise, 0)

        dph_noise = np.angle(np.multiply(dph_space, dph_smooth.conj()))
        K_weights = np.divide(1, ifg_var)
        K = MatlabUtils.lscov(bperp_meaned, dph_noise.conj().transpose(), K_weights).conj().transpose()
        dph_noise -= K * bperp_meaned.transpose()

        edge_std = MatlabUtils.std(dph_noise, axis=1)
        edge_max = np.max(np.abs(dph_noise), axis=1)

        return edge_std, edge_max

    def __get_ps_arrays(self, edge_std: np.ndarray, edge_max: np.ndarray,
                        selectable_ps_true_count: int, edges: np.ndarray) -> (np.ndarray, np.ndarray):
        def get_min(ps_array: np.ndarray, edge_array: np.ndarray, edge_ind: np.ndarray, index: int):
            array = np.array(
                [ps_array[edge_ind], [edge_array[index], edge_array[index]]]).transpose()
            return np.min(array, axis=1)

        ps_std = np.full(selectable_ps_true_count, np.inf)
        ps_max = np.full(selectable_ps_true_count, np.inf)
        for i in range(len(edges)):
            edge = edges[i, 1:3] - 1
            ps_std[edge] = get_min(ps_std, edge_std, edge, i)
            ps_max[edge] = get_min(ps_max, edge_max, edge, i)

        return ps_std, ps_max

    def __estimate_max_noise(self, ps_std: np.ndarray, ps_max: np.ndarray,
                             selectable_ps: np.ndarray) -> (np.ndarray, np.ndarray):

        weeded = np.logical_and(ps_std < self.__weed_standard_dev, ps_max < self.__weed_max_noise)
        selectable_ps[selectable_ps] = weeded

        return selectable_ps, weeded


