from datetime import datetime
import numpy as np
import numpy.matlib
import sys
import math
from pathlib import Path

from scripts.MetaSubProcess import MetaSubProcess
from scripts.processes.PsEstGamma import PsEstGamma
from scripts.processes.PsFiles import PsFiles
from scripts.processes.PsSelect import PsSelect
from scripts.utils.ArrayUtils import ArrayUtils
from scripts.utils.FolderConstants import FolderConstants
from scripts.utils.LoggerFactory import LoggerFactory
from scripts.utils.MatlabUtils import MatlabUtils


class PsWeed(MetaSubProcess):
    """Pikslite filtreerimine teiste naabrusest. Valitakse hulgast vaid selgemad"""

    __IND_ARRAY_TYPE = np.int32
    __DEF_NEIGHBOUR_VAL = -1

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

        #todo object? tuple?
        #todo milleks üldse ps_weed_edge_nr? see on ju len(ps_weed_edge_data)
        self.__ps_weed_edge_nr, self.__ps_weed_edge_data = self.__load_psweed_edge_file(path_to_patch)

    def __load_psweed_edge_file(self, path: str) -> (int, np.ndarray):
        """Põhjus miks me ei loe seda faili sisse juba PsFiles'ides on see, et me ei pruugi
        PsWeed protsessi jõuda enda töötluses ja seda läheb ainult siin vaja"""
        # todo selle võib teha @lazy'ga PsFiles'idesse

        file_name = "psweed.2.edge"
        path = Path(path, FolderConstants.PATCH_FOLDER_NAME, file_name)
        self.__logger.debug("Path to psweed_edgke file: " + str(path))
        if path.exists():
            header = np.genfromtxt(path, max_rows=1, dtype=self.__IND_ARRAY_TYPE)
            data = np.genfromtxt(path, skip_header=True, dtype=self.__IND_ARRAY_TYPE)
            return header[0], data
        else:
            raise FileNotFoundError("{1} not found. AbsPath {0}".format(str(path.absolute()), file_name))

    class __DataDTO(object):

        def __init__(self, ind: np.ndarray, ph_res: np.ndarray, coh_thresh_ind: np.ndarray,
                     k_ps: np.ndarray, c_ps: np.ndarray, coh_ps: np.ndarray, pscands_ij: np.matrix,
                     xy: np.ndarray, lonlat: np.matrix, hgt: np.ndarray, ph: np.ndarray,
                     ph2: np.ndarray, ph_patch_org: np.ndarray, bperp_meaned: np.ndarray, nr_ifgs: int,
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
            self.ph2 = ph2
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

        selectable_ps = self.__select_best(neighbour_ps, coh_thresh_ind_len, data.coh_ps, data.hgt)
        self.__logger.debug("selectable_ps.len: {0}, true vals: {1}"
                            .format(len(selectable_ps), np.count_nonzero(selectable_ps)))
        del neighbour_ps

        xy, selectable_ps = self.__filter_xy(data.xy, selectable_ps, data.coh_ps)

        # PsWeed'is tehakse oma inteferogrammide massiiv. Stamps'is oli muutuja nimi 'ifg_index'
        ifg_ind = np.arange(0, data.nr_ifgs, dtype=self.__IND_ARRAY_TYPE)
        if len(self.__drop_ifg_index) > 0:
            self.__logger.debug("Droping indexes {0}".format(self.__drop_ifg_index))

        # Stamps'is oli selle asemel 'no_weed_noisy'
        if not (self.__weed_standard_dev >= math.pi and self.__weed_max_noise >= math.pi):
            self.__drop_noisy(data, selectable_ps, ifg_ind)

        self.__logger.info("End")

    def __load_ps_params(self):

        def get_from_ps_select():
            ind = self.ps_select.keep_ind
            ph_res = self.ps_select.ph_res[ind]

            if len(ind) > 0:
                coh_thresh_ind = self.ps_select.coh_thresh_ind[ind]
                c_ps = self.ps_select.c_ps[ind]
                k_ps = self.ps_select.k_ps[ind]
                coh_ps = self.ps_select.coh_ps2[ind]
            else:
                coh_thresh_ind = self.ps_select.coh_thresh_ind
                c_ps = self.ps_select.c_ps
                k_ps = self.ps_select.k_ps
                coh_ps = self.ps_select.coh_ps2

            return ind, ph_res, coh_thresh_ind, k_ps, c_ps, coh_ps

        def get_from_ps_files():
            pscands_ij = self.ps_files.pscands_ij[coh_thresh_ind]
            xy = self.ps_files.xy[coh_thresh_ind]
            ph = self.ps_files.ph[coh_thresh_ind]
            lonlat = self.ps_files.lonlat[coh_thresh_ind]
            hgt = self.ps_files.hgt[coh_thresh_ind]

            master_nr = self.ps_files.master_nr
            ifg_dates = self.ps_files.ifg_dates

            return pscands_ij, xy, ph, lonlat, hgt, master_nr, ifg_dates

        def get_from_ps_est_gamma():

            ph_patch_org = self.ps_est_gamma.ph_patch[coh_thresh_ind, :]
            ph, _, nr_ifgs, nr_ps, _, _ = self.ps_files.get_ps_variables()
            bperp_meaned = self.ps_files.bperp_meaned
            master_date = self.ps_files.master_date

            return ph_patch_org, ph, bperp_meaned, nr_ifgs, nr_ps, master_date

        # fixme ph_path'e on Stampsis ainult üks.

        ind, ph_res, coh_thresh_ind, k_ps, c_ps, coh_ps = get_from_ps_select()

        pscands_ij, xy, ph2, lonlat, hgt, master_nr, ifg_dates = get_from_ps_files()

        ph_patch_org, ph, bperp_meaned, nr_ifgs, nr_ps, master_date = get_from_ps_est_gamma()

        # Stamps'is oli siin oli ka lisaks 'all_da_flag' ja leiti teised väärtused muutujatele k_ps,
        # c_ps, coh_ps, ph_patch_org, ph_res

        return self.__DataDTO(ind, ph_res, coh_thresh_ind, k_ps, c_ps, coh_ps, pscands_ij, xy,
                              lonlat, hgt, ph, ph2, ph_patch_org, bperp_meaned, nr_ifgs, nr_ps,
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
        selectable_ps = np.ones(coh_thresh_ind_len, dtype=bool)  # Stamps'is oli see 'ix_weed'

        for i in range(coh_thresh_ind_len):
            ps_ind = neighbour_ps[i]
            if len(ps_ind) != 0:
                j = 0
                while j < len(ps_ind):
                    ps_i = ps_ind[j]
                    ps_ind = np.append(ps_ind, neighbour_ps[ps_i]).astype(self.__IND_ARRAY_TYPE)
                    neighbour_ps[ps_i] = np.array([]) # todo jätaks selle äkki ära? pole mõtet muuta kui pärast neid andmeid ei kasuta
                    j += 1

                ps_ind = np.unique(ps_ind)
                highest_coh_ind = coh_ps[ps_ind].argmax()

                low_coh_ind = np.ones(len(ps_ind), dtype=bool)
                low_coh_ind[highest_coh_ind] = False

                ps_ind = ps_ind[low_coh_ind]
                selectable_ps[ps_ind] = False

        self.__logger.debug("self.__weed_zero_elevation: {0}, len(htg)")
        if self.__weed_zero_elevation and len(htg) > 0:
            self.__logger.debug("Fiding sea evel")
            sea_ind = htg < 1e-6
            selectable_ps[sea_ind] = False

        return selectable_ps

    def __filter_xy(self, xy: np.ndarray, selectable_ps: np.ndarray, coh_ps: np.ndarray):
        """Leiame xy massiiv filteeritult
        Siin oli veel lisaks kas tehtud dublikaatide massiv on tühi, aga selle peale leti weeded_xy
        uuesti, aga mina sellisel tegevusel mõtet ei näinud"""

        #todo funksioon väiksemaks? eraldi xy ja eraldi weed_ind?
        weeded_xy = xy[selectable_ps] # Stamps'is oli see 'xy_weed'

        weed_ind = np.nonzero(selectable_ps)[0] # Stamsp*is oli see 'ix_weed_num' #todo iteratalbe get array?!??
        unique_rows = np.unique(weeded_xy, return_index=True, axis=0)[1].astype(self.__IND_ARRAY_TYPE)
        # Stamps'is transponeeriti ka veel seda järgmist, aga siin ei tee see midagi
        last = np.arange(0, len(weed_ind))
        # Stamps'is oli see 'dps'. Pikslid topelt lon/ lat'iga
        duplicates = np.setxor1d(unique_rows, last)

        for i in range(len(duplicates)): #todo for-each
            duplicate = duplicates[i]
            weeded_duplicates_ind = np.where((weeded_xy[:, 0] == weeded_xy[duplicate, 0]) &
                                      ((weeded_xy[:, 1]) == weeded_xy[duplicate, 1])) # 'dups_ix_weed' oli originaalis
            duplicates_ind = weed_ind[weeded_duplicates_ind] #
            high_coh_ind = coh_ps[duplicates_ind].argmax()
            selectable_ps[duplicates_ind != high_coh_ind] = False

        return xy, selectable_ps

    def __drop_noisy(self, data: __DataDTO, selectable_ps: np.ndarray, ifg_ind: np.ndarray):

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

        ph_filtered = data.ph2[selectable_ps]
        k_ps_filtered = data.k_ps[selectable_ps]
        c_ps_filtered = data.c_ps[selectable_ps]
        bperp_meaned = data.bperp_meaned
        master_nr = data.master_nr
        ifg_dates = data.ifg_dates

        ph_weed = get_ph_weed(bperp_meaned, k_ps_filtered, ph_filtered, c_ps_filtered, master_nr)

        edges = self.__ps_weed_edge_data
        #todo lõpus pole neid : vaja vist
        dph_space = np.multiply(ph_weed[edges[:, 2] - 1, :], ph_weed[edges[:, 1] - 1, :].conj())
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

            repmat = np.matlib.repmat(weight_factor, len(edges), 1)
            dph_mean = np.sum(np.multiply(dph_space, repmat), axis=1)

            # Stamps'is tehti dph_mean'ile conj() ehk konjugeerimine, aga ühemõõtmeliste
            # Numpy array'ide puhul see ei tee midagi ja siis teeme ta siin veerumaatriksiks
            # kasutades abifunksiooni
            repmat = np.matlib.repmat(ArrayUtils.to_col_matrix(dph_mean).conj(), 1, len(ifg_ind))
            dph_mean_adj = np.angle(np.multiply(dph_space, repmat))

            G = np.array([np.ones(len(ifg_ind)), time_delta]).transpose()
            # Stamps'is oli 'm'
            weighted_least_sqrt = MatlabUtils.lscov(G, dph_mean_adj.transpose(), weight_factor)
            #todo parem muutja nimi
            least_sqrt_G = np.asarray((np.asmatrix(G) * np.asmatrix(weighted_least_sqrt)).transpose())
            dph_mean_adj = np.angle(np.exp(1j * (dph_mean_adj - least_sqrt_G)))
            # Stamps'is oli 'm2'
            weighted_least_sqrt2 = MatlabUtils.lscov(G, dph_mean_adj.transpose(), weight_factor)

            # weighted_least_sqrt'te juures jätame transponeerimise tegemata sest see ei mõjuta midagi
            dph_smooth_val_exp = np.exp(1j * (weighted_least_sqrt[0, :] + weighted_least_sqrt2[0, :]))
            dph_smooth[:, i] = np.multiply(dph_mean, dph_smooth_val_exp)
            weight_factor[i] = 0 # Jätame ennast välja

            #todo koodikordus ülemisega dph_mean_adj
            repmat = np.matlib.repmat(weight_factor, len(edges), 1)
            dph_smooth2[:, i] = np.sum(np.multiply(dph_space, repmat), axis=1)

        # todo tegelikult pole vaja nii palju neid dph_noise'e, saaks ka ühega hakkama, vaata kasutamise järjekorda
        dph_noise = np.angle(np.multiply(dph_space, dph_smooth.conj()))
        dph_noise2 = np.angle(np.multiply(dph_space, dph_smooth2.conj()))
        ifg_var = np.var(dph_noise2, 0)

        K_weights = np.divide(1, ifg_var)
        K = MatlabUtils.lscov(bperp_meaned, dph_noise.transpose(), K_weights).conj().transpose()
        dph_noise -= K * bperp_meaned.transpose()

        edge_std = MatlabUtils.std(dph_noise, axis=1)
        edge_max = np.max(np.abs(dph_noise), axis=1)