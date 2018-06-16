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
    """Pixels filtering/ weeding from around others. Select only best/ clearest"""

    __IND_ARRAY_TYPE = np.int32
    __DEF_NEIGHBOUR_VAL = -1
    __FILE_NAME = "ps_weed"

    selectable_ps = np.array([])

    def __init__(self, path_to_patch: str, ps_files: PsFiles, ps_est_gamma: PsEstGamma,
                 ps_select: PsSelect):
        self.__ps_files = ps_files
        self.__ps_select = ps_select
        self.__ps_est_gamma = ps_est_gamma

        self.__logger = LoggerFactory.create("PsWeed")

        self.__time_win = 730
        self.__weed_standard_dev = 1
        self.__weed_max_noise = sys.maxsize  # In StaMPS this is inf
        self.__weed_zero_elevation = False
        self.__weed_neighbours = True
        # todo drop_ifg_index on juba PsSelect'is
        self.__drop_ifg_index = np.array([])

        self.__ps_weed_edge_data = self.__load_psweed_edge_file(path_to_patch)
        self.__logger.debug("self.__ps_weed_edge_data.len: " + str(len(self.__ps_weed_edge_data)))

    def __load_psweed_edge_file(self, path: str) -> (int, np.ndarray):
        """We load this file here because our process may not reach in this step and this file's
        data is needed only here.


        In StaMPS also where read how large is array (for header) but I don't see a point for that"""
        # todo Maybe use @lazy and put this to PsFiles class

        file_name = "psweed.2.edge"
        psweed_path = Path(path, FolderConstants.PATCH_FOLDER_NAME, file_name)
        self.__logger.debug("Path to psweed edge file: " + str(psweed_path))
        if psweed_path.exists():
            data = np.genfromtxt(psweed_path, skip_header=True, dtype=self.__IND_ARRAY_TYPE)
            return data
        else:
            raise FileNotFoundError("File named '{1}' not found. AbsPath '{0}'".format(
                str(psweed_path.absolute()), file_name))

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
        # In Stamps this is called 'nr_ps' but we already have that named variable
        coh_thresh_ind_len = len(data.coh_thresh_ind)
        if coh_thresh_ind_len == 0:
            self.__logger.warn("coh_thresh_ind is empty")
        self.__logger.debug("Loaded data. coh_thresh_ind.len: {0}, data.nr_ps: {1}"
                            .format(coh_thresh_ind_len, data.nr_ps))

        ij_shift = self.__get_ij_shift(data.pscands_ij, coh_thresh_ind_len)
        self.__logger.debug("ij_shift.len: {0}".format(len(ij_shift)))

        neighbour_ind = self.__init_neighbours(ij_shift, coh_thresh_ind_len)
        self.__logger.debug("neighbours.len: {0}".format(len(neighbour_ind)))

        neighbour_ps = self.__find_neighbours(ij_shift, coh_thresh_ind_len, neighbour_ind)
        # todo kas saab logida ka tühjade arvu?
        self.__logger.debug("neighbour_ps.len: {0}".format(len(neighbour_ps)))

        # 'ix_weed' in StaMPS
        selectable_ps = self.__select_best(neighbour_ps, coh_thresh_ind_len, data.coh_ps, data.hgt)
        self.__logger.debug("selectable_ps.len: {0}, true vals: {1}"
                            .format(len(selectable_ps), np.count_nonzero(selectable_ps)))
        del neighbour_ps

        selectable_ps = self.__filter_xy(data.xy, selectable_ps, data.coh_ps)

        # In PsWeed we make our own interferograms array.
        # In Stamps this variable is called 'ifg_index'
        ifg_ind = np.arange(0, data.nr_ifgs, dtype=self.__IND_ARRAY_TYPE)
        if len(self.__drop_ifg_index) > 0:
            self.__logger.debug("Dropping indexes {0}".format(self.__drop_ifg_index))
            np.setdiff1d(ifg_ind, self.__drop_ifg_index)

        # In Stamps there is parameter 'no_weed_noisy' that is calculated similarly
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

        # Results to class variables
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
        """Because filtering is made using selectable_ps values and we already have all those
        parameters (also class privates and parameters that are loaded only or this process) that
        are filtered here we can do that filtering in this class.

        In StaMPS they made new .mat files for saving results."""

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

        bperp = self.__ps_files.bperp[data.coh_thresh_ind]
        bperp = bperp[self.selectable_ps]

        sort_ind = self.__ps_files.sort_ind[data.coh_thresh_ind]
        sort_ind = sort_ind[self.selectable_ps]

        return coh_ps, k_ps, c_ps, ph_patch, ph, xy, pscands_ij, lonlat, hgt, bperp, sort_ind

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
            # Lets use get_variables function to get parameters
            ph, _, nr_ifgs, nr_ps, xy, _ = ps_files.get_ps_variables()

            # And then filter based on coh_thresh
            pscands_ij = ps_files.pscands_ij[coh_thresh_ind]
            xy = xy[coh_thresh_ind]
            ph = ph[coh_thresh_ind]
            lonlat = ps_files.lonlat[coh_thresh_ind]
            hgt = ps_files.hgt[coh_thresh_ind]

            # And then get parameters that are not filtered
            master_nr = ps_files.master_nr
            ifg_dates = ps_files.ifg_dates
            bperp_meaned = ps_files.bperp_meaned
            master_date = ps_files.master_date

            return pscands_ij, xy, ph, lonlat, hgt, nr_ifgs, nr_ps, master_nr, ifg_dates, \
                   bperp_meaned, master_date

        def get_from_ps_est_gamma(ps_est_gamma: PsEstGamma):
            ph_patch = ps_est_gamma.ph_patch[coh_thresh_ind, :]

            return ph_patch

        ind, ph_res, coh_thresh_ind, k_ps, c_ps, coh_ps = get_from_ps_select(self.__ps_select)

        pscands_ij, xy, ph, lonlat, hgt, nr_ifgs, nr_ps, master_nr, ifg_dates, bperp_meaned,\
        master_date = get_from_ps_files(self.__ps_files, coh_thresh_ind)

        ph_patch_org = get_from_ps_est_gamma(self.__ps_est_gamma)

        # In Stamps there also is param 'all_da_flag' and found variables k_ps, c_ps, coh_ps,
        # ph_patch_org, ph_res

        return self.__DataDTO(ind, ph_res, coh_thresh_ind, k_ps, c_ps, coh_ps, pscands_ij, xy,
                              lonlat, hgt, ph, ph_patch_org, bperp_meaned, nr_ifgs, nr_ps,
                              master_date, master_nr, ifg_dates)

    def __get_ij_shift(self, pscands_ij: np.matrix, coh_ps_len: int) -> np.ndarray:
        ij = np.asarray(pscands_ij[:, 1:3])
        repmated = np.matlib.repmat(np.array([2, 2]) - ij.min(axis=0), coh_ps_len, 1)
        ij_shift = ij + repmated

        return ij_shift

    def __init_neighbours(self, ij_shift: np.ndarray, coh_ps_len: int) -> np.ndarray:
        """In StaMPS the init value is zero, I use -1 (DEF_NEIGHBOUR_VAL). Because 0 is correct
        index value in Python, but not in Matlab. -1 is not index in Python"""

        def arange_neighbours_select_arr(i, ind):
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

            # To get len(start) * len(end) array in Numpy we need to select it like that.
            # You can use neighbour_ind[start, :][:, end] but then you need to add values some other
            # way
            neighbours_val = neighbour_ind[np.ix_(start, end)]
            neighbours_val[(neighbours_val == self.__DEF_NEIGHBOUR_VAL) & (miss_middle == True)] = i

            neighbour_ind[np.ix_(start, end)] = neighbours_val

        return neighbour_ind

    def __find_neighbours(self, ij_shift: np.ndarray, coh_thresh_ind_len: int,
                          neighbour_ind: np.ndarray) -> np.ndarray:
        # List of empty Numpy arrays
        neighbour_ps = [np.array([], self.__IND_ARRAY_TYPE)] * (coh_thresh_ind_len + 1)
        for i in range(coh_thresh_ind_len):
            ind = neighbour_ind[ij_shift[i, 0] - 1, ij_shift[i, 1] - 1]
            if ind != self.__DEF_NEIGHBOUR_VAL:
                neighbour_ps[ind] = np.append(neighbour_ps[ind], [i])

        return np.array(neighbour_ps)

    def __select_best(self, neighbour_ps: np.ndarray, coh_thresh_ind_len: int,
                      coh_ps: np.ndarray, htg: np.ndarray) -> np.ndarray:
        """
        Returns boolean array what is used to filter other arrays. In StaMPS it is array of int's.
        """

        selectable_ps = np.ones(coh_thresh_ind_len, dtype=bool)

        for i in range(coh_thresh_ind_len):
            ps_ind = neighbour_ps[i]
            if len(ps_ind) != 0:
                j = 0
                while j < len(ps_ind):
                    ps_i = ps_ind[j]
                    ps_ind = np.append(ps_ind, neighbour_ps[ps_i]).astype(self.__IND_ARRAY_TYPE)
                    neighbour_ps[ps_i] = np.array([]) # Empty the read array elements
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

    def __filter_xy(self, xy: np.ndarray, selectable_ps: np.ndarray, coh_ps: np.ndarray) -> np.ndarray:
        """Find xy array filtered.
        In StaMPS there is also logic to find if duplicates array is empty and when it is
        it found weeded_xy array again. Here that kind of logic isn't because I didn't find point
        for that"""

        weeded_xy = xy[selectable_ps] # 'xy_weed' Stamps

        weed_ind = np.flatnonzero(selectable_ps) # 'ix_weed_num' in StaMPS
        unique_rows = np.unique(weeded_xy, return_index=True, axis=0)[1].astype(self.__IND_ARRAY_TYPE)
        # In Stamps there is also additional transposing but in this case this does not do anything
        last = np.arange(0, len(weed_ind))
        # In Stamps this is called 'dps'. Pixels that have same lon/ lat
        duplicates = np.setxor1d(unique_rows, last)

        for duplicate in duplicates:
            weeded_duplicates_ind = np.where((weeded_xy[:, 0] == weeded_xy[duplicate, 0]) &
                                      ((weeded_xy[:, 1]) == weeded_xy[duplicate, 1])) # 'dups_ix_weed' in StaMPS
            duplicates_ind = weed_ind[weeded_duplicates_ind]
            high_coh_ind = coh_ps[duplicates_ind].argmax()
            selectable_ps[duplicates_ind != high_coh_ind] = False

        return selectable_ps

    def __drop_noisy(self, data: __DataDTO, selectable_ps: np.ndarray, ifg_ind: np.ndarray,
                     edges: np.ndarray) -> (np.ndarray, np.ndarray):

        def get_ph_weed(bperp: np.ndarray, k_ps: np.ndarray, ph: np.ndarray, c_ps: np.ndarray,
                        master_nr: int):
            exped = np.exp(-1j * (k_ps * bperp.conj().transpose()))
            ph_weed = np.multiply(ph, exped)
            ph_weed = np.divide(ph_weed, np.abs(ph_weed))
            # Adding master noise. It is done when small_baseline_flag != 'y'. Reshape is needed
            # because 'c_ps' is array of array's
            ph_weed[:, (master_nr - 1)] = np.exp(1j * c_ps).reshape(len(ph_weed))

            return ph_weed

        def get_time_deltas_in_days(index: int) -> np.ndarray:
            """For getting days in ints from date object"""
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

        #todo drop_ifg_index logic

        # This all is made when small_baseline_flag != 'y'

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
            # 'm' in Stamps
            weighted_least_sqrt = MatlabUtils.lscov(G, dph_mean_adj.conj().transpose(),
                                                    weight_factor)
            #todo Find better name
            least_sqrt_G = np.asarray((np.asmatrix(G) * np.asmatrix(weighted_least_sqrt))
                                      .conj().transpose())
            dph_mean_adj = np.angle(np.exp(1j * (dph_mean_adj - least_sqrt_G)))
            # 'm2' in Stamps
            weighted_least_sqrt2 = MatlabUtils.lscov(G, dph_mean_adj.conj().transpose(),
                                                     weight_factor)

            # We don't make transpose for weighted_least_sqrt because it doesn't
            # do anything in this case
            dph_smooth_val_exp = np.exp(1j * (weighted_least_sqrt[0, :] + weighted_least_sqrt2[0, :]))
            dph_smooth[:, i] = np.multiply(dph_mean, dph_smooth_val_exp)
            weight_factor[i] = 0 # Let's make ourselves as zero

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


