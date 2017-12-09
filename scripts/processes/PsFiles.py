import os
from datetime import date
from pathlib import Path

import re
from typing import Callable

from numpy import matlib

from scripts.MetaSubProcess import MetaSubProcess
from scripts.utils.FolderConstants import FolderConstants

import numpy as np
import math

from scripts.utils.LoggerFactory import LoggerFactory
from scripts.utils.MatlabUtils import MatlabUtils
from scripts.utils.MatrixUtils import MatrixUtils
from scripts.utils.ProcessDataSaver import ProcessDataSaver


class PsFiles(MetaSubProcess):
    """Siin täidame ära kõik muutujad mida võib pärast vaja minna.
    Tehtud StaMPS'i ps_load_inital_gamma järgi."""

    heading = None
    mean_range = 0.0
    wavelength = 0.0
    mean_incidence = 0.0
    master_nr = -1 # Stamps'is oli master_ix
    bperp_meaned = np.ndarray
    bperp = np.ndarray # Stamps'is oli see bperp_mat
    ph = np.ndarray
    ll = np.ndarray
    xy = np.ndarray
    da = np.ndarray
    sort_ind = np.ndarray # Stamps'is oli see la1.mat olev la
    master_date = date
    ifgs = np.ndarray # todo privaatseks muutujaks?
    hgt = np.ndarray
    ifg_dates = [] # Stamps'is oli 'day'

    def __init__(self, path: str, pscands_ij_array: np.ndarray, lonlat: np.ndarray):
        # Parameetrid mis failidest sisse loetakse ja pärast läheb edasises töös vaja
        self.__FILE_NAME = "ps_files"
        self.__params = {}
        self.__rg = None

        self.__path = Path(path)
        self.__patch_path = Path(path, FolderConstants.PATCH_FOLDER_NAME)

        self.pscands_ij = np.asmatrix(pscands_ij_array)
        self.lonlat = np.asmatrix(lonlat)

        if not self.__path.exists():
            raise FileNotFoundError("No PATCH folder. Load abs.path " + str(self.__path.absolute()))
        if not self.__patch_path.exists():
            raise FileNotFoundError(
                "No PATCH folder. Load abs.path " + str(self.__patch_path.absolute()))
        if self.pscands_ij is None:
            raise AttributeError("pscands_ij_array is None")

        self.__logger = self.__logger = LoggerFactory.create("PsFiles")

    def start_process(self):
        self.__logger.info("Start")

        self.__load_params_from_rsc_file()

        # Parameetrid mida läheb väljaspool seda protsessi tarvis. Matlab'is loeti need param'itesse
        self.heading = float(self.__params['heading'])
        self.mean_range = float(self.__params['center_range_slc'])

        self.wavelength = self.__get_wavelength()

        self.ifgs = self.__load_ifg_info_from_pscphase()

        self.master_date = self.__get_master_date()
        self.master_nr = self.__get_nr_ifgs_less_than_master(self.master_date, self.ifgs)

        self.ifg_dates = self.__get_ifg_dates()

        self.__rg = self.__get_rg()

        sat_look_angle = self.__get_look_angle()

        self.bperp_meaned, self.bperp = self.__get_bprep(self.ifgs, sat_look_angle)

        self.mean_incidence = self.__get_meaned_incidence()

        self.ph = self.__get_ph(len(self.ifgs))

        self.ll = self.__get_ll_array()

        self.xy, sort_ind = self.__get_xy()

        self.da = self.__get_da()

        self.hgt = self.__get_hgt()

        self.__sort_results(sort_ind, sat_look_angle)

        self.__logger.info("End")

    def save_results(self, save_path: str):
        ProcessDataSaver(save_path, self.__FILE_NAME).save_data(
            heading=self.heading,
            mean_range=self.mean_range,
            wavelength=self.wavelength,
            mean_incidence=self.mean_incidence,
            master_nr=self.master_nr,
            bprep_meaned=self.bperp_meaned,
            bperp=self.bperp,
            ph=self.ph,
            ll=self.ll,
            xy=self.xy,
            da=self.da,
            sort_ind=self.sort_ind,
            master_date=self.master_date,
            ifgs=self.ifgs,
            hgt=self.hgt,
            ifg_dates=self.ifg_dates)

    def load_results(self, load_path: str):
        file_with_path = os.path.join(load_path, self.__FILE_NAME + ".npz")
        data = np.load(file_with_path)

        self.heading = data['heading']
        self.mean_range = data['mean_range']
        self.wavelength = data['wavelength']
        self.mean_incidence = data['mean_incidence']
        self.master_nr = data['master_nr']
        self.bperp_meaned = data['bprep_meaned']
        self.bperp = data['bperp']
        self.ph = data['ph']
        self.ll = data['ll']
        self.xy = data['xy']
        self.da = data['da']
        self.sort_ind = data['sort_ind']
        self.master_date = data['master_date']
        self.ifgs = data['ifgs']
        self.hgt = data['hgt']
        self.ifg_dates = data['ifg_dates']

    def __get_wavelength(self):
        velocity = 299792458  # Signaali levimise kiirus (m/s)
        freg = float(self.__params['radar_frequency']) * math.pow(10, 9)  # Signaali sagedus (GHz)
        return velocity / freg

    def __get_bprep(self, ifgs: np.ndarray, sat_look_angle: np.ndarray):
        """Leitakse bprep_meaned ja bprep_arr. StaMPS'is vastavalt bperp ja bperp_mat. 
        Salvestan mõlemad igaksjuhuks.
        
        StaMPS'is loeti terve ij fail kohaliku muutujasse ja tehti tehteid maatriksi
        kolmanda veeruga. Python'is on selline protsess liiga aeglane (ligi 30 sekundit)"""
        ARRAY_TYPE = np.float64



        cos_sat_look_angle = np.cos(sat_look_angle)
        sin_sat_look_angle = np.sin(sat_look_angle)

        mean_azimuth_line = float(self.__params['azimuth_lines']) / 2 - 0.5

        ij_lon = self.pscands_ij[:, 1]
        nr_ifgs = len(ifgs)
        bperp = matlib.zeros((len(self.pscands_ij), nr_ifgs), dtype=ARRAY_TYPE)

        bc_bn_formula = lambda tcn, baseline_rate: tcn + baseline_rate * (
            ij_lon - mean_azimuth_line) / float(self.__params['prf'])

        for i in range(nr_ifgs):
            tcn, baseline_rate = self.__get_baseline_params(ifgs[i])

            bc = bc_bn_formula(tcn[1], baseline_rate[1])
            bn = bc_bn_formula(tcn[2], baseline_rate[2])
            bprep_line = np.multiply(bc, cos_sat_look_angle) - np.multiply(bn, sin_sat_look_angle)
            bperp[:, i] = bprep_line

        bprep_meaned = np.mean(bperp, 0).transpose()
        # Kustutame püsivpeegeldajate asukohast veeru
        bperp = MatrixUtils.delete_master_col(bperp, self.master_nr)

        return bprep_meaned, bperp

    def __get_ph(self, nr_ifgs):
        """pscands.1.ph lugemine. Tegemist on binaarkujul failiga kus on kompleksarvud"""
        BINARY_COMPLEX_TYPE = np.dtype('>c8')  # "big-endian" 64bit kompleksarvud

        COMPLEX_TYPE = np.complex64
        imag_array_raw = np.fromfile(self.__patch_path.joinpath("pscands.1.ph").open("rb"),
                                     BINARY_COMPLEX_TYPE)
        imag_mx_len = int(len(imag_array_raw) / nr_ifgs)
        imag_list = []
        count = 0
        for i in range(0, len(imag_array_raw), imag_mx_len):
            matrix_row = imag_array_raw[i:i + imag_mx_len]
            if count == self.master_nr - 1:
                matrix_row = np.ones((imag_mx_len), dtype=COMPLEX_TYPE)
            imag_list.append(matrix_row)

            count += 1

        return np.asarray(imag_list, COMPLEX_TYPE).transpose()

    def __get_meaned_incidence(self):
        sar_to_earth_center_sq = math.pow(float(self.__params['sar_to_earth_center']), 2)
        earth_radius_below_sensor_sq = math.pow(float(self.__params['earth_radius_below_sensor']),
                                                2)

        rg = self.__rg

        incidence = np.arccos(
            np.divide(
                (sar_to_earth_center_sq - earth_radius_below_sensor_sq - np.power(rg, 2)),
                (2 * float(self.__params['earth_radius_below_sensor']) * rg)))
        return incidence.mean()

    def __get_baseline_params(self, ifg_name: str):
        """Tagastab kaks muutujat tcn (initial baseline) ja baseline_rate. 
        Need leitakse inteferogrammile vastavast .base failist. Igasühes on 
        np.narray kolme muutujaga"""

        name_and_ext = ifg_name.split(".")
        base_file_name = name_and_ext[0] + ".base"
        path = Path(base_file_name)

        if path.exists():
            tcn = None
            baseline_rate = None
            with path.open() as basefile:
                for line in basefile:
                    splited = line.split('	')
                    if splited[0] == "initial_baseline(TCN):":
                        tcn = np.array((
                            splited[1], splited[2], splited[3]), dtype=np.float64)
                    elif splited[0] == "initial_baseline_rate:":
                        baseline_rate = np.array((
                            splited[1], splited[2], splited[3]), dtype=np.float64)
                    else:
                        break

            return tcn, baseline_rate
        else:
            raise FileNotFoundError(base_file_name + " not found.")

    def __load_params_from_rsc_file(self):
        """Esimesest failist loetakse teise faili asukoht, kus on sateliidi metadata.
        Lubatud parameetrid salvestame self.__params'i."""
        # TODO tagastada dict mitte globaalsesse salvestada

        ALLOWED_PARAMS = ["azimuth_lines",
                          "heading",
                          "range_pixel_spacing",
                          "azimuth_pixel_spacing",
                          "radar_frequency",
                          "prf",
                          "sar_to_earth_center",
                          "earth_radius_below_sensor",
                          "near_range_slc",
                          "center_range_slc",
                          "date"]

        value_regex = re.compile(r"-?[\d*.]+")
        with self.__load_file("rsc.txt", self.__path) as rsc_file:
            rsc_par_file_abs_path = rsc_file.read()

            rsc_par_file = Path(rsc_par_file_abs_path)
            if rsc_par_file.exists():
                with rsc_par_file.open() as rsc_par:
                    for line in rsc_par:
                        # Sellest parameetrist alates pole meil enam vaja
                        if line == "state_vector_position_1":
                            break

                        splited = line.split(':')
                        key = splited[0]

                        if key in ALLOWED_PARAMS:
                            if key == 'date':
                                # Kui on näiteks mitte komaga või üldse sõne siis puhastab tühikutest
                                value = re.sub("[\t\n\v]", "", splited[1])
                            else:
                                value = value_regex.findall(splited[1])[0]

                            self.__params[key] = value
            else:
                raise FileNotFoundError(
                    "No file. Abs.path " + str(rsc_par_file.absolute()))

    def __get_master_date(self):
        """load_params_from_rsc_file saadud 'date' on masteri kuupäev. Selle splitime ja teeme
        datetime'iks"""
        date_arr = self.__params["date"].split('  ')
        return date(int(date_arr[0]), int(date_arr[1]), int(date_arr[2]))

    def __load_file(self, name: str, path: Path):
        file_path = Path(path, name)

        if file_path.exists():
            return file_path.open()
        else:
            raise FileNotFoundError("No file " + name + ". Abs.path " + str(file_path.absolute()))

    def __load_ifg_info_from_pscphase(self):
        """pscphase.in failis on inteferogammide asukohad (kataloogiteed). Need teed ka tagastatakse
        siin fuksioonis. Failinimes on mis on masteri kuupäev ja mis on inteferogammi pildi kuupäev."""

        path = self.__path.joinpath("pscphase.in")
        if path.exists():
            pscphase = np.genfromtxt(str(path), dtype=str)
            return pscphase[1:]  # Esimest rida pole vaja
        else:
            raise FileNotFoundError("pscphase.in not found. AbsPath " + str(path.absolute()))

    def __get_nr_ifgs_less_than_master(self, master_date: date, ifgs: np.ndarray):
        """Mitu intefegorammi on enne masteri kuupäeva"""
        comp_fun = lambda x, y: x > y
        return self.get_nr_ifgs_copared_to_master(comp_fun, ifgs, master_date)

    def __get_ll_array(self):
        return (MatlabUtils.max(self.lonlat) + MatlabUtils.min(self.lonlat)) / 2

    def __get_xy(self):
        """Võetakse ij massiivist x ja y väärtused (viimased kaks veergu). 
        Korrutatakse skalaaridega läbi, püütakse pilti parandada keeramise teel ning sorteeritakse 
        y järgi kogu massiv. Siin saadakse ka sorteerimise indeks massiiv mille järgi pärast teised
        sorteerida"""

        xy = np.fliplr(self.pscands_ij.copy())[:, 0:2]
        xy[:, 0] *= 20
        xy[:, 1] *= 4

        xy = self.__scene_rotate(xy)

        # Kuna maatriksitega sorteerimine ei toimi hästi siis peab läbi view tegema seda
        xy_view = xy.view(np.ndarray)
        sort_ind = np.lexsort((xy_view[:, 0], xy_view[:, 1]))
        sorted_xy = np.asmatrix(xy_view[sort_ind])

        # TODO Korrutame läbi et ümarada millimeetrini? Aga see on juba int
        sorted_xy = np.around(sorted_xy * 1000) / 1000

        return sorted_xy, sort_ind

    def __scene_rotate(self, xy: np.matrix):
        # TODO leida parem muutuja nimi
        theta = (180 - self.heading) * math.pi / 180
        if theta > math.pi:
            theta -= 2 * math.pi

        rotm = np.array([[math.cos(theta), math.sin(theta)], [-math.sin(theta), math.cos(theta)]])
        xy = xy.H

        rotated_xy = rotm * xy
        # Ei kasuta siin massiivi veergude järgi max'i, kuna siin on vaja suurimat elementi
        is_improved = np.amax(rotated_xy[0]) - np.amin(rotated_xy[0]) < np.amax(xy[0]) - np.amin(
            xy[0]) and np.amax(rotated_xy[1]) - np.amin(rotated_xy[1]) < np.amax(xy[1]) - np.amin(
            xy[1])
        if is_improved:
            xy = rotated_xy

        xy = xy.H
        return xy

    def __sort_results(self, sort_ind: np.ndarray, sat_look_angle: np.ndarray):
        self.ph = self.ph[sort_ind]
        self.bperp = self.bperp[sort_ind]
        self.da = self.da[sort_ind]

        self.pscands_ij = MatrixUtils.sort_matrix_with_sort_array(self.pscands_ij, sort_ind)
        self.lonlat = MatrixUtils.sort_matrix_with_sort_array(self.lonlat, sort_ind)

        self.sort_ind = sat_look_angle[sort_ind]
        self.hgt = self.hgt[sort_ind]

    def __get_da(self):
        """Kuna failis on vaid üks tulp siis on loadtxt piisavalt kiire"""
        return np.loadtxt(str(Path(self.__patch_path, "pscands.1.da")))

    def __get_look_angle(self):
        sar_to_earth_center_sq = math.pow(float(self.__params['sar_to_earth_center']), 2)
        earth_radius_below_sensor_sq = math.pow(float(self.__params['earth_radius_below_sensor']),
                                                2)
        return np.arccos(np.divide(
            sar_to_earth_center_sq + np.power(self.__rg, 2) - earth_radius_below_sensor_sq,
            2 * float(self.__params['sar_to_earth_center']) * self.__rg))

    def __get_hgt(self):
        FLOAT_TYPE = ">f4" # "big-endian" float32
        hgt_raw = np.fromfile(self.__patch_path.joinpath("pscands.1.hgt").open("rb"), FLOAT_TYPE)
        hgt = hgt_raw.conj().transpose()

        return hgt

    def __get_rg(self):
        ij_lat = self.pscands_ij[:, 2]
        return float(self.__params['near_range_slc']) + ij_lat * float(
            self.__params['range_pixel_spacing'])

    def __get_ifg_dates(self) -> []:
        ifgs = self.ifgs

        ifg_dates = []
        for ifg_path in ifgs:
            ifg_date_str_yyyymmdd = ifg_path[-13:-5]
            ifg_datetime = date(int(ifg_date_str_yyyymmdd[:4]),
                                int(ifg_date_str_yyyymmdd[4:6]),
                                int(ifg_date_str_yyyymmdd[6:8]))

            ifg_dates.append(ifg_datetime)

        return ifg_dates

    def get_ps_variables(self):
        """Meetod millega eksportida muutujad mida läheb PsEstGamma ja PsSelect is vaja"""

        nr_ifgs = len(self.ifgs)
        nr_ps = len(self.pscands_ij)

        return self.ph, self.bperp, nr_ifgs, nr_ps, self.xy, self.da

    def get_nr_ifgs_copared_to_master(self, comp_fun: Callable[[date, date], bool],
                                      ifgs=np.array([]), master_date=None):
        """Mitu intefegorammi on enne masteri kuupäeva juurde liidetud üks.
         Kuupäev saadakse failiteest (ifgs muutja)"""
        if ifgs is None or len(ifgs) == 0:
            ifgs = self.ifgs

        if master_date is None:
            master_date = self.master_date

        result = 1  # StaMPS'is liideti üks juurde peale töötlust
        for ifg_path in ifgs:
            ifg_date_str_yyyymmdd = ifg_path[-13:-5]
            ifg_datetime = date(int(ifg_date_str_yyyymmdd[:4]),
                                int(ifg_date_str_yyyymmdd[4:6]),
                                int(ifg_date_str_yyyymmdd[6:8]))

            if comp_fun(ifg_datetime, master_date):
                result += 1
        return result

