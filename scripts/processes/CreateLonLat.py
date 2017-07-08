import os
from pathlib import Path

import numpy as np
from snappy import ProductIO

from scripts.MetaSubProcess import MetaSubProcess
from scripts.utils.FolderConstants import FolderConstants
from scripts.utils.LoggerFactory import LoggerFactory

from scripts.utils.ProcessDataSaver import ProcessDataSaver


class CreateLonLat(MetaSubProcess):
    __PATCH = FolderConstants.PATCH_FOLDER
    __ARRAY_TYPE = np.float32

    # Seda on tarvis pärast PsFiles protsessis andmete laadisel
    pscands_ij_array = None

    def __init__(self, path: str, geo_ref_product: str):
        self.__FILE_NAME = "lonlat_process"
        self.path = Path(path)
        self.geo_ref_product = geo_ref_product

        self.logger = LoggerFactory.create('CreateLonLat')

    def start_process(self):
        self.logger.debug("Start")

        product_with_geo_ref = ProductIO.readProduct(self.geo_ref_product)
        lon_band, lat_band = self.__get_lon_bans(product_with_geo_ref)

        if lon_band is None or lat_band is None:
            raise FileNotFoundError("lon_band, lat_band missing")

        lonlat = []
        with self.__load_pscands() as pscands:
            for row in pscands:
                if row == '':
                    break

                line = row.split(' ')
                y = int(line[1])
                x = int(line[2])

                # Ajutine tmp__pixel_array on vajalik seepärast, et readPixels tagastab kolmanda
                # parameertina kui palju
                # (massiivis x sihis), aga puudub parameeter mis piiraks massiivi y'it
                # ning see täidab palju on võimalik.
                # See aga võib olla üks koht kuidas optimeerida,
                # et massiivi laetakse kõik väärtused ja
                # siis ainult täidab lonlat massiivi
                tmp__pixel_array = np.zeros((1, 1), dtype=self.__ARRAY_TYPE)
                tmp_lonlat = np.zeros((1, 2), dtype=self.__ARRAY_TYPE)
                self.__read_pixel(x, y, lon_band, tmp__pixel_array)
                tmp_lonlat[0, 0] = tmp__pixel_array[0]
                self.__read_pixel(x, y, lat_band, tmp__pixel_array)
                tmp_lonlat[0, 1] = tmp__pixel_array[0]

                lonlat.append(tmp_lonlat)

                self.__add_to_pscands_array(int(line[0]), int(line[1]), int(line[2]))

        self.pscands_ij_array = np.reshape(self.pscands_ij_array, (len(self.pscands_ij_array), 3))

        self.logger.debug("Done")

        return np.reshape(lonlat, (len(lonlat), 2))

    def save_results(self, lonlat: np.array):
        if self.pscands_ij_array is None:
            raise ValueError("pscands is None")
        if lonlat is None:
            raise ValueError("pscands is None")

        ProcessDataSaver(FolderConstants.SAVE_PATH, self.__FILE_NAME).save_data(
            pscands_ij_array=self.pscands_ij_array, lonlat=lonlat)

    def load_results(self):
        file_with_path = os.path.join(FolderConstants.SAVE_PATH, self.__FILE_NAME + ".npz")
        data = np.load(file_with_path)

        self.pscands_ij_array = data["pscands_ij_array"]
        lonlat = data["lonlat"]

        return lonlat

    # TODO kõige aeglasem koht
    # noinspection PyMethodMayBeStatic
    def __read_pixel(self, x, y, band, tmp_array):
        band.readPixels(x, y, 1, 1, tmp_array)

    def __add_to_pscands_array(self, arr1: int, arr2: int, arr3: int):
        """Samal ajal kui me käime läbi seda listi me täidame lisaks array algsete väärtustega"""
        if (self.pscands_ij_array is None):
            self.pscands_ij_array = []

        self.pscands_ij_array.append(np.array([arr1, arr2, arr3]))

    # noinspection PyMethodMayBeStatic
    def __get_lon_bans(self, product_with_geo_ref):
        lon_band = product_with_geo_ref.getBand('lon_band')
        lat_band = product_with_geo_ref.getBand('lat_band')

        return lon_band, lat_band

    def __load_pscands(self):
        if self.path.is_dir():
            # TODO Kontroll kui juba on seal kaustas
            path_to_pscands = Path(self.path.joinpath(self.__PATCH), "pscands.1.ij")
            if path_to_pscands.exists():
                return path_to_pscands.open()
            else:
                return None
