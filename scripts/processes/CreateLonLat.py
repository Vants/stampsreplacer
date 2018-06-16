import os
from pathlib import Path

import numpy as np
from snappy import ProductIO

from scripts.MetaSubProcess import MetaSubProcess
from scripts.utils.internal.FolderConstants import FolderConstants
from scripts.utils.internal.LoggerFactory import LoggerFactory

from scripts.utils.internal.ProcessDataSaver import ProcessDataSaver


class CreateLonLat(MetaSubProcess):
    __ARRAY_TYPE = np.float32

    # Needed in PsFiles process to load data
    pscands_ij = None
    lonlat = None

    def __init__(self, path: str, geo_ref_product: str):
        self.__FILE_NAME = "lonlat_process"
        self.__geo_ref_product = geo_ref_product

        self.__PATCH_FOLDER = Path(path, FolderConstants.PATCH_FOLDER_NAME)

        self.__logger = LoggerFactory.create('CreateLonLat')

        self.__logger.debug("PATCH_FOLDER {0}".format(self.__PATCH_FOLDER))

    def start_process(self):
        self.__logger.debug("Start")

        product_with_geo_ref = ProductIO.readProduct(self.__geo_ref_product)
        lon_band, lat_band = self.__get_lon_bands(product_with_geo_ref)

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

                # tmp__pixel_array is needed because readPixels returns x-size that much what is set
                # in third parameter. But there isn't any limiter for y-size and this means it takes
                # how much it can.
                # This needs optimization. That you load all rows in one take and set them to lonlat.
                tmp__pixel_array = np.zeros((1, 1), dtype=self.__ARRAY_TYPE)
                tmp_lonlat = np.zeros((1, 2), dtype=self.__ARRAY_TYPE)
                self.__read_pixel(x, y, lon_band, tmp__pixel_array)
                tmp_lonlat[0, 0] = tmp__pixel_array[0]
                self.__read_pixel(x, y, lat_band, tmp__pixel_array)
                tmp_lonlat[0, 1] = tmp__pixel_array[0]

                lonlat.append(tmp_lonlat)

                self.__add_to_pscands_array(int(line[0]), int(line[1]), int(line[2]))

        self.pscands_ij = np.reshape(self.pscands_ij, (len(self.pscands_ij), 3))

        self.__logger.debug("Done")

        self.lonlat = np.reshape(lonlat, (len(lonlat), 2))

    def save_results(self, save_path: str):
        if self.pscands_ij is None:
            raise ValueError("pscands is None")
        if self.lonlat is None:
            raise ValueError("pscands is None")

        ProcessDataSaver(save_path, self.__FILE_NAME).save_data(
            pscands_ij_array=self.pscands_ij,
            lonlat=self.lonlat)

    def load_results(self, load_path:str):
        file_with_path = os.path.join(load_path, self.__FILE_NAME + ".npz")
        data = np.load(file_with_path)

        self.pscands_ij = data["pscands_ij_array"]
        self.lonlat = data["lonlat"]

    # TODO slowest part
    # noinspection PyMethodMayBeStatic
    def __read_pixel(self, x, y, band, tmp_array):
        band.readPixels(x, y, 1, 1, tmp_array)

    def __add_to_pscands_array(self, arr1: int, arr2: int, arr3: int):
        """When we process this we append values to pscands"""
        if (self.pscands_ij is None):
            self.pscands_ij = []

        self.pscands_ij.append(np.array([arr1, arr2, arr3]))

    # noinspection PyMethodMayBeStatic
    def __get_lon_bands(self, product_with_geo_ref):
        lon_band = product_with_geo_ref.getBand('lon_band')
        lat_band = product_with_geo_ref.getBand('lat_band')

        return lon_band, lat_band

    def __load_pscands(self):
        if self.__PATCH_FOLDER.is_dir():
            # TODO Check if it is in this folder
            path_to_pscands = Path(self.__PATCH_FOLDER, "pscands.1.ij")
            if path_to_pscands.exists():
                return path_to_pscands.open()
            else:
                raise FileNotFoundError("Path {0}".format(path_to_pscands))
