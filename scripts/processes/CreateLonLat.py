from pathlib import Path

import numpy as np
from snappy import ProductIO

from scripts.MetaSubProcess import MetaSubProcess


class CreateLonLat(MetaSubProcess):
    __PATCH = "PATCH_1"
    __ARRAY_TYPE = np.float32

    def __init__(self, path: str, geo_ref_product: str):
        self.path = Path(path)
        self.geo_ref_product = geo_ref_product

    def start_process(self, save_process=False):
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

        return np.reshape(lonlat, (len(lonlat), 2))

    # TODO kõige aeglasem koht
    # noinspection PyMethodMayBeStatic
    def __read_pixel(self, x, y, band, tmp_array):
        band.readPixels(x, y, 1, 1, tmp_array)

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
