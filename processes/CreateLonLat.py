import os
from snappy import ProductIO
from pathlib import Path
from MetaSubProcess import MetaSubProcess
import numpy as np


class CreateLonLat(MetaSubProcess):
    PATCH = "PATCH_1"
    ARRAY_TYPE = np.float32

    def __init__(self, path: str, geo_ref_product: str):
        self.path = Path(path)
        self.geo_ref_product = geo_ref_product

    def start_process(self, save_process=False):
        pscands = self.load_pscands()
        pscands_size = pscands.__sizeof__()

        if pscands_size == 0:
            raise FileNotFoundError("pscands file is empty")
        elif pscands_size is None:
            raise FileNotFoundError("pscands file is None")

        product_with_geo_ref = ProductIO.readProduct(self.geo_ref_product)
        lon_band, lat_band = self.get_lon_bans(product_with_geo_ref)

        if lon_band is None or lat_band is None:
            raise FileNotFoundError("lon_band, lat_band missing")

        lonlat = np.zeros((pscands_size, 2), dtype=self.ARRAY_TYPE)

        for row_num in range(pscands_size):
            line = pscands.readline().split(' ')
            x = int(line[1])
            y = int(line[2])

            # Ajutine on vajalik seepärast, et readPixels tagastab kolmanda parameertina kui palju
            # (massiivis x sihis), aga puudub parameeter mis piiraks massiivi y'it
            # ning see täidab palju on võimalik.
            # See aga võib olla üks koht kuidas optimeerida, et massiivi laetakse kõik väärtused ja
            # siis ainult täidab lonlat massiivi
            tmp_array = np.zeros((pscands_size, 1), dtype=self.ARRAY_TYPE)
            self.readPixel(x, y, lon_band, tmp_array)
            lonlat[row_num, 0] = tmp_array[0]
            self.readPixel(x, y, lat_band, tmp_array)
            lonlat[row_num, 1] = tmp_array[0]

        pscands.close()

        return lonlat

    def readPixel(self, x, y, band, tmp_array):
        band.readPixels(x, y, 1, 1, tmp_array)

    def get_lon_bans(self, product_with_geo_ref):
        lon_band = product_with_geo_ref.getBand('lon_band')
        lat_band = product_with_geo_ref.getBand('lat_band')

        return lon_band, lat_band

    def load_pscands(self):
        if self.path.is_dir():
            # TODO Kontroll kui juba on seal kaustas
            path_to_pscands = Path(os.path.join(self.path, self.PATCH), "pscands.1.ij")
            if path_to_pscands.exists():
                return path_to_pscands.open()
            else:
                return None
