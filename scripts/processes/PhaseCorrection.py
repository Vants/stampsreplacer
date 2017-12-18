import numpy as np
import numpy.matlib
import os

from scripts.MetaSubProcess import MetaSubProcess
from scripts.processes.PsFiles import PsFiles
from scripts.processes.PsWeed import PsWeed
from scripts.utils.internal.LoggerFactory import LoggerFactory
from scripts.utils.internal.ProcessDataSaver import ProcessDataSaver


class PhaseCorrection(MetaSubProcess):
    """Tehtud 'ps_correct_phase' järgi"""

    __FILE_NAME = "phase_correction"

    def __init__(self, ps_files: PsFiles, ps_weed: PsWeed):
        self.__logger = LoggerFactory.create("PsWeed")

        self.__ps_files = ps_files
        self.__ps_weed = ps_weed

    class __DataDTO(object):
        def __init__(self, master_nr: int, nr_ifgs: int, bperp: np.ndarray, ph: np.ndarray,
                     k_ps: np.ndarray,
                     c_ps: np.ndarray, ph_patch: np.ndarray):
            self.master_nr = master_nr
            self.nr_ifgs = nr_ifgs
            self.bperp = bperp
            self.ph = ph
            self.k_ps = k_ps
            self.c_ps = c_ps
            self.ph_patch = ph_patch

    def start_process(self):
        self.__logger.info("Started")

        data = self.__load_ps_params()

        ph_rc = self.get_ph_rc(data)
        self.__logger.debug("ph_rc.len {0}".format(len(ph_rc)))

        ph_reref = self.get_ph_reref(data)
        self.__logger.debug("ph_reref.len {0}".format(len(ph_reref)))

        # Paneme arvutatud väljad klassimuutujatesse
        self.ph_rc = ph_rc
        self.ph_reref = ph_reref

    def save_results(self, save_path: str):
        ProcessDataSaver(save_path, self.__FILE_NAME).save_data(
            ph_rc = self.ph_rc,
            ph_reref = self.ph_reref
        )

    def load_results(self, load_path: str):
        file_with_path = os.path.join(load_path, self.__FILE_NAME + ".npz")
        data = np.load(file_with_path)

        self.ph_rc = data['ph_rc']
        self.ph_reref = data['ph_reref']

    def __load_ps_params(self) -> __DataDTO:
        master_nr = self.__ps_files.master_nr - 1  # Stamps'is oli see master_ix

        nr_ifgs = len(self.__ps_files.ifgs)

        _, k_ps, c_ps, ph_patch, ph, _, _, _, _, bperp = self.__ps_weed.get_filtered_results()

        return self.__DataDTO(master_nr, nr_ifgs, bperp, ph, k_ps, c_ps, ph_patch)

    def get_ph_rc(self, data: __DataDTO):
        bperp = data.bperp
        master_nr = data.master_nr
        # Justkui pistame vahele
        bperp_master_col_zeros = np.insert(bperp, master_nr, values=0, axis=1)

        nr_ifgs = data.nr_ifgs
        repmated_bperp = np.multiply(np.matlib.repmat(data.k_ps, 1, nr_ifgs),
                                     bperp_master_col_zeros)
        repmated_c_ps = np.matlib.repmat(data.c_ps, 1, nr_ifgs)
        ph_rc = np.multiply(data.ph, np.exp(-1j * (repmated_bperp + repmated_c_ps)))

        return ph_rc

    def get_ph_reref(self, data: __DataDTO):
        ph_patch = data.ph_patch
        master_nr = data.master_nr

        ph_reref = np.insert(ph_patch, master_nr, values=1, axis=1)

        return ph_reref
