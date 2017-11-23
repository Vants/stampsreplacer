import numpy as np
import numpy.matlib

from scripts.MetaSubProcess import MetaSubProcess
from scripts.processes.PsEstGamma import PsEstGamma
from scripts.processes.PsFiles import PsFiles
from scripts.processes.PsSelect import PsSelect


class PhaseCorrection(MetaSubProcess):
    """Tehtud 'ps_correct_phase' järgi"""

    def __init__(self, ps_files: PsFiles, ps_select: PsSelect, ps_est_gamma: PsEstGamma):
        self.__ps_files = ps_files
        self.__ps_select = ps_select
        self.__ps_est_gamma = ps_est_gamma

    def start_process(self):
        master_nr = self.__ps_files.master_nr - 1 # Stamps'is oli see master_ix
        ps_files_bperp = self.__ps_files.bperp
        # Justkui pistame vahele
        bperp = np.insert(ps_files_bperp, master_nr, values=0, axis=1)

        k_ps = self.__ps_est_gamma.k_ps
        c_ps = self.__ps_est_gamma.c_ps
        nr_ifgs = len(self.__ps_files.ifgs)

        repmated_bperp = np.multiply(-1j * np.matlib.repmat(k_ps, 1, nr_ifgs), bperp)
        repmated_c_ps = np.matlib.repmat(c_ps, 1, nr_ifgs)

        ph = self.__ps_files.ph
        ph_rc = np.multiply(ph, np.exp(repmated_bperp + repmated_c_ps))

        ph_patch = self.__ps_est_gamma.ph_patch
        ph_reref = np.insert(ph_patch, master_nr, values=0, axis=1)





