import numpy as np

import math

from scripts.utils.ArrayUtils import ArrayUtils
from scripts.utils.MatlabUtils import MatlabUtils

# Todo klassile testid!
class PsTopofit:
    """Topofit arvutamise objekt. Põhiprotsess toimub topofit ps_topofit_loop funkstioonis"""

    k_ps = np.ndarray
    c_ps = np.ndarray
    coh_ps = np.ndarray
    n_opt = np.ndarray
    ph_res = np.ndarray

    def __init__(self, sw_array_shape, nr_ps, nr_ifg):
        self.__nr_ps = nr_ps

        self.k_ps = np.zeros(sw_array_shape)
        self.c_ps = np.zeros(sw_array_shape)
        self.coh_ps = np.zeros(sw_array_shape)
        self.n_opt = np.zeros(sw_array_shape)
        self.ph_res = np.zeros((nr_ps, nr_ifg))

    def ps_topofit_loop(self, ph: np.ndarray, ph_patch: np.ndarray, bprep: np.ndarray,
                        nr_trial_wraps: float, ifg_ind=None):
        """Tegemist on topofit'i leidmise for loopiga, kus täidetakse ära kõik konstruktoris tehtud
        sisemised muutujad.
        ifg_ind on massiiv mis leitakse PsSelect'is ja seal tehakse tegevusi natuke omamoodi, aga üldiselt
        samamoodi"""

        for i in range(self.__nr_ps):
            psdph = ph[i, :] * ph_patch[i, :].conj()

            if np.count_nonzero(np.isnan(psdph)) == 0 and np.count_nonzero(psdph == 0) == 0:
                if ifg_ind is not None:
                    psdph = psdph / np.abs(psdph)
                    psdph = psdph[ifg_ind]

                phase_residual, coh_0, static_offset, k_0 = self.ps_topofit_fun(psdph,
                                                                       bprep[i, ifg_ind].conj().transpose(),
                                                                       nr_trial_wraps)
                self.k_ps[i] = k_0[0]
                self.c_ps[i] = static_offset[0]
                self.coh_ps[i] = coh_0[0]
                self.n_opt[i] = len(k_0)
                if psdph is None:
                    self.ph_res[i, :] = np.angle(phase_residual).transpose()
                else:
                    self.ph_res[i, ifg_ind] = np.angle(phase_residual).transpose()
            else:
                self.k_ps[i] = np.nan
                self.coh_ps[i] = 0

    @staticmethod
    def ps_topofit_fun(phase: np.ndarray, bperp_meaned: np.ndarray, nr_trial_wraps: float):
        # Et edasipidi oleksid tulemid õiged teeme bperp'i veerumaatriksiks
        bperp_meaned = ArrayUtils.to_col_matrix(bperp_meaned)

        # Siin ei saa get_nr_trial_wraps leitut kasutada, kuna seal oli see üldisem
        bperp_range = np.amax(bperp_meaned) - np.amin(bperp_meaned)

        CONST = 8 * nr_trial_wraps  # todo aga mis const? Miks see 8 on?
        trial_multi_start = -np.ceil(CONST)
        trial_multi_end = np.ceil(CONST)
        trial_multi = ArrayUtils.arange_include_last(trial_multi_start, trial_multi_end, 1)

        trial_phase = bperp_meaned / bperp_range * math.pi / 4
        trial_phase = np.exp(np.outer(-1j * trial_phase, trial_multi))

        # Selleks, et korrutamine õnnestuks teeme ta veeruvektoriks
        phase = ArrayUtils.to_col_matrix(phase)
        phase_tile = np.tile(phase, (1, len(trial_multi)))
        phaser = np.multiply(trial_phase, phase_tile)

        phaser_sum = MatlabUtils.sum(phaser)

        phase_abs_sum = MatlabUtils.sum(np.abs(phase))
        trial_coherence = np.abs(phaser_sum) / phase_abs_sum
        trial_coherence_max_ind = np.where(trial_coherence == MatlabUtils.max(trial_coherence))

        k_0 = (math.pi / 4 / bperp_range) * trial_multi[trial_coherence_max_ind][0]

        re_phase = np.multiply(phase, np.exp(-1j * (k_0 * bperp_meaned)))
        phase_offset = MatlabUtils.sum(re_phase)
        re_phase = np.angle(re_phase * phase_offset.conjugate())
        weigth = np.abs(phase)
        mopt = np.linalg.lstsq(weigth * bperp_meaned, weigth * re_phase)[0][0]
        k_0 = k_0 + mopt

        phase_residual = np.multiply(phase, np.exp(-1j * (k_0 * bperp_meaned)))
        phase_residual_sum = MatlabUtils.sum(phase_residual)
        static_offset = np.angle(phase_residual_sum)
        coherence_0 = np.abs(phase_residual_sum) / MatlabUtils.sum(np.abs(phase_residual))

        return phase_residual, coherence_0, static_offset, k_0