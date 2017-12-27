from unittest import TestCase

import numpy as np

from scripts.funs.PsTopofit import PsTopofit


class TestPsTopofit(TestCase):
    def test_ps_topofit_fun(self):
        phase = np.array(
            [-0.976699430122930 - 0.214611796501365j, -0.0247590537401134 + 0.999693447641773j,
             0.564516193510588 + 0.825421993445968j, -0.959108655003691 - 0.283038138590915j,
             -0.925244947451622 + 0.379370250830564j])
        bperp_meaned = np.array(
            [16.9962832291, 63.4369473318, 21.5419811209, 54.5277511008, -47.8618831925])
        nr_trial_wraps = 0.0413031180032

        actual_phase_residual, actual_coherence_0, actual_static_offset, actual_k_0 = \
            PsTopofit.ps_topofit_fun(phase, bperp_meaned, nr_trial_wraps)

        # Väärtused kopeeritud Matlab'ist
        expected_phase_residual = np.array([-0.956075301670529 - 0.293121165280857j,
                                            -0.321975606737151 + 0.946747964701400j,
                                            0.476734083633067 + 0.879047560432619j,
                                            -0.853853155561476 - 0.520513965939157j,
                                            -0.815175873258845 + 0.579213514739323j])
        expected_coherence_0 = 0.587710117439414
        expected_static_offset = 2.569312122050138
        expected_k_0 = -0.004777245862620

        np.testing.assert_array_almost_equal(expected_coherence_0, actual_coherence_0[0])
        np.testing.assert_array_almost_equal(expected_phase_residual,
                                             np.squeeze(actual_phase_residual))
        np.testing.assert_array_almost_equal(expected_static_offset, actual_static_offset)
        np.testing.assert_array_almost_equal(expected_k_0, actual_k_0[0])
