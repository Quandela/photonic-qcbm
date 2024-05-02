import numpy as np
import perceval as pcvl

from ..src.models import QCBM
from ..src.helpers import ParametrizedQuantumCircuit
from ..src.helpers.utils import gaussian_mixture_pdf, kl_divergence


def is_bm_ok(bm):
    # Initialize and get loss for the lossy QCBM with photon recycling
    _ = bm.pqc.init_params()
    value, _, _, _ = bm.get_loss()
    assert value is not None

    # Is the sum of the probabilities of the generated distribution equal to 1.0?
    assert abs(1 - sum(bm.pdf())) < 0.000001


def test_QCBM():
    # Input config
    input_state = pcvl.BasicState("|1, 1, 1, 0, 0, 0, 0, 0, 0>")
    pnr = False
    arch = ["var"]

    # Losses and number of samples
    sample_count = 20000
    loss_parameter = 0.8

    # QCBM lossy with photon recycling
    pqc = ParametrizedQuantumCircuit(input_state.m, arch)
    bm_lossy = QCBM(pqc, input_state=input_state, sample_count=sample_count, loss_parameter=loss_parameter, pnr=pnr,
                    use_samples_only=False, use_photon_recycling=True, miti=True)

    # Target and loss function
    bin_count = bm_lossy.bin_count
    target_pdf = gaussian_mixture_pdf(bin_count)
    target_space = np.arange(bin_count)
    loss_fun = kl_divergence()

    # Assign to the QCBM
    bm_lossy.target_pdf = target_pdf
    bm_lossy.loss_fun = loss_fun
    bm_lossy.target_space = target_space

    # Is the sum of the probabilities equal to 1.0?
    assert abs(1 - sum(target_pdf)) < 0.000001

    is_bm_ok(bm_lossy)

    # Now check that a lossless QCBM trains
    pqc = ParametrizedQuantumCircuit(input_state.m, arch)
    bm_lossless = QCBM(pqc, input_state=input_state, sample_count=sample_count, loss_parameter=0, pnr=pnr,
                       use_samples_only=False, use_photon_recycling=False)

    bm_lossless.target_pdf = target_pdf
    bm_lossless.loss_fun = loss_fun
    bm_lossless.target_space = target_space

    is_bm_ok(bm_lossless)
