import numpy as np
from cpbench.detectors import make_detector
from cpbench.detectors.dmd import dmd_errors
from cpbench.eval.metrics import cp_metrics


def test_dmd_reconstructs_sines_at_sufficient_rank():
    t = np.arange(40.0)
    one = np.sin(0.3 * t)[None, :, None]
    two = (np.sin(0.3 * t) + np.sin(0.9 * t + 1))[None, :, None]
    assert dmd_errors(one, d=5, r=2)[0] < 1e-12
    assert dmd_errors(two, d=5, r=2)[0] > 0.1 and dmd_errors(two, d=5, r=4)[0] < 1e-12


def test_cpdmd_alarms_on_frequency_changes():
    rng = np.random.default_rng(0); n = 3000
    S = (np.arange(n) // 300) % 2
    x = np.sin(2 * np.pi * np.cumsum(np.where(S == 0, 0.04, 0.0533))) + 0.1 * rng.standard_normal(n)
    cp = np.zeros((1, n), bool); cp[0, 1:] = S[1:] != S[:-1]
    out = make_detector("cpdmd", w=100, d=20, r=2, L=5).detect(x[None, :, None], 2)
    m = cp_metrics(out["cp_score"], np.arange(n) - 0.5, cp, 20, out["alarms"])
    assert m["f1"] > 0.9
    m = cp_metrics(make_detector("dmd_error", m=60, d=10, ranks=[2]).detect(x[None, :, None], 2)["cp_score"],
                   np.arange(n) - 0.5, cp, 20)
    assert m["ap"] > 0.5
