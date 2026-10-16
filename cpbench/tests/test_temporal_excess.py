import numpy as np
from cpbench.data.regime import regime_path
from cpbench.detectors import make_detector
from cpbench.eval.metrics import best_perm_acc


def test_temporal_excess_finds_linear_regime_directions_label_free():
    rng = np.random.default_rng(1); R, T, K, d = 4, 1500, 4, 12
    S = regime_path(R, T, K, 40, rng)
    sig = rng.standard_normal((K, 3))[S] + 0.3 * rng.standard_normal((R, T, 3))
    e = rng.standard_normal((R, T, d - 3)); nui = np.zeros_like(e)
    for t in range(1, T):
        nui[:, t] = 0.9 * nui[:, t - 1] + e[:, t]                       # Gaussian AR nuisances
    Z = np.concatenate([sig, nui], -1) @ np.linalg.qr(rng.standard_normal((d, d)))[0]
    out = make_detector("temporal_excess", k=12).detect(Z, K)
    assert out["n_dirs"] >= 1 and out["excess"][0] > 5 * out["null"]
    assert best_perm_acc(S.ravel(), out["labels"].ravel(), K) > 0.9
