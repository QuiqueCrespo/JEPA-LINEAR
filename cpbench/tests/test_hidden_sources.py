"""Guarantees of the structured sources built to hide regimes: reversal (equal statistics up to third order across
regimes, opposite fourth-order time asymmetry) and gauss_switch (exactly N(0, I) marginal, heavy-tailed increments)."""
import numpy as np
from scipy import stats

from cpbench.data import Dataset


def _segments(x, S, k, L=128):
    return np.stack([x[r, t:t + L] for r in range(len(x)) for t in range(0, x.shape[1] - L, L) if (S[r, t:t + L] == k).all()])


def test_reversal_equal_up_to_third_order():
    ds = Dataset(dict(regime=dict(type="markov", K=4, dwell=640), sources=[dict(type="reversal")], seed=0))
    d = ds.sample(8, 32000, np.random.default_rng(0))
    ac, a3, a4 = [], [], []
    for k in range(4):
        a = _segments(d["obs"].astype(float), d["S_s"], k)[..., 0]
        ac.append([np.mean(a[:, :-l] * a[:, l:]) for l in (1, 5, 10, 20)])
        a3.append(np.mean(a[:, :-3] ** 2 * a[:, 3:] - a[:, :-3] * a[:, 3:] ** 2))
        a4.append(np.mean(a[:, :-8] ** 3 * a[:, 8:] - a[:, :-8] * a[:, 8:] ** 3))
    assert np.ptp(np.array(ac), 0).max() < 0.02
    assert np.abs(a3).max() < 0.02
    assert min(abs(v) for v in a4) > 0.5 and np.sign(a4[0]) == -np.sign(a4[1])     # bit 0 flips channel 0


def test_gauss_switch_exact_marginal_and_jumps():
    for coding in ("fresh", "bands"):
        ds = Dataset(dict(regime=dict(type="markov", K=4, dwell=50), standardize=False, seed=0,
                          sources=[dict(type="gauss_switch", coding=coding, d=4, timescale=4)]))
        o = ds.sample(5000, 300, np.random.default_rng(1))["obs"].astype(float)
        z = o[:, 150]                                                    # independent draws
        assert np.abs(np.cov(z.T) - np.eye(4)).max() < 0.06
        assert min(stats.kstest(z[:, j], "norm").pvalue for j in range(4)) > 1e-3
        u = ds.sources[0].Q[:, 0]
        inc = np.diff(o @ u, axis=1).ravel()
        assert stats.kurtosis(inc) > 2
