import numpy as np
from cpbench.eval.metrics import clustering, cp_metrics


def test_clustering_perfect_and_permuted():
    S = np.random.default_rng(0).integers(0, 4, (3, 100))
    assert clustering(S, S, 4)["acc"] == 1.0
    assert clustering(S, (S + 1) % 4, 4)["acc"] == 1.0                    # label permutation does not matter
    assert clustering(S, np.zeros_like(S), 4)["acc"] <= 0.26


def test_cp_metrics_sample_resolution():
    W, T = 32, 100; n = W * T
    positions = np.arange(T) * W.__float__()                              # score t located at sample 32 t
    cp_s = np.zeros((2, n), bool); cp_s[0, [640, 1600]] = True; cp_s[1, 2245] = True
    score = np.zeros((2, T)); score[0, [20, 50]] = 1; score[1, 70] = 1     # 2245 is 5 samples from 2240
    m = cp_metrics(score, positions, cp_s, margin=32)
    assert m["ap"] == 1.0 and m["f1_best"] == 1.0
    assert cp_metrics(np.roll(score, 3, axis=1), positions, cp_s, margin=32)["f1_best"] == 0.0
    assert cp_metrics(score, positions, cp_s, margin=2)["f1_best"] < 1.0   # 2245 vs 2240 misses a 2-sample margin
