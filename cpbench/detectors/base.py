"""Detectors: change-point and regime-clustering algorithms that run on top of any feature sequence.

    detect(Z, K, seed=0) -> dict(labels=(R, T) or None, cp_score=(R, T) or None)

Z (R, T, F) is a window-level feature sequence produced by a feature learner (cpbench/features).  cp_score[t] scores a
change between windows t-1 and t (higher = more likely).  `kind` is "cluster" (labels + a switch score) or "cp"
(change score only).  Detectors are fitted transductively on the series they are given and receive the number of
regimes K.  New detectors: subclass Detector, declare `defaults`, implement detect(), @register_detector(name)."""
import numpy as np
from sklearn.decomposition import PCA

from cpbench.core import Component, Registry

DETECTORS = Registry("detector")
register_detector = DETECTORS.register


def make_detector(spec, **params):
    return DETECTORS.make(spec, **params)


class Detector(Component):
    kind = "cluster"

    def detect(self, Z, K, seed=0):
        raise NotImplementedError


def reduce(F, k=32, seed=0, scaling="whiten"):
    """Centre and PCA to <= k dims.  scaling: "whiten" (unit variance per component), "keep" (component variances kept,
    one global scale) or "zscore" (features standardised first, component variances kept).  Returns (N, k')."""
    F = F - F.mean(0)
    if scaling == "zscore":
        F = F / (F.std(0) + 1e-9)
    k = min(k, F.shape[1], len(F) - 1)
    P = PCA(k, random_state=seed, svd_solver="randomized" if F.shape[1] > 4 * k else "full").fit_transform(F)
    if scaling == "whiten":
        return P / (P.std(0) + 1e-9)
    return P / np.sqrt((P ** 2).sum(1).mean() / k + 1e-12)


def persistence(labels, R):
    """1 - (rate of label changes between consecutive windows) / (rate expected for shuffled labels of the same
    cluster sizes): 1 for perfectly persistent labels, ~0 for labels unrelated to time."""
    l = labels.reshape(R, -1)
    p = np.bincount(labels.ravel()) / labels.size
    return 1 - (l[:, 1:] != l[:, :-1]).mean() / max(1 - (p ** 2).sum(), 1e-9)


def flat(Z):
    return Z.reshape(-1, Z.shape[-1])
