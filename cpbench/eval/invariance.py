"""Sensitivity of a representation to controlled interventions, from counterfactual pairs.

Two samples drawn with the same rng seed share every regime path, innovation and noise draw; holding a regime process
at different values (Dataset.sample(fix=...)) changes only what that process controls.  E.g. with a "sensor" process
driving noise / calibration transforms and the target process driving the dynamics:

    interventions = {"sensor": ({"sensor": 0}, {"sensor": 1}),                     # same trajectory, other sensor
                     "dynamics": ({"main": 0, "sensor": 0}, {"main": 1, "sensor": 0})}   # same noise, other mechanism

    sensitivity(learner, ds, interventions)  -> {name: score}
    score = mean ||z_a(w) - z_b(w)|| over paired windows w  /  mean ||z(u) - z(u')|| over random unrelated windows u, u'
            drawn from both conditions pooled

0: the representation ignores the intervention; ~1: paired windows are as far apart as unrelated windows.  The ratio
is independent of the overall feature scale; it is measured in the representation's own geometry (no whitening).
A representation suited to regime discovery should score high on dynamics and low on sensor interventions.
"""
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from cpbench.data.windows import majority_labels


def paired_distance(learner, ds, fix_a, fix_b, R=8, n=4000, seed=0):
    rng_seed = [ds.seed, 7700 + seed]
    za = learner.represent(ds.sample(R, n, np.random.default_rng(rng_seed), fix=fix_a))[0]["features"]
    zb = learner.represent(ds.sample(R, n, np.random.default_rng(rng_seed), fix=fix_b))[0]["features"]
    za, zb = za.reshape(-1, za.shape[-1]).astype(np.float64), zb.reshape(-1, zb.shape[-1]).astype(np.float64)
    pool = np.concatenate([za, zb]); rng = np.random.default_rng(seed)
    i, j = rng.integers(len(pool), size=(2, len(za)))
    return float(np.linalg.norm(za - zb, axis=1).mean() / (np.linalg.norm(pool[i] - pool[j], axis=1).mean() + 1e-12))


def sensitivity(learner, ds, interventions, R=8, n=4000, seed=0):
    """{name: paired distance ratio} for interventions {name: (fix_a, fix_b)}."""
    return {k: paired_distance(learner, ds, a, b, R, n, seed) for k, (a, b) in interventions.items()}


def decodability(Z, S_s, starts, W, K, seed=0):
    """Linear-probe accuracy of window labels (majority of S_s per window) from features Z (R, T, d), trained on the
    first half of the series and tested on the rest: how much of a factor (e.g. the sensor condition) the
    representation keeps in linearly readable form.  Diagnostic only (uses labels)."""
    y = majority_labels(S_s, starts, W, K)
    h = max(1, Z.shape[0] // 2)
    Xtr, Xte = Z[:h].reshape(-1, Z.shape[-1]), Z[h:].reshape(-1, Z.shape[-1])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    clf = LogisticRegression(max_iter=2000, C=0.1, random_state=seed).fit((Xtr - mu) / sd, y[:h].ravel())
    return float(clf.score((Xte - mu) / sd, y[h:].ravel()))


def nuisance_r2(Z, U, starts, W, offsets=None, alpha=1.0):
    """Linear (ridge) R^2 of a continuous latent U (R, n, m), e.g. an exposed nuisance, from window features Z (R, T, d):
    targets are U at window offsets (default 0, W // 2, W - 1), trained on the first half of the series and tested on
    the rest, R^2 averaged over targets.  0: the features do not linearly carry U; 1: they carry it fully."""
    offsets = (0, W // 2, W - 1) if offsets is None else offsets
    Y = np.concatenate([U[:, starts + o] for o in offsets], -1).astype(np.float64)      # (R, T, m * len(offsets))
    h = max(1, Z.shape[0] // 2)
    Xtr, Xte = Z[:h].reshape(-1, Z.shape[-1]).astype(np.float64), Z[h:].reshape(-1, Z.shape[-1]).astype(np.float64)
    Ytr, Yte = Y[:h].reshape(-1, Y.shape[-1]), Y[h:].reshape(-1, Y.shape[-1])
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    pred = Ridge(alpha=alpha).fit((Xtr - mu) / sd, Ytr).predict((Xte - mu) / sd)
    r2 = 1 - ((Yte - pred) ** 2).sum(0) / (((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12)
    return float(np.mean(r2))
