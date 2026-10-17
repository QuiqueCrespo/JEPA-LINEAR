"""Linear identifiability of the true signal latent (EXPLORATORY, post hoc: specified after the confirmatory HPC results;
it decides no hypothesis).

    python experiments/recon_vs_pred/identifiability.py <runs dir>     # load-only, cached test features; compute node

For every (condition, learner, seed) of the confirmatory sweep, window features Z are mapped to two ground-truth targets,
fit on the first half of the (fresh, eval_seed) test series and scored by held-out R^2 on the rest:
    lat_*   the 2-D rotation latent at window offsets 0, W/2, W-1  (same construction as the nuisance probes)
    stat_*  the oracle's lag-covariance features of the true latent in the window (they carry the rotation direction,
            i.e. the regime); targets standardised on the training series, near-constant targets dropped
with a linear map (ridge, alpha 1; *_lin) and a fixed MLP (64, 64), 100 iterations, <= 2000 training windows,
random_state 0 (*_mlp).  High *_lin = linearly identifiable (up to an affine map); *_mlp - *_lin = nonlinear
entanglement.  Negative R^2 is retained.  Rows go to cpbench_results/recon_vs_pred_identifiability.csv."""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(HERE))]
import cpbench  # noqa: F401  (before torch: OpenMP)
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from cpbench import Benchmark
from cpbench.data import Dataset
from cpbench.eval.invariance import nuisance_mlp_r2, nuisance_r2
from design import CONDITIONS, CONTROLS, LEARNERS, dataset
from experiment import FAMILY, control, learner


def window_r2(Z, Y, kind, seed=0, max_train=2000):
    """Held-out R^2 (uniform over standardised targets) of window-level targets Y (R, T, m) from features Z (R, T, d)."""
    h = Z.shape[0] // 2
    Xtr, Xte = Z[:h].reshape(-1, Z.shape[-1]), Z[h:].reshape(-1, Z.shape[-1])
    Ytr, Yte = Y[:h].reshape(-1, Y.shape[-1]), Y[h:].reshape(-1, Y.shape[-1])
    keep = Ytr.std(0) > 1e-6
    Ytr, Yte = Ytr[:, keep], Yte[:, keep]
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    ym, ys = Ytr.mean(0), Ytr.std(0)
    Xtr, Xte, Ytr, Yte = (Xtr - mu) / sd, (Xte - mu) / sd, (Ytr - ym) / ys, (Yte - ym) / ys
    if kind == "lin":
        pred = Ridge(alpha=1.0).fit(Xtr, Ytr).predict(Xte)
    else:
        idx = np.random.default_rng(seed).choice(len(Xtr), min(len(Xtr), max_train), replace=False)
        pred = MLPRegressor(hidden_layer_sizes=(64, 64), max_iter=100, random_state=seed).fit(Xtr[idx], Ytr[idx]).predict(Xte)
    r2 = 1 - ((Yte - pred) ** 2).sum(0) / (((Yte - Yte.mean(0)) ** 2).sum(0) + 1e-12)
    return float(np.mean(r2))


if __name__ == "__main__":
    out = sys.argv[1]
    rows = []
    for c in CONDITIONS:
        bench = Benchmark(Dataset(dataset(c)), out=out, log=lambda *a: None, load_only=True)
        U = bench.test["latent"]
        views, starts = bench.features(control("oracle"))
        stat = views["features"].astype(np.float64)
        cands = [(k, 0, control(k)) for k in CONTROLS] + [(n, s, learner(n, 20000, s)) for n in LEARNERS for s in (0, 1, 2)]
        for key, seed, f in cands:
            v, st = bench.features(f)
            assert np.array_equal(st, starts)
            Z = v["features"].astype(np.float64)
            rows.append(dict(cond=c, learner=key, family=FAMILY[key], seed=seed,
                             lat_lin=nuisance_r2(Z, U, starts, f.window), lat_mlp=nuisance_mlp_r2(Z, U, starts, f.window),
                             stat_lin=window_r2(Z, stat, "lin"), stat_mlp=window_r2(Z, stat, "mlp"),
                             eval_seed=bench.ds.evaluation_seed, analysis="exploratory_post_hoc"))
            r = rows[-1]
            print(f"{c:<3} {key:<14} s{seed}  lat lin {r['lat_lin']:+.3f} mlp {r['lat_mlp']:+.3f}   "
                  f"stat lin {r['stat_lin']:+.3f} mlp {r['stat_mlp']:+.3f}", flush=True)
    os.makedirs("cpbench_results", exist_ok=True)
    pd.DataFrame(rows).to_csv("cpbench_results/recon_vs_pred_identifiability.csv", index=False)
