"""Each change-type recipe of the standard grid keeps the statistics it claims to match equal across regimes (on the
clean signal), while the statistic it changes differs."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from cpbench.data import Dataset, expand
from cpbench.data.windows import to_windows, majority_labels

GRID = "cpbench/configs/grid_synthetic.yaml"
TOL = dict(mean=0.08, var=0.10, acf1=0.05, corr0=0.06)
# recipe order in the grid file -> (name, statistics matched across regimes, statistic that changes)
RECIPES = [("mean", ("var", "acf1", "corr0"), "mean"), ("trend", (), "mean"), ("variance", ("mean", "acf1", "corr0"), "var"),
           ("meanvar", ("acf1", "corr0"), "var"), ("autocorr", ("mean", "var", "corr0"), "acf1"),
           ("spectral", ("mean", "var", "corr0"), "acf1"), ("corr", ("mean", "var", "acf1"), "corr0"),
           ("leadlag", ("mean", "var", "acf1", "corr0"), None), ("skew", ("mean", "var", "acf1", "corr0"), None),
           ("phase", ("mean", "var", "corr0"), None), ("transition", ("mean", "var", "acf1", "corr0"), None)]


def signals():
    cfgs = [c for c in expand(GRID) if len(c["sources"]) == 1 and not c["transforms"]]
    assert len(cfgs) == len(RECIPES)
    return {name: dict(c, standardize=False) for (name, _, _), c in zip(RECIPES, cfgs)}


def regime_stats(x, S_s, K):
    """Per-regime statistics of a sample-level signal x (R, n, d)."""
    out = {k: [] for k in ("mean", "var", "acf1", "corr0")}
    same = S_s[:, 1:] == S_s[:, :-1]
    for k in range(K):
        xs = x[S_s == k]
        out["mean"].append(xs.mean(0)); out["var"].append(xs.var(0))
        mk = same & (S_s[:, 1:] == k)
        a, b = x[:, :-1][mk], x[:, 1:][mk]
        out["acf1"].append(((a - a.mean(0)) * (b - b.mean(0))).mean(0) / (a.std(0) * b.std(0)))
        c = np.corrcoef(xs.T); out["corr0"].append(c[np.triu_indices_from(c, 1)])
    return {k: np.array(v) for k, v in out.items()}


@pytest.fixture(scope="module")
def samples():
    res = {}
    for name, cfg in signals().items():
        ds = Dataset(cfg)
        d = ds.sample(32, 400 * 32, np.random.default_rng(5))
        res[name] = (ds, regime_stats(d["obs"].astype(np.float64), d["S_s"], ds.K), d)
    return res


@pytest.mark.parametrize("name,matched,changed", RECIPES)
def test_matched_and_changed_statistics(samples, name, matched, changed):
    ds, st, _ = samples[name]
    for inv in matched:
        spread = np.abs(st[inv] - st[inv].mean(0)).max()
        assert spread < TOL[inv], f"{name}: {inv} differs across regimes by {spread:.3f}"
    if changed:
        assert np.abs(st[changed] - st[changed].mean(0)).max() > 3 * TOL[changed], f"{name}: {changed} does not change"


def test_transition_has_no_single_window_information(samples):
    ds, _, d = samples["transition"]
    c, starts = to_windows(d["obs"], 32, 32)
    feats = np.concatenate([c.mean(2), c.std(2), (c[:, :, 1:] * c[:, :, :-1]).mean(2)], -1).reshape(-1, c.shape[-1] * 3)
    y = majority_labels(d["S_s"], starts, 32, ds.K).ravel(); h = len(y) // 2
    assert LogisticRegression(max_iter=2000).fit(feats[:h], y[:h]).score(feats[h:], y[h:]) < 0.35


def test_switches_not_aligned_to_any_grid(samples):
    cp = samples["mean"][2]["cp_s"]
    assert len(np.unique(np.flatnonzero(cp.ravel()) % cp.shape[1] % 32)) > 20
