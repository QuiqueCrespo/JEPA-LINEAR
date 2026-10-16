"""Run and score the reconstruction vs contrastive vs prediction experiment; the pre-registered verdicts.

evaluate(conds, steps, seeds, out) fits (or loads) every learner and control and returns one tidy row per
(condition, learner, seed) with the primary metrics:
    probe     linear-probe balanced accuracy of the regime (supervised diagnostic; chance 0.5)
    probe_mlp MLP probe on pairs of consecutive windows (secondary: sees regimes held in the feature covariance)
    nuis_r2   ridge R^2 of the nuisance latent at window offsets 0, W/2, W-1 (cpbench.eval.invariance.nuisance_r2)
    hmm, gmm  unsupervised sample-level clustering accuracy (k=16 = d: no truncation); hmm is primary
    eff_rank  participation ratio of the feature covariance; loss_drop  final / step-1000 training loss
verdicts(df) applies the thresholds fixed in README.md before any run."""
import json, os
import numpy as np
import pandas as pd

from cpbench import Benchmark, make_learner
from cpbench.data import Dataset
from cpbench.eval.invariance import nuisance_r2

from design import CONDITIONS, CONTROLS, COMMON, DETECTORS, LEARNERS, dataset

FAMILY = {**LEARNERS, "untrained": "control", "raw_flat": "control", "raw_lagcov": "control", "oracle": "control"}
PRED, RECON = ("jepa_separate", "jepa_shared"), ("ae", "vae")


def learner(name, steps, seed=0, workers=1):
    """A trained learner with the common settings; the HPC sweep builds the same identity."""
    return make_learner(name, **COMMON, steps=steps, warmup=min(1000, steps), seed=seed, workers=workers,
                        eval_every=2000, ckpt_every=2000, eval_detectors=["gmm", "hmm"])


def control(key):
    name, extra = CONTROLS[key]
    p = dict(window=COMMON["window"], **extra)
    if name.startswith("jepa"):
        p.update({k: v for k, v in COMMON.items() if k != "window"})
    return make_learner(name, **p)


def _eff_rank(Z):
    ev = np.linalg.eigvalsh(np.atleast_2d(np.cov(Z.reshape(-1, Z.shape[-1]).T)))
    return float(ev.sum() ** 2 / (ev ** 2).sum())


def _loss_drop(run_dir):
    p = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(p):
        return np.nan
    tr = [r for r in map(json.loads, open(p)) if r.get("kind") == "train"]
    if len(tr) < 2:
        return np.nan
    first = next((r for r in tr if r["step"] >= 1000), tr[0])
    return tr[-1]["loss"] / first["loss"] if first["loss"] > 0 else np.nan


def score_one(bench, f, key, seed, cond):
    res = bench.run(f)
    res = res[res.view == "features"]
    views, starts = bench.features(f)
    Z = views["features"].astype(np.float64)
    get = lambda det, m: float(res[(res.detector == det) & (res.metric == m)].value.iloc[0])
    return dict(cond=cond, speed=CONDITIONS[cond][0], sigma_n=CONDITIONS[cond][1], learner=key, family=FAMILY[key],
                seed=seed, probe=get("probe", "probe_linear"), probe_mlp=get("probe", "probe_mlp_pair"), nuis_r2=nuisance_r2(Z, bench.test["nuisance"], starts, f.window),
                hmm=get("hmm", "acc"), gmm=get("gmm", "acc"), eff_rank=_eff_rank(Z),
                loss_drop=_loss_drop(bench.run_dir(f)) if f.trainable else np.nan)


def evaluate(conds, steps, seeds, out, learners=tuple(LEARNERS), controls=tuple(CONTROLS), log=print, quick=False):
    rows = []
    for c in conds:
        bench = Benchmark(Dataset(dataset(c, quick)), detectors=DETECTORS, out=out, log=lambda *a: None)
        for key in controls:
            rows.append(score_one(bench, control(key), key, 0, c)); log(_fmt(rows[-1]))
        for name in learners:
            for s in seeds:
                rows.append(score_one(bench, learner(name, steps, s), name, s, c)); log(_fmt(rows[-1]))
    return pd.DataFrame(rows)


def _fmt(r):
    return (f"{r['cond']:<3} {r['learner']:<14} s{r['seed']}  probe {r['probe']:.3f}  nuis_r2 {r['nuis_r2']:.3f}  "
            f"hmm {r['hmm']:.3f}  gmm {r['gmm']:.3f}  eff_rank {r['eff_rank']:.1f}")


# ---------------------------------------------------------------------- pre-registered tests
def _boot_diff(a, b, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return float(np.mean(a) - np.mean(b)), np.percentile(d, [2.5, 97.5])


def verdicts(df):
    """[(test, verdict, numbers)] for the gates and H1-H4 (thresholds from README.md)."""
    g = lambda c, ls, m: df[(df.cond == c) & df.learner.isin(ls)][m].to_numpy()
    out = []
    have = set(df.cond)
    # validity gates
    orc = df[df.learner == "oracle"].set_index("cond").hmm
    out.append(("gate: oracle HMM acc >= 0.95 everywhere", "pass" if (orc >= 0.95).all() else "FAIL",
                ", ".join(f"{c} {v:.3f}" for c, v in orc.items())))
    if "N0" in have and df.learner.isin(LEARNERS).any():
        best = df[(df.cond == "N0") & df.learner.isin(LEARNERS)].groupby("learner").probe.mean()
        out.append(("gate: some trained learner has probe >= 0.9 on N0", "pass" if best.max() >= 0.9 else "FAIL",
                    f"best {best.idxmax()} {best.max():.3f}"))
    tr = df[df.learner.isin(LEARNERS)]
    if tr.empty:
        return pd.DataFrame(out, columns=["test", "verdict", "numbers"])
    bad = tr[(tr.eff_rank <= 2) | (tr.loss_drop >= 1)]
    out.append(("gate: no collapse (eff_rank > 2, final loss < loss at 1k)", "pass" if bad.empty else "FAIL",
                "; ".join(f"{r.cond} {r.learner} s{r.seed} rank {r.eff_rank:.1f} drop {r.loss_drop:.2f}" for r in bad.itertuples()) or "-"))
    for c in sorted(have):
        u = df[(df.cond == c) & (df.learner == "untrained")].probe.mean()
        fam = {f: g(c, ls, "probe").mean() for f, ls in (("pred", PRED), ("recon", RECON))}
        out.append((f"control {c}: family probe - untrained probe ({u:.3f}) >= 0.05", "info",
                    ", ".join(f"{f} {v - u:+.3f}" for f, v in fam.items())))
    if "F2" in have:
        d, ci = _boot_diff(g("F2", PRED, "probe"), g("F2", RECON, "probe"))
        v = "supported" if d >= 0.10 and ci[0] > 0 else "refuted" if d < 0.05 else "inconclusive"
        out.append(("H1: F2 probe(pred) - probe(recon) >= 0.10, CI > 0", v, f"diff {d:+.3f}, 95% CI [{ci[0]:+.3f}, {ci[1]:+.3f}]"))
    if {"F2", "S2"} <= have:
        pf, ps = g("F2", PRED, "nuis_r2").mean(), g("S2", PRED, "nuis_r2").mean()
        rf, rs = g("F2", RECON, "nuis_r2").mean(), g("S2", RECON, "nuis_r2").mean()
        v = ("supported" if pf <= 0.3 and ps >= 0.6 and rf >= 0.6 and rs >= 0.6 else
             "refuted" if pf >= rf - 0.1 else "inconclusive")
        out.append(("H2: nuisance R2 pred F2 <= 0.3, S2 >= 0.6; recon >= 0.6 on both", v,
                    f"pred F2 {pf:.3f} S2 {ps:.3f}; recon F2 {rf:.3f} S2 {rs:.3f}"))
        hf, hs = g("F2", PRED, "hmm").mean(), g("S2", PRED, "hmm").mean()
        v = "supported" if hf >= 0.80 and hf - hs >= 0.15 else "refuted" if hf < 0.65 else "inconclusive"
        out.append(("H3: HMM acc pred F2 >= 0.80 and >= S2 + 0.15", v, f"pred F2 {hf:.3f}, S2 {hs:.3f}"))
    if "F2" in have:
        t, c_ = g("F2", ("ts2vec",), "nuis_r2").mean(), g("F2", ("cpc",), "nuis_r2").mean()
        out.append(("H4 (exploratory): F2 nuisance R2 ts2vec - cpc >= 0.3", "consistent" if t - c_ >= 0.3 else "not consistent",
                    f"ts2vec {t:.3f}, cpc {c_:.3f}, jepa {g('F2', PRED, 'nuis_r2').mean():.3f}, ae/vae {g('F2', RECON, 'nuis_r2').mean():.3f}"))
    return pd.DataFrame(out, columns=["test", "verdict", "numbers"])
