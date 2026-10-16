"""Plots for the workbench: datasets, detector decisions against the truth, feature spaces, results and training."""
import json, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA

from cpbench.data.windows import labels_to_samples, majority_labels, score_positions

COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#9b59b6", "#c0392b", "#7f8c8d", "#f1c40f", "#16a085"]


def _bar(ax, S, label):
    """A regime path (n,) as a coloured bar."""
    b = np.r_[0, np.flatnonzero(np.diff(S)) + 1, len(S)]
    for a, c in zip(b[:-1], b[1:]):
        ax.axvspan(a, c, color=COLORS[int(S[a]) % len(COLORS)], lw=0)
    ax.set_yticks([]); ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8)


def align(true, pred, K):
    """Relabel predicted regimes to best match the truth (display only)."""
    M = np.zeros((K, max(K, int(pred.max()) + 1)))
    np.add.at(M, (true.ravel(), pred.ravel()), 1)
    r, c = linear_sum_assignment(-M)
    mapping = dict(zip(c, r))
    return np.vectorize(lambda p: mapping.get(p, p))(pred)


def plot_dataset(ds, data, series=0, n=3000, channels=16):
    """Clean regime-driven signal, observed channels and the true regime path of one test series."""
    S = data["S_s"][series, :n]; clean = data["clean"][series, :n]; obs = data["obs"][series, :n]
    fig, axs = plt.subplots(3, 1, figsize=(13, 5.5), sharex=True, gridspec_kw=dict(height_ratios=[1, 2.2, 2.2]))
    _bar(axs[0], S, "regime")
    for j in range(min(3, clean.shape[1])):
        axs[1].plot(clean[:, j], lw=0.7, label=f"clean {j}")
    axs[1].set_ylabel("clean signal", fontsize=8); axs[1].legend(fontsize=7, loc="upper right", ncol=3)
    axs[2].imshow(obs[:, :channels].T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
    axs[2].set_ylabel(f"observed\n(first {min(channels, obs.shape[1])} of {obs.shape[1]} ch.)", fontsize=8)
    for t in np.flatnonzero(data["cp_s"][series, :n]):
        axs[1].axvline(t, color="k", ls=":", lw=0.6)
    axs[-1].set_xlabel("sample"); fig.suptitle(ds.info.id, fontsize=8, x=0.01, ha="left")
    plt.tight_layout(); plt.show()


def plot_decisions(data, decisions, starts, W, K, title="", series=0, n=3000):
    """True regimes, each clustering detector's (aligned) labels, and each change score against the true changes."""
    S = data["S_s"][:, :]; N = S.shape[1]
    clus = {k: v for k, v in decisions.items() if v.get("labels") is not None}
    cps = {k: v for k, v in decisions.items() if v.get("cp_score") is not None}
    fig, axs = plt.subplots(1 + len(clus) + len(cps), 1, figsize=(13, 0.9 + 0.55 * len(clus) + 1.0 * len(cps)), sharex=True,
                            gridspec_kw=dict(height_ratios=[0.6] + [0.6] * len(clus) + [1.4] * len(cps)))
    axs = np.atleast_1d(axs)
    _bar(axs[0], S[series, :n], "truth")
    for ax, (name, d) in zip(axs[1:], clus.items()):
        lab = align(S, labels_to_samples(d["labels"], starts, W, N), K)
        _bar(ax, lab[series, :n], name)
    pos = score_positions(starts, W); keep = pos < n
    true_cp = np.flatnonzero(data["cp_s"][series, :n])
    for ax, (name, d) in zip(axs[1 + len(clus):], cps.items()):
        sc = np.where(np.isfinite(d["cp_score"][series]), d["cp_score"][series], np.nan)
        ax.plot(pos[keep], sc[keep], lw=0.8, color="#333")
        for t in true_cp:
            ax.axvline(t, color="#c0392b", lw=0.6, alpha=0.6)
        ax.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8); ax.set_yticks([])
    axs[-1].set_xlabel("sample (red: true change points)"); fig.suptitle(title, fontsize=9, x=0.01, ha="left")
    plt.tight_layout(); plt.show()


def plot_features(views, data, starts, W, K, titles=None):
    """2-D PCA of window features coloured by the (majority) true regime, one panel per feature set."""
    S = majority_labels(data["S_s"], starts, W, K).ravel()
    fig, axs = plt.subplots(1, len(views), figsize=(4 * len(views), 3.6), squeeze=False)
    for ax, (name, Z) in zip(axs[0], views.items()):
        P = PCA(2).fit_transform(Z.reshape(-1, Z.shape[-1]))
        idx = np.random.default_rng(0).choice(len(P), min(4000, len(P)), replace=False)
        ax.scatter(P[idx, 0], P[idx, 1], c=[COLORS[s % len(COLORS)] for s in S[idx]], s=3, alpha=0.6)
        ax.set_title(name if titles is None else titles[name], fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout(); plt.show()


def plot_results(df, metrics=("acc", "cp_ap"), view="features"):
    """Heatmaps of feature learner x detector for each metric (chance for acc = 1/K)."""
    d = df[(df.view == view) & (df.detector != "probe")]
    fig, axs = plt.subplots(1, len(metrics), figsize=(6.5 * len(metrics), 0.45 * d.features_label.nunique() + 1.6), squeeze=False)
    for ax, m in zip(axs[0], metrics):
        piv = d[d.metric == m].pivot_table(index="features_label", columns="detector_label", values="value")
        if piv.empty:
            ax.set_visible(False); continue
        im = ax.imshow(piv.values, cmap="viridis", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(range(piv.shape[1]), piv.columns, rotation=30, ha="right", fontsize=7)
        ax.set_yticks(range(piv.shape[0]), piv.index, fontsize=7)
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                v = piv.values[i, j]
                if np.isfinite(v): ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7, color="w" if v < 0.6 else "k")
        ax.set_title(m, fontsize=9); plt.colorbar(im, ax=ax, fraction=0.03)
    plt.tight_layout(); plt.show()


def plot_training(run_dir, title=""):
    """Training losses and periodic validation of a learnt feature learner (from its log.jsonl)."""
    path = os.path.join(run_dir, "log.jsonl")
    if not os.path.exists(path):
        print("no training log (fixed feature learner?)"); return
    log = [json.loads(l) for l in open(path)]
    tr = pd.DataFrame([r for r in log if r["kind"] == "train"]); ev = pd.DataFrame([r for r in log if r["kind"] == "eval"])
    fig, axs = plt.subplots(1, 2, figsize=(11, 2.8))
    for c in [c for c in tr.columns if c not in ("step", "kind", "t", "lr")]:
        axs[0].plot(tr.step, tr[c], label=c)
    axs[0].set_yscale("symlog"); axs[0].legend(fontsize=7); axs[0].set_title("training losses", fontsize=9)
    for c in ("probe", "gmm_acc"):
        if c in ev: axs[1].plot(ev.step, ev[c], marker="o", label=c)
    axs[1].legend(fontsize=7); axs[1].set_title("validation (probe = supervised diagnostic)", fontsize=9); axs[1].set_ylim(0, 1)
    fig.suptitle(title, fontsize=9, x=0.01, ha="left"); plt.tight_layout(); plt.show()
