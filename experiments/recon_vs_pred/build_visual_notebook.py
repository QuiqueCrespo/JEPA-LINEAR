"""Write experiment_recon_vs_pred_visual.ipynb: a visual tour of the confirmatory HPC run (display only).

    python experiments/recon_vs_pred/build_visual_notebook.py
    CPBENCH_LAUNCH=<launch dir> jupyter nbconvert --to notebook --execute experiment_recon_vs_pred_visual.ipynb \
        --output experiment_recon_vs_pred_visual_run.ipynb          # on a compute node (t-SNE, HMM decoding)
"""
import os
import nbformat as nbf

cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip()))

md("""
# Reconstruction vs contrastive vs prediction: visual tour of the confirmatory run

Display only. It reads the frozen HPC launch: cached test-set features, run logs and the committed result CSVs.
Nothing is retrained, and nothing here changes a verdict. The identifiability panels are **exploratory and post hoc**.

1. **The data**: regime, rotation latent, nuisance (fast vs slow), observations, and the context/gap/target geometry
2. **Results overview**: verdicts, every metric as a heatmap, paired effects with CIs, the linear-vs-nonlinear geometry
3. **Latent spaces**: PCA, t-SNE and the linear-probe axis for every learner and condition, coloured by regime and by nuisance
4. **Time domain**: feature trajectories and the HMM's decoded regime against the truth
5. **Training dynamics**: loss, effective rank and validation clustering over 20k steps
6. **CPC trainability check**
""")

code(r'''
import os, sys, json, glob
L = os.environ.get("CPBENCH_LAUNCH", "sweeps/recon_vs_pred/launch_20261016T200516Z")
os.environ["CPBENCH_CACHE"] = os.path.abspath(os.path.join(L, "cache"))      # before importing cpbench
RUNS = os.path.abspath(os.path.join(L, "runs"))
sys.path[:0] = ["experiments/recon_vs_pred", "."]
import cpbench  # noqa: F401
import numpy as np, pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
from cpbench import Benchmark
from cpbench.data import Dataset
from cpbench.data.windows import majority_labels
from cpbench.detectors import make_detector
from design import CONDITIONS, CONTROLS, LEARNERS, W, GAP, TAU, EVAL_SEED, dataset
from experiment import learner, control, FAMILY, PRED, RECON, _boot_diff

# palette: one hue per family (validated first three slots of the reference palette), gray controls,
# marker shape separates the two learners of a family; regime = blue / red poles; signed values = diverging.
BLUE, ORANGE, AQUA, RED, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#e34948", "#8a8984"
INK, INK2, MUTED, RULE, MID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#f0efec"
FAMC = {"reconstruction": BLUE, "prediction": ORANGE, "contrastive (future target)": AQUA,
        "contrastive (same-time target)": AQUA, "control": GRAY}
MARK = dict(ae="o", vae="s", jepa_separate="o", jepa_shared="s", cpc="o", ts2vec="s",
            untrained="v", raw_flat="D", raw_lagcov="P", oracle="*")
REG = [BLUE, RED]
DIV = LinearSegmentedColormap.from_list("div", [BLUE, MID, RED])
SEQ = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"])
ORDER = ["ae", "vae", "jepa_separate", "jepa_shared", "cpc", "ts2vec", "untrained", "raw_flat", "raw_lagcov", "oracle"]
TRAINED = ORDER[:6]
CONDS = list(CONDITIONS)
plt.rcParams.update({"figure.dpi": 80, "savefig.dpi": 80, "font.size": 8.5, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "axes.titlesize": 9,
                     "axes.grid": False, "legend.frameon": False, "image.interpolation": "nearest"})
def learner_handles(keys=ORDER):
    return [Line2D([], [], ls="", marker=MARK[k], color=FAMC[FAMILY[k]], mec="white", mew=0.6, ms=7, label=k) for k in keys]

benches = {c: Benchmark(Dataset(dataset(c)), out=RUNS, log=lambda *a: None, load_only=True) for c in CONDS}
def spec(key, seed=0):
    return control(key) if key in CONTROLS else learner(key, 20000, seed)
def feats(c, key, seed=0):
    v, starts = benches[c].features(spec(key, seed))
    return v["features"].astype(np.float64), starts
_, STARTS = feats("N0", "raw_flat")
def win_regime(c):  return majority_labels(benches[c].test["S_s"], STARTS, W, 2)          # (R, T)
def win_nuis(c):    return benches[c].test["nuisance"][:, STARTS + W // 2, 0] * CONDITIONS[c][1]

res = pd.read_csv("cpbench_results/recon_vs_pred_hpc.csv")
ident = pd.read_csv("cpbench_results/recon_vs_pred_identifiability.csv")
df = res.merge(ident[["cond", "learner", "seed", "lat_lin", "lat_mlp", "stat_lin", "stat_mlp"]], on=["cond", "learner", "seed"])
assert df.protocol.nunique() == 1 and (df.eval_seed == EVAL_SEED).all()
print(L, "|", len(df), "rows |", df.protocol.iloc[0], "| eval_seed", EVAL_SEED)
''')

# ------------------------------------------------------------------ 1. data
md("""
## 1. The data

`obs = render(rotation latent) + sigma_n · render(nuisance AR(1)) + noise`. The regime is the **direction** of a damped
2-D rotation (ω = ±0.12 rad/sample, period about 52 samples). Only the nuisance changes between conditions.
Below is the first 1500 samples of test series 0 in every condition. Shading marks regime 1.
""")
code(r'''
seg = slice(0, 1500)
fig, ax = plt.subplots(3, 5, figsize=(17, 7.2), sharex=True, gridspec_kw=dict(height_ratios=[1, 1, 1.7], hspace=0.12, wspace=0.12))
for j, c in enumerate(CONDS):
    t = benches[c].test; S = t["S_s"][0, seg]; x = np.arange(len(S))
    for i in range(2):
        ax[i, j].fill_between(x, 0, 1, where=S == 1, transform=ax[i, j].get_xaxis_transform(), color=RED, alpha=0.14, lw=0)
    for k, col in enumerate((INK, MUTED)):
        ax[0, j].plot(x, t["latent"][0, seg, k], color=col, lw=0.6)
        ax[1, j].plot(x, CONDITIONS[c][1] * t["nuisance"][0, seg, k], color=col, lw=0.6)
    ax[1, j].set_ylim(-7, 7)
    ax[2, j].imshow(t["obs"][0, seg].T, aspect="auto", cmap=DIV, vmin=-3, vmax=3)
    sp, sn = CONDITIONS[c]
    ax[0, j].set_title(f"{c}: {'no nuisance' if sn == 0 else f'{sp} nuisance (tau={TAU[sp]}), sigma_n={sn:g}'}")
    ax[2, j].set_xlabel("sample")
    if j:
        for i in range(3): ax[i, j].set_yticklabels([])
ax[0, 0].set_ylabel("rotation latent\n(regime shaded)"); ax[1, 0].set_ylabel("nuisance x sigma_n"); ax[2, 0].set_ylabel("64 observed channels")
plt.show()
''')
md("""
**The regime is the rotation direction.** Each regime traces the same circle, in opposite directions. A single
window's mean or variance can't tell them apart. It needs a phase-ordered (lag-covariance) statistic, which is why
raw-window linear probes sit at chance.
""")
code(r'''
t = benches["N0"].test; S = t["S_s"][0]; Z = t["latent"][0]
ma = lambda x, k=5: np.stack([np.convolve(x[:, q], np.ones(k) / k, mode="valid") for q in range(2)], 1)
fig, ax = plt.subplots(1, 3, figsize=(14, 4), gridspec_kw=dict(width_ratios=[1, 1, 1.4]))
for k in (0, 1):
    i0 = next(i for i in range(0, len(S) - 300) if (S[i:i + 300] == k).all())
    z = ma(Z[i0:i0 + 60])
    turn = np.mean(z[:-1, 0] * z[1:, 1] - z[:-1, 1] * z[1:, 0])
    ax[k].plot(Z[S == k][::3, 0], Z[S == k][::3, 1], ".", color=REG[k], ms=1.5, alpha=0.15)
    ax[k].plot(z[:, 0], z[:, 1], color=INK, lw=1.2)
    for q in range(0, len(z) - 4, 5):
        ax[k].annotate("", z[q + 3], z[q], arrowprops=dict(arrowstyle="->", color=INK, lw=1))
    ax[k].set_title(f"regime {k}: {'counter-clockwise' if turn > 0 else 'clockwise'}\n(one period, 5-sample moving average)")
    ax[k].set_aspect("equal"); ax[k].set_xlabel("latent 1")
ax[0].set_ylabel("latent 2")
L2 = t["latent"]; Sa = t["S_s"]
phi = np.arctan2(L2[..., 1], L2[..., 0]); dphi = np.angle(np.exp(1j * (phi[:, 1:] - phi[:, :-1])))
ok = (np.linalg.norm(L2[:, :-1], axis=-1) > 1) & (Sa[:, 1:] == Sa[:, :-1])
bins = np.linspace(-1.2, 1.2, 81)
for k in (0, 1):
    ax[2].hist(dphi[ok & (Sa[:, 1:] == k)], bins=bins, color=REG[k], alpha=0.55, lw=0, density=True, label=f"regime {k}")
for w in (-0.12, 0.12): ax[2].axvline(w, ls="--", color=INK2, lw=1)
ax[2].set_xlabel("phase advance per sample (rad); dashed = ±0.12"); ax[2].set_yticks([]); ax[2].legend(fontsize=7.5)
ax[2].set_title("per-sample rotation (|latent| > 1): the regime is the sign")
plt.tight_layout(); plt.show()
''')
md("""
**The timing manipulation.** Context = samples [0, 64), target = [96, 160). The nearest context and target samples
are 33 apart. Fast nuisance (tau = 16) decorrelates across the gap, while slow nuisance (tau = 2000) carries straight
through. So a predictor can use the slow nuisance and can barely use the fast one.
""")
code(r'''
fig, ax = plt.subplots(1, 3, figsize=(15, 3.2), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
for a, c in zip(ax[:2], ("F2", "S2")):
    u = 2 * benches[c].test["nuisance"][0, :400, 0]
    for lo, hi, lab in ((0, W, "context"), (W + GAP, 2 * W + GAP, "target")):
        a.axvspan(lo, hi, color=BLUE if lab == "context" else ORANGE, alpha=0.12, lw=0)
        a.text((lo + hi) / 2, 1.02, lab, transform=a.get_xaxis_transform(), ha="center", color=INK2)
    a.plot(u, color=INK, lw=0.8); a.set_title(f"{c} nuisance (coordinate 1)", pad=14); a.set_xlabel("sample")
lag = np.arange(0, 301)
for sp, ls in (("fast", "-"), ("slow", "--")):
    ax[2].plot(lag, np.exp(-lag / TAU[sp]), ls, color=INK, lw=1.4, label=f"{sp} nuisance (tau={TAU[sp]})")
lat = benches["N0"].test["latent"][:, :, 0]; lat = (lat - lat.mean()) / lat.std()
ac = [np.mean(lat[:, :lat.shape[1] - l] * lat[:, l:]) for l in lag]
ax[2].plot(lag, ac, color=MUTED, lw=1, label="rotation latent (empirical)")
for x, yl, lab in ((33, 0.80, "nearest context-target (33)"), (W, 0.62, "window W (64)"), (W + GAP, 0.44, "start-to-start (96)")):
    ax[2].axvline(x, color=RULE, lw=1.5, zorder=0); ax[2].text(x + 4, yl, lab, color=INK2, fontsize=7, backgroundcolor="white")
ax[2].set_xlabel("lag (samples)"); ax[2].set_ylabel("autocorrelation"); ax[2].legend(loc="lower right", fontsize=7)
plt.tight_layout(); plt.show()
''')

# ------------------------------------------------------------------ 2. results overview
md("""
## 2. Results overview

All gates passed, so these verdicts are confirmatory. Rows marked `secondary`, `exploratory`, `info` or `reported`
decide nothing.
""")
code(r'''
v = pd.read_csv(os.path.join(L, "verdicts.csv"))
with pd.option_context("display.max_colwidth", 140, "display.width", 250):
    display(v[~v.test.str.startswith("control")])
''')
md("""
**Every metric, learner × condition** (mean over 3 seeds for trained learners). Probes and clustering use a blue
scale from 0.5 (chance) to 1. R² uses a diverging scale centred on 0: blue is below the test-mean baseline, red above.
`lat_*` and `stat_*` are the exploratory identifiability scores (true rotation latent, and the regime-carrying
lag-covariance statistic).
""")
code(r'''
M = [("probe_mlp_single", "regime: MLP probe (H1)", "p"), ("probe", "regime: linear probe", "p"),
     ("hmm", "HMM accuracy (H3)", "p"), ("gmm", "GMM accuracy", "p"), ("eff_rank", "effective rank", "r"),
     ("nuis_mlp_r2", "nuisance: MLP R² (H2)", "d"), ("nuis_r2", "nuisance: ridge R²", "d"),
     ("lat_lin", "latent: linear R²", "d"), ("lat_mlp", "latent: MLP R²", "d"),
     ("stat_lin", "regime stat: linear R²", "d"), ("stat_mlp", "regime stat: MLP R²", "d")]
mean = df.groupby(["learner", "cond"]).mean(numeric_only=True)
fig, axes = plt.subplots(2, 6, figsize=(19, 7.5), gridspec_kw=dict(wspace=0.08, hspace=0.3))
for n, (a, (m, title, kind)) in enumerate(zip(axes.ravel(), M)):
    T = mean[m].unstack()[CONDS].reindex(ORDER)
    cmap, lo, hi = (SEQ, 0.5, 1.0) if kind == "p" else (DIV, -1, 1) if kind == "d" else (SEQ, 0, 16)
    a.imshow(T.values, cmap=cmap, vmin=lo, vmax=hi, aspect="auto")
    for (i, j), val in np.ndenumerate(T.values):
        frac = (np.clip(val, lo, hi) - lo) / (hi - lo)
        dark = frac > 0.6 if kind != "d" else abs(val) > 0.6
        a.text(j, i, f"{val:.2f}" if kind != "r" else f"{val:.1f}", ha="center", va="center", fontsize=6.5,
               color="white" if dark else INK)
    a.set_xticks(range(5), CONDS); a.set_title(title)
    a.set_yticks(range(len(ORDER)), ORDER if n % 6 == 0 else [])
    for s in a.spines.values(): s.set_visible(False)
axes.ravel()[-1].axis("off")
plt.show()
''')
md("""
**Paired family effects (prediction − reconstruction), every condition.** Gray dots are the three per-seed paired
differences. Black is the mean with its 95% bootstrap CI. The dashed lines are the pre-registered bands: < 0.05
refutes and ≥ 0.10 with CI > 0 supports. Only F2 was pre-registered.
""")
code(r'''
def fam_seed(c, ls, m):
    return df[(df.cond == c) & df.learner.isin(ls)].groupby("seed")[m].mean().reindex([0, 1, 2]).to_numpy()
panels = [("probe_mlp_single", "H1: single-window MLP probe"), ("probe", "secondary: linear probe"),
          ("probe_mlp_pair", "secondary: pair MLP probe"), ("hmm", "H3b: HMM accuracy"), ("nuis_mlp_r2", "nuisance MLP R² (H2 metric)")]
fig, axes = plt.subplots(1, 5, figsize=(18, 3.2), sharey=True)
for a, (m, title) in zip(axes, panels):
    for i, c in enumerate(CONDS):
        p, r = fam_seed(c, PRED, m), fam_seed(c, RECON, m)
        d, ci = _boot_diff(p, r)
        a.plot(p - r, [i] * 3, "o", color=MUTED, ms=4, alpha=0.8)
        a.errorbar(d, i, xerr=[[d - ci[0]], [ci[1] - d]], fmt="o", color=INK, ms=6, capsize=3, lw=1.5)
    if m != "nuis_mlp_r2":
        for x in (0.05, 0.10):
            a.axvline(x, ls="--", color=MUTED, lw=1)
    a.axvline(0, color=RULE, lw=1.5, zorder=0)
    a.set_title(title); a.set_xlabel("pred − recon")
axes[0].set_yticks(range(5), [c + (" (pre-registered)" if c == "F2" else "") for c in CONDS]); axes[0].invert_yaxis()
plt.tight_layout(); plt.show()
''')
md("""
**Geometry: is the information *linearly* accessible?** Each point is one learner in one condition (seed mean).
Points on the diagonal are as readable linearly as nonlinearly. Points far below it hold the information in a
nonlinear form.

- **Left (regime):** the JEPAs sit on the diagonal, while AE/VAE sit at linear ≈ 0.55 against MLP ≈ 0.95.
- **Middle (instantaneous latent):** the pattern reverses.
- **Right (regime statistic):** the JEPAs are linear, AE/VAE nonlinear.
""")
code(r'''
pairs = [("probe", "probe_mlp_single", "regime probe", (0.45, 1.0)),
         ("lat_lin", "lat_mlp", "rotation latent R² (exploratory)", (-0.7, 1.0)),
         ("stat_lin", "stat_mlp", "regime statistic R² (exploratory)", (-1.0, 1.05))]
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
for a, (x, y, title, lim) in zip(axes, pairs):
    a.plot(lim, lim, color=RULE, lw=1.5, zorder=0)
    for (key, c), r in mean.iterrows():
        if key in ("raw_lagcov", "oracle") and x != "probe":
            continue
        a.scatter(np.clip(r[x], *lim), np.clip(r[y], *lim), marker=MARK[key], color=FAMC[FAMILY[key]], s=46,
                  edgecolor="white", lw=0.6, zorder=3)
    a.set_xlim(lim); a.set_ylim(lim); a.set_xlabel("linear"); a.set_ylabel("MLP"); a.set_title(title)
axes[0].legend(handles=learner_handles(), fontsize=7, loc="lower right", ncol=2)
plt.tight_layout(); plt.show()
''')
md("""
**Mechanism (exploratory association).** Nuisance accessibility against HMM accuracy, one point per run. On F the
JEPAs sit to the left (nuisance dropped), yet AE/VAE cluster just as well with the nuisance kept. On S everyone
fails to cluster.
""")
code(r'''
fig, axes = plt.subplots(1, 5, figsize=(18, 3.4), sharey=True, sharex=True)
for a, c in zip(axes, CONDS):
    for r in df[(df.cond == c) & df.learner.isin(ORDER)].itertuples():
        a.scatter(r.nuis_mlp_r2, r.hmm, marker=MARK[r.learner], color=FAMC[r.family], s=40, edgecolor="white", lw=0.6)
    a.axhline(0.5, color=RULE, lw=1.5, zorder=0); a.set_title(c); a.set_xlabel("nuisance MLP R²")
axes[0].set_ylabel("HMM accuracy"); axes[-1].legend(handles=learner_handles(), fontsize=6.5, loc="center left", bbox_to_anchor=(1, 0.5))
plt.tight_layout(); plt.show()
''')
md("""
**Dose response.** Each metric against nuisance amplitude σ_n. Solid is fast nuisance and dashed is slow; σ_n = 0
is N0, shared by both. Thin lines are seeds, thick lines their mean.
""")
code(r'''
rows = [("hmm", "HMM acc"), ("probe_mlp_single", "MLP probe"), ("probe", "linear probe"), ("nuis_mlp_r2", "nuisance MLP R²")]
fig, axes = plt.subplots(len(rows), 6, figsize=(17, 9), sharex=True, sharey="row")
for j, l in enumerate(TRAINED):
    col = FAMC[FAMILY[l]]
    for i, (m, lab) in enumerate(rows):
        a = axes[i, j]
        for sp, ls in (("fast", "-"), ("slow", "--")):
            sub = df[(df.learner == l) & ((df.speed == sp) | (df.cond == "N0"))]
            for s, g in sub.groupby("seed"):
                g = g.sort_values("sigma_n"); a.plot(g.sigma_n, g[m], ls, color=col, lw=0.6, alpha=0.4)
            g = sub.groupby("sigma_n")[m].mean(); a.plot(g.index, g.values, ls, color=col, lw=2, marker=MARK[l], ms=5, label=sp)
        if j == 0: a.set_ylabel(lab)
        if i == 0: a.set_title(l)
        if i == len(rows) - 1: a.set_xlabel("sigma_n")
axes[0, 0].legend(fontsize=7)
plt.tight_layout(); plt.show()
''')

# ------------------------------------------------------------------ 3. latent spaces
md("""
## 3. Latent spaces

One figure per condition, with seed 0 and all 2000 test windows (8 fresh series × 250 non-overlapping windows).
Features are standardised before projection. Columns:

1. PCA coloured by regime
2. PCA coloured by the nuisance (σ_n × nuisance coordinate 1 at the window centre)
3. t-SNE coloured by regime
4. t-SNE coloured by nuisance
5. The **linear-probe axis**: a logistic regression fitted on the first 4 series and projected onto the held-out 4.
   Separated histograms mean the regime is linearly readable.

Inputs above 50 dimensions are reduced to 50 PCs before t-SNE (perplexity 30, PCA init, random_state 0).
""")
code(r'''
def embed(Z):
    Zs = StandardScaler().fit_transform(Z)
    k = min(2, Zs.shape[1]); P = PCA(k).fit_transform(Zs)
    if k == 1: P = np.c_[P, np.zeros(len(P))]
    X = PCA(50).fit_transform(Zs) if Zs.shape[1] > 50 else Zs
    E = TSNE(2, perplexity=30, init="pca", random_state=0).fit_transform(X)
    return P, E

def latent_figure(c):
    y, u = win_regime(c), win_nuis(c)
    R = y.shape[0]; h = R // 2
    lim = np.percentile(np.abs(u), 98) or 1
    fig, ax = plt.subplots(len(ORDER), 5, figsize=(14, 2.3 * len(ORDER)), gridspec_kw=dict(hspace=0.35, wspace=0.18))
    for i, key in enumerate(ORDER):
        Z, _ = feats(c, key)
        P, E = embed(Z.reshape(-1, Z.shape[-1]))
        yf, uf = y.ravel(), u.ravel()
        for j, (XY, colour) in enumerate(((P, "reg"), (P, "nuis"), (E, "reg"), (E, "nuis"))):
            a = ax[i, j]
            if colour == "reg":
                for k in (0, 1):
                    a.scatter(*XY[yf == k].T, s=2, color=REG[k], alpha=0.5, lw=0, rasterized=True)
            else:
                a.scatter(*XY.T, s=2, c=uf, cmap=DIV, vmin=-lim, vmax=lim, alpha=0.7, lw=0, rasterized=True)
            a.set_xticks([]); a.set_yticks([])
        sc = StandardScaler().fit(Z[:h].reshape(-1, Z.shape[-1]))
        clf = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(Z[:h].reshape(-1, Z.shape[-1])), y[:h].ravel())
        Xte = sc.transform(Z[h:].reshape(-1, Z.shape[-1])); s = clf.decision_function(Xte); yt = y[h:].ravel()
        bins = np.linspace(*np.percentile(s, [0.5, 99.5]), 40)
        for k in (0, 1):
            ax[i, 4].hist(s[yt == k], bins=bins, color=REG[k], alpha=0.55, lw=0)
        ax[i, 4].set_yticks([]); ax[i, 4].set_title(f"bal. acc {balanced_accuracy_score(yt, clf.predict(Xte)):.2f}", fontsize=7.5)
        ax[i, 0].set_ylabel(key, color=FAMC[FAMILY[key]], fontsize=9, fontweight="bold")
    for j, t in enumerate(["PCA · regime", "PCA · nuisance", "t-SNE · regime", "t-SNE · nuisance"]):
        ax[0, j].set_title(t)
    handles = [Line2D([], [], ls="", marker="o", color=REG[k], label=f"regime {k}") for k in (0, 1)]
    fig.legend(handles=handles, loc="upper right", ncol=2, fontsize=8)
    fig.subplots_adjust(top=0.965, bottom=0.02)
    fig.suptitle(f"{c}: {CONDITIONS[c][0]} nuisance, sigma_n = {CONDITIONS[c][1]:g}  (nuisance colour: blue < 0 < red)", y=0.99, fontsize=11)
    plt.show()
''')
for c in ("N0", "F1", "F2", "S1", "S2"):
    code(f'latent_figure("{c}")')

# ------------------------------------------------------------------ 4. time domain
md("""
## 4. Time domain: what the features do over time, and what the HMM decodes

Each panel shows one held-out test series (seed 0) and three normalised traces:

- **black:** the learner's features projected on the **linear regime axis**, a logistic regression fitted on the other
  test series
- **gray:** the features' **first principal component**, the dominant direction an unsupervised model sees first
- **dotted:** the **true nuisance**

The strips underneath compare the true regime with the primary detector's decoding (HMM, k = 16, persistence
selection, refitted here as in scoring). Each HMM state is mapped to its majority regime. On S, the regime axis still
tracks the regime, but PC1 tracks the nuisance, and so does the HMM.
""")
code(r'''
HMM = dict(type="hmm", k=16, select="persistence")
def time_figure(c):
    y, u = win_regime(c), win_nuis(c); R = y.shape[0]; h = R // 2; s0 = h          # display the first held-out series
    keys = TRAINED + ["raw_flat", "raw_lagcov"]
    fig, ax = plt.subplots(len(keys), 1, figsize=(15, 1.75 * len(keys)), sharex=True)
    t = np.arange(y.shape[1]); nz = lambda v: v / (np.abs(v).max() + 1e-9)
    for a, key in zip(ax, keys):
        Z, _ = feats(c, key); d = Z.shape[-1]
        sc = StandardScaler().fit(Z[:h].reshape(-1, d))
        clf = LogisticRegression(C=0.1, max_iter=3000).fit(sc.transform(Z[:h].reshape(-1, d)), y[:h].ravel())
        score = clf.decision_function(sc.transform(Z[s0]))
        pc = PCA(1).fit(sc.transform(Z.reshape(-1, d))).transform(sc.transform(Z[s0]))[:, 0]
        if CONDITIONS[c][1] and np.corrcoef(pc, u[s0])[0, 1] < 0: pc = -pc
        a.fill_between(t, 0, 1, where=y[s0] == 1, transform=a.get_xaxis_transform(), color=RED, alpha=0.12, lw=0)
        a.plot(t, nz(score), color=INK, lw=1.1, label="linear regime axis (fit on other series)")
        a.plot(t, nz(pc), color=MUTED, lw=1, label="feature PC1")
        if CONDITIONS[c][1]: a.plot(t, nz(u[s0]), ":", color=INK2, lw=1, label="true nuisance")
        lab = make_detector(HMM).detect(Z, 2, seed=0)["labels"]
        m = {q: np.bincount(y[lab == q], minlength=2).argmax() for q in np.unique(lab)}
        dec = np.vectorize(m.get)(lab)
        a.bar(t, 0.22, bottom=-1.33, width=1.0, color=[REG[k] for k in y[s0]], lw=0)
        a.bar(t, 0.22, bottom=-1.62, width=1.0, color=[REG[k] for k in dec[s0]], lw=0)
        a.text(-2, -1.22, "truth", ha="right", va="center", fontsize=7, color=INK2)
        a.text(-2, -1.51, "HMM", ha="right", va="center", fontsize=7, color=INK2)
        a.set_ylim(-1.7, 1.1); a.set_yticks([])
        acc = balanced_accuracy_score(y[h:].ravel(), clf.predict(sc.transform(Z[h:].reshape(-1, d))))
        a.text(0.005, 0.97, f"{key}   linear-axis bal. acc {acc:.2f}   HMM window acc {(dec == y).mean():.2f}",
               transform=a.transAxes, color=FAMC[FAMILY[key]], fontsize=8.5, fontweight="bold", va="top")
    ax[0].legend(loc="lower right", bbox_to_anchor=(1, 1.12), ncol=3, fontsize=7.5)
    ax[-1].set_xlabel(f"window index (64 samples each), held-out test series {s0}")
    fig.suptitle(f"{c}: regime 1 shaded; bottom strips = true regime vs HMM-decoded regime (blue 0, red 1)", fontsize=10, x=0.3)
    plt.tight_layout(); plt.show()
''')
for c in ("N0", "F1", "F2", "S1", "S2"):
    code(f'time_figure("{c}")')

# ------------------------------------------------------------------ 5. training
md("""
## 5. Training dynamics (20k steps, 3 seeds)

Training loss is shown relative to its value at step 1000 (the collapse gate compares the final value with this).
Effective rank and HMM/GMM accuracy come from the periodic validation checks. Those use the default detector
settings, not the persistence-selected scoring detector, and are for monitoring only. Thin lines are seeds,
thick lines their mean.
""")
code(r'''
def logs(c, key, seed):
    p = os.path.join(benches[c].run_dir(spec(key, seed)), "log.jsonl")
    recs = [json.loads(l) for l in open(p)]
    tr = pd.DataFrame([r for r in recs if r["kind"] == "train"]); ev = pd.DataFrame([r for r in recs if r["kind"] == "eval"])
    return tr, ev
LS = dict(ae="-", vae="--", jepa_separate="-", jepa_shared="--", cpc="-", ts2vec="--")
rows = [("loss", "loss / loss at 1k", "tr"), ("eff_rank", "val eff. rank", "ev"), ("hmm_acc", "val HMM acc (monitor)", "ev"), ("probe", "val linear probe", "ev")]
fig, axes = plt.subplots(len(rows), 5, figsize=(18, 10), sharex=True, sharey="row")
for j, c in enumerate(CONDS):
    for l in TRAINED:
        curves = {m: [] for m, _, _ in rows}
        for s in (0, 1, 2):
            tr, ev = logs(c, l, s)
            base = tr.loc[tr.step == 1000, "loss"].iloc[0]
            for m, _, src in rows:
                d = tr if src == "tr" else ev
                yv = d[m] / base if m == "loss" else d[m]
                curves[m].append(pd.Series(yv.values, index=d.step.values))
        for i, (m, lab, _) in enumerate(rows):
            a = axes[i, j]
            for cs in curves[m]:
                a.plot(cs.index, cs.values, LS[l], color=FAMC[FAMILY[l]], lw=0.5, alpha=0.35)
            mm = pd.concat(curves[m], axis=1).mean(axis=1)
            a.plot(mm.index, mm.values, LS[l], color=FAMC[FAMILY[l]], lw=1.8, label=l)
            if j == 0: a.set_ylabel(lab)
            if i == 0: a.set_title(c)
axes[0, 0].set_ylim(0, 2.5); axes[0, -1].legend(fontsize=7, loc="upper right")
for a in axes[-1]:
    a.set_xlabel("step"); a.set_xticks([0, 5000, 10000, 15000, 20000], ["0", "5k", "10k", "15k", "20k"])
plt.tight_layout(); plt.show()
''')

# ------------------------------------------------------------------ 6. CPC check
md("""
## 6. CPC trainability check (label-free, before HPC release)

F2, seed 0, 3k steps. Selection used only the loss ratio (rule < 0.9) and the effective rank (rule > 2). No setting
passed, so CPC kept its defaults and was pre-declared "failed to train". In the confirmatory run none of its 15 runs
collapsed.
""")
code(r'''
cc = pd.read_csv("cpbench_results/recon_vs_pred_cpc_check.csv")
lab = [f"tau {r.temperature:g}\nlr {r.lr:g}" for r in cc.itertuples()]
fig, ax = plt.subplots(1, 2, figsize=(10, 3))
for a, (m, thr, name) in zip(ax, (("loss_ratio", 0.9, "loss at 3k / loss at 1k (must be < 0.9)"), ("eff_rank", 2, "effective rank (must be > 2)"))):
    if m == "loss_ratio":
        a.plot(range(len(cc)), cc[m], "o", color=AQUA, ms=9)
    else:
        a.bar(range(len(cc)), cc[m], color=AQUA, width=0.6)
    a.axhline(thr, ls="--", color=INK2, lw=1)
    for i, v in enumerate(cc[m]): a.text(i, v, f"{v:.2f}", ha="center", va="bottom", color=INK2, fontsize=8)
    a.set_xticks(range(len(cc)), lab); a.set_title(name)
ax[0].set_ylim(0.85, 1.02); ax[0].set_xlim(-0.5, len(cc) - 0.5)
plt.tight_layout(); plt.show()
''')

nb = nbf.v4.new_notebook(); nb["cells"] = cells
nb["metadata"]["kernelspec"] = dict(name="python3", display_name="Python 3", language="python")
out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "experiment_recon_vs_pred_visual.ipynb")
nbf.write(nb, out); print("written", out)
