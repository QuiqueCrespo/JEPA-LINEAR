import os, sys, math, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, nbformat, pandas as pd
from dataclasses import dataclass
os.environ["OMP_NUM_THREADS"] = "4"; torch.set_num_threads(4)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.optimize import linear_sum_assignment
ns = dict(np=np, math=math, torch=torch, nn=nn, F=F, dataclass=dataclass, DEVICE=torch.device("cpu"), EMBED_DIM=124, HIDDEN=256,
          LogisticRegression=LogisticRegression, LinearRegression=LinearRegression, MLPClassifier=MLPClassifier, GaussianMixture=GaussianMixture,
          balanced_accuracy_score=balanced_accuracy_score, StandardScaler=StandardScaler, PCA=PCA, linear_sum_assignment=linear_sum_assignment, pd=pd)
# helper functions: evaluation + clustering cells of regime_large_scale.ipynb; model + generator from long_run.py
nb = nbformat.read("regime_large_scale.ipynb", 4)
for c in nb.cells:
    if c.cell_type == "code" and ("def cluster_battery" in c.source or "def predictable_variance" in c.source):
        exec(c.source, ns)
src = open("long_run.py").read()
for a, b in [("class SIGReg", "\ndef batch_whiten"), ("@dataclass\nclass BigSpec", "\ndef window_features"), ("class BigTCN", "\n# %%")]:
    i = src.index(a); j = src.find(b, i); chunk = src[i:j if j > 0 else len(src)]
    if a == "class BigTCN" and "EVAL_S" in chunk: chunk = chunk[: chunk.index("EVAL_S")]
    exec(chunk, ns)
spec = ns["BigSpec"](); ns["spec"] = spec
rng = np.random.default_rng(spec.seed)
_ = ns["simulate_big"](spec, 64, 2000, rng); test = ns["simulate_big"](spec, 16, 2000, rng)    # same test set as the run
torch.manual_seed(0); corr = ns["Corruptor"](spec)

# ---- helpers (from the 12-D simulation)
def contrasts(Z, m, step=1):
    R, T, d = Z.shape
    cs = np.concatenate([np.zeros((R, 1, d)), np.cumsum(Z, axis=1)], axis=1)
    t = np.arange(m, T - m + 1, step)
    return ((cs[:, t + m] - 2 * cs[:, t] + cs[:, t - m]) / m).reshape(-1, d)

def phase_surrogate(Z, rng):
    R, T, d = Z.shape
    Fz = np.fft.rfft(Z, axis=1)
    ph = np.exp(2j * np.pi * rng.random((R, Fz.shape[1], 1))); ph[:, 0] = 1
    return np.fft.irfft(Fz * ph, n=T, axis=1)

def whitener(X):
    w, V = np.linalg.eigh(np.cov(X.T)); w = np.maximum(w, 1e-8 * w.max())
    return V @ np.diag(w ** -0.5) @ V.T

def kurt(x):
    x = x - x.mean(0); return (x ** 4).mean(0) / (x ** 2).mean(0) ** 2 - 3

def max_kurt_dirs(X, r, rng, restarts=8, iters=300):
    """Deflationary fixed-point search for the r most positive-kurtosis directions of whitened X."""
    Wd = []
    for _ in range(r):
        best = (-np.inf, None)
        for _ in range(restarts):
            w = rng.standard_normal(X.shape[1])
            for _ in range(iters):
                for u in Wd: w -= (w @ u) * u
                w /= np.linalg.norm(w)
                y = X @ w
                wn = (X * (y ** 3)[:, None]).mean(0) - 3 * w
                for u in Wd: wn -= (wn @ u) * u
                wn /= np.linalg.norm(wn)
                done = abs(wn @ w) > 1 - 1e-9; w = wn
                if done: break
            k = kurt(X @ w)
            if k > best[0]: best = (k, w)
        Wd.append(best[1])
    return np.array(Wd).T

def scaling_exponent(lab):
    r = lambda l: np.mean(lab[:, :-l] != lab[:, l:])
    return np.log2((r(8) - r(4)) / (r(4) - r(2)))

def kmeans(X, K, rng, restarts=10, iters=100):
    best = (np.inf, None)
    for _ in range(restarts):
        c = X[rng.choice(len(X), K, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None] - c[None]) ** 2).sum(-1).argmin(1)
            new = np.array([X[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(K)])
            if np.allclose(new, c): break
            c = new
        cost = ((X - c[lab]) ** 2).sum()
        if cost < best[0]: best = (cost, lab)
    return best[1]

def angles(U, V):
    Qu, _ = np.linalg.qr(U); Qv, _ = np.linalg.qr(V)
    return np.degrees(np.arccos(np.clip(np.linalg.svd(Qu.T @ Qv, compute_uv=False), -1, 1)))


from scipy.ndimage import median_filter
from sklearn.decomposition import FastICA
import warnings; warnings.filterwarnings("ignore")
best_perm_acc, hmm_regression = ns["best_perm_acc"], ns["hmm_regression"]
dat = np.load(sys.argv[1]); Z, S = dat["Z"], dat["S"]; R_, T_, d = Z.shape; K = 4
rng = np.random.default_rng(0)
Zf = Z.reshape(-1, d); Zw = ((Zf - Zf.mean(0)) @ whitener(Zf)).reshape(R_, T_, d)
Mreg = np.array([Zw.reshape(-1, d)[S.reshape(-1) == k].mean(0) for k in range(K)]); Q = np.linalg.qr((Mreg - Mreg.mean(0)).T)[0][:, :K - 1]

# ---------------- candidate components (label-free)
cands, origin = [], []
for mm in (1, 2):
    D = contrasts(Zw, mm); Wd_ = whitener(D)
    ica = FastICA(n_components=d, whiten=False, fun="cube", max_iter=400, tol=1e-4, random_state=0).fit((D - D.mean(0)) @ Wd_)
    cands.append(Wd_ @ ica.components_.T); origin += [f"cICA m={mm}"] * d; print(f"contrast ICA m={mm} done, {ica.n_iter_} iters", flush=True)
F = np.concatenate(cands, 1); F /= np.linalg.norm(F, axis=0)
F = np.concatenate([F, Q], 1); origin += ["ORACLE regime dir"] * Q.shape[1]          # oracle rows: scored identically, excluded from selection
share = ((Q.T @ F) ** 2).sum(0)

# ---------------- label-free scores per component
def levels(c, L=3):
    """quantise a smoothed component series into L levels (1-D k-means); returns labels, margin to nearest boundary"""
    cs = np.apply_along_axis(lambda x: np.convolve(x, np.ones(3) / 3, mode="same"), 1, c)
    x = cs.reshape(-1); cen = np.quantile(x, (np.arange(L) + 0.5) / L)
    for _ in range(50):
        lab = np.abs(x[:, None] - cen[None]).argmin(1)
        cen = np.array([x[lab == k].mean() if (lab == k).any() else cen[k] for k in range(L)])
    cen = np.sort(cen); bnd = (cen[1:] + cen[:-1]) / 2
    lab = np.digitize(x, bnd); mg = np.abs(x[:, None] - bnd[None]).min(1)
    return lab.reshape(c.shape), mg.reshape(c.shape), np.bincount(lab, minlength=L) / len(lab)

def trans(lab, L, l):
    a, b = lab[:, :-l].ravel(), lab[:, l:].ravel(); M = np.zeros((L, L)); np.add.at(M, (a, b), 1)
    return M / np.maximum(M.sum(1, keepdims=True), 1)

def majority(lab, L, width=3):
    h = width // 2; oh = np.stack([(lab == k) for k in range(L)], -1).astype(float)
    cs = np.concatenate([np.zeros((lab.shape[0], 1, L)), np.cumsum(oh, 1)], 1); T = lab.shape[1]
    lo, hi = np.clip(np.arange(T) - h, 0, T), np.clip(np.arange(T) + h + 1, 0, T)
    return (cs[:, hi] - cs[:, lo]).argmax(-1)

rows = []
import time; t0 = time.time()
for j in range(F.shape[1]):
    if j % 100 == 0: print(f"scoring component {j}/{F.shape[1]}  {time.time()-t0:.0f}s", flush=True)
    c = Zw @ F[:, j]
    kc = [kurt(contrasts(c[..., None], mm))[0] for mm in (1, 2, 4, 8)]
    lab, mg, frac = levels(c)
    lf = majority(lab, 3)
    ck = max(np.abs(trans(lf, 3, L) - np.linalg.matrix_power(trans(lf, 3, 2), L // 2)).max() for L in (4, 8, 16))
    a, b, m_ = lab[:, :-5].ravel(), lab[:, 5:].ravel(), mg[:, :-5].ravel()
    g = np.digitize(m_, np.quantile(m_, [1 / 3, 2 / 3]))
    p = [np.mean(a[g == i] != b[g == i]) for i in range(3)]
    rows.append(dict(origin=origin[j], share=share[j], marg_kurt=kurt(c.reshape(-1, 1))[0], ck1=kc[0], ck2=kc[1], ck4=kc[2], ck8=kc[3],
                     minfrac=frac.min(), CK=ck, autonomy=p[0] / max(p[2], 1e-3)))
df = pd.DataFrame(rows)
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
print("oracle regime directions (labels used only to define them):"); print(df[df.origin.str.startswith("ORACLE")].round(3).to_string())
cand = df[~df.origin.str.startswith("ORACLE")]
print(f"\n{len(cand)} candidates; how many have >50% of their direction in the regime subspace: {(cand.share > 0.5).sum()}")
print("\ncandidates with share > 0.5 (diagnostic):"); print(cand[cand.share > 0.5].round(3).to_string())
print("\ntop 8 by contrast kurtosis at m=2 (the old criterion):"); print(cand.sort_values("ck2", ascending=False).head(8).round(3).to_string())

# ---------------- temporal-excess rule: non-Gaussian in the window contrast, Gaussian in the marginal
np.savez_compressed("contrast_rank_F.npz", F=F)
cand = cand.copy(); cand["excess"] = cand.ck1 - cand.marg_kurt.abs()
print("top 10 by temporal excess (contrast kurtosis m=1 minus |marginal kurtosis|):")
print(cand.sort_values("excess", ascending=False).head(10).round(3).to_string())
def readout(idx, name):
    G = np.linalg.qr(F[:, idx])[0]
    P = Zw.reshape(-1, d) @ G; P = (P - P.mean(0)) / P.std(0); Sf = S.reshape(-1)
    lab_g = GaussianMixture(K, n_init=5, random_state=0).fit(P).predict(P)
    lab_h = hmm_regression(None, P, K, [T_] * R_, init=lab_g)
    print(f"  {name:<44} dims {G.shape[1]:2d}  mean share {((Q.T @ G) ** 2).sum() / G.shape[1]:.2f}  GMM {best_perm_acc(Sf, lab_g, K):.3f}  HMM {best_perm_acc(Sf, lab_h, K):.3f}", flush=True)
print("\nlabel recovery (K = 4, chance 0.25)")
order = cand.sort_values("excess", ascending=False).index
for n in (2, 3, 4, 6):
    sel = []
    for j in order:
        if all(abs(F[:, j] @ F[:, i]) < 0.8 for i in sel): sel.append(j)
        if len(sel) == n: break
    readout(sel, f"temporal-excess rule, top {n} (de-duplicated)")
readout(list(df.index[df.origin.str.startswith("ORACLE")]), "ORACLE regime subspace")
# label-free choice of the number of directions: count components whose excess clears a phase-surrogate null
Zs_ = phase_surrogate(Zw, rng); D = contrasts(Zs_, 1); Wd_ = whitener(D)
ics = FastICA(n_components=d, whiten=False, fun="cube", max_iter=400, tol=1e-4, random_state=0).fit((D - D.mean(0)) @ Wd_)
Fs = Wd_ @ ics.components_.T; Fs /= np.linalg.norm(Fs, axis=0)
null = [kurt(contrasts((Zs_ @ Fs[:, j])[..., None], 1))[0] - abs(kurt((Zs_ @ Fs[:, j]).reshape(-1, 1))[0]) for j in range(d)]
thr = max(null); n_sig = []
for j in order:
    if cand.loc[j, "excess"] <= thr: break
    if all(abs(F[:, j] @ F[:, i]) < 0.8 for i in n_sig): n_sig.append(j)
print(f"\nphase-surrogate null: max temporal excess {thr:.2f}; distinct real components above it: {len(n_sig)}")
if n_sig: readout(n_sig, "all components above the surrogate null")
