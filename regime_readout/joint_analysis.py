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
with torch.no_grad():
    u = torch.tensor(_["u"][:8, :256]).reshape(-1, spec.W, spec.n_lat); lg = torch.tensor(_["logg"][:8, :256]).reshape(-1, spec.W)
    o = corr(u, lg, gen=torch.Generator().manual_seed(1)); corr.mu.copy_(o.mean((0, 1))); corr.sd.copy_(o.std((0, 1)) + 1e-6)
    SER = list(range(16))
    obs = torch.cat([corr(torch.tensor(test["u"][s]), torch.tensor(test["logg"][s]), gen=torch.Generator().manual_seed(10_000 + s)) for s in SER])
S = test["S"][SER]
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

# Is regime structure in the UNREGULARISED joint law of the two encoders' outputs?
from sklearn.decomposition import FastICA
import warnings; warnings.filterwarnings("ignore")
best_perm_acc, hmm_regression = ns["best_perm_acc"], ns["hmm_regression"]
m = ns["BigJEPA"]("separate"); ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); m.load_state_dict(ck["model"]); m.eval()
with torch.no_grad():
    enc = lambda f: torch.cat([f(obs[k:k + 500]) for k in range(0, len(obs), 500)]).numpy().astype(np.float64).reshape(len(SER), 2000, -1)
    Z, Y = enc(m.ctx), enc(m.tgt)
np.savez_compressed("regime_readout/gap1_latest_ZY.npz", Z=Z, Y=Y, S=S)
R_, T_, d = Z.shape; K = 4; rng = np.random.default_rng(0); n2 = R_ // 2
def white(X):
    X = X - X.mean(0); w, V = np.linalg.eigh(np.cov(X.T)); keep = w > 1e-6 * w.max()
    return X @ V[:, keep] / np.sqrt(w[keep])
def probe(X, Sl):
    X = X.reshape(R_, -1, X.shape[-1]); Sl = Sl.reshape(R_, -1)
    return balanced_accuracy_score(Sl[n2:].ravel(), LogisticRegression(max_iter=3000, C=0.1).fit(X[:n2].reshape(-1, X.shape[-1]), Sl[:n2].ravel()).predict(X[n2:].reshape(-1, X.shape[-1])))
def cluster(P, Sl, Tl):
    P = (P - P.mean(0)) / P.std(0)
    lg = GaussianMixture(K, n_init=5, random_state=0).fit(P).predict(P)
    return best_perm_acc(Sl, lg, K), best_perm_acc(Sl, hmm_regression(None, P, K, [Tl] * R_, init=lg), K)

for lag in (0, 2):
    Zc = Z[:, : T_ - lag].reshape(-1, d); Yc = Y[:, lag:].reshape(-1, d); Sl = S[:, : T_ - lag].ravel(); Tl = T_ - lag
    print(f"\n================ {'SAME TIME (Z_t, Y_t)' if lag == 0 else f'LAGGED (Z_t, Y_t+{lag})'} ================")
    # canonical coordinates
    Wz, Wy = whitener(Zc), whitener(Yc); A_ = (Zc - Zc.mean(0)) @ Wz; B_ = (Yc - Yc.mean(0)) @ Wy
    U, cc, Vt = np.linalg.svd(A_.T @ B_ / len(A_)); c, u = A_ @ U, B_ @ Vt.T
    # ---- 1. joint ICA on [Z, Y]
    J = white(np.concatenate([Zc, Yc], 1)); dj = J.shape[1]
    sub = rng.choice(len(J), 16000, replace=False)
    ica = FastICA(n_components=dj, whiten=False, fun="cube", max_iter=400, tol=1e-4, random_state=0).fit(J[sub])
    C = J @ ica.components_.T; kc = kurt(C)
    # how much of each component comes from Z vs Y: regress component on Z alone and on Y alone
    r2z = np.array([1 - np.var(C[:, j] - A_ @ np.linalg.lstsq(A_[sub], C[sub, j], rcond=None)[0]) / np.var(C[:, j]) for j in range(dj)])
    r2y = np.array([1 - np.var(C[:, j] - B_ @ np.linalg.lstsq(B_[sub], C[sub, j], rcond=None)[0]) / np.var(C[:, j]) for j in range(dj)])
    ordr = np.argsort(kc)
    print(f"  joint ICA ({dj} comps): kurtosis min {kc.min():.2f} max {kc.max():.2f}; most negative 6: {np.round(kc[ordr[:6]], 2)}")
    print(f"     their R² from Z alone {np.round(r2z[ordr[:6]], 2)}  from Y alone {np.round(r2y[ordr[:6]], 2)}")
    print(f"     comps that are 'cross' (R² < 0.5 from Z alone AND from Y alone): {int(((r2z < .5) & (r2y < .5)).sum())}; "
          f"their kurtosis range {np.round(kc[(r2z < .5) & (r2y < .5)].min(), 2) if ((r2z < .5) & (r2y < .5)).any() else '-'} .. "
          f"{np.round(kc[(r2z < .5) & (r2y < .5)].max(), 2) if ((r2z < .5) & (r2y < .5)).any() else '-'}")
    cross = np.flatnonzero((r2z < .5) & (r2y < .5))
    print(f"  (labels) probe: Z {probe(Zc, Sl):.3f}  Y {probe(Yc, Sl):.3f}  cross components only {probe(C[:, cross], Sl) if len(cross) else float('nan'):.3f}")
    for nm, idx in (("3 most negative-kurtosis joint comps", ordr[:3]), ("6 most negative", ordr[:6]), ("3 most positive", ordr[-3:])):
        g, h = cluster(C[:, idx], Sl, Tl); print(f"     unsupervised on {nm:<38} GMM {g:.3f}  HMM {h:.3f}   probe {probe(C[:, idx], Sl):.3f}")
    # ---- 2. canonical pair products (multi-dim analogue of sign(z·y)) and differences
    prod = c * u; diff = c - u
    print(f"  canonical correlations top {np.round(cc[:3], 3)}, median {np.median(cc):.3f}")
    kp, kd = kurt(prod), kurt(diff)
    print(f"  pair products: probe (labels) all {probe(prod, Sl):.3f}; top-32 pairs {probe(prod[:, :32], Sl):.3f}")
    print(f"  pair differences c_i − u_i: kurtosis range {kd.min():.2f}..{kd.max():.2f}; probe {probe(diff, Sl):.3f}")
    Pp = white(prod)
    icap = FastICA(n_components=Pp.shape[1], whiten=False, fun="cube", max_iter=400, tol=1e-4, random_state=0).fit(Pp[sub])
    Cp = Pp @ icap.components_.T; kcp = kurt(Cp); o2 = np.argsort(kcp)
    for nm, idx in (("3 most negative-kurtosis product ICs", o2[:3]), ("3 most positive", o2[-3:])):
        g, h = cluster(Cp[:, idx], Sl, Tl); print(f"     unsupervised on {nm:<38} GMM {g:.3f}  HMM {h:.3f}   probe {probe(Cp[:, idx], Sl):.3f}  (kurt {np.round(kcp[idx], 2)})")
