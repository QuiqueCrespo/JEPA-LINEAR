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

# Robustness / baseline check of the temporal-excess contrast-ICA readout (rule fixed as in temporal_excess_method.html; m = 1 only).
from sklearn.decomposition import FastICA
import warnings, json; warnings.filterwarnings("ignore")
best_perm_acc, hmm_regression = ns["best_perm_acc"], ns["hmm_regression"]
src = sys.argv[1]; K = 4; R_, T_ = len(SER), 2000; rng = np.random.default_rng(0)
LR = "long_runs/separate_lam0.5_lr0.0003_fresh_gap1_scratch_"
def encode(model):
    with torch.no_grad():
        return torch.cat([model.ctx(obs[k:k + 500]) for k in range(0, len(obs), 500)]).numpy().astype(np.float64)
if src.startswith("shared"):                                             # e.g. shared_ckpt, shared_step10000
    m = ns["BigJEPA"]("shared"); sd = torch.load("long_runs/shared_lam0.5_lr0.0003_fresh_gap1_scratch_" + src[len("shared_"):] + ".pt",
                                                 map_location="cpu", weights_only=False)
    m.load_state_dict(sd["model"] if "model" in sd else sd); m.eval(); X = encode(m)
elif src.startswith("step") and src != "step55000":
    m = ns["BigJEPA"]("separate"); sd = torch.load(LR + src + ".pt", map_location="cpu", weights_only=False)
    m.load_state_dict(sd["model"] if "model" in sd else sd); m.eval(); X = encode(m)
elif src == "step55000":
    pass
elif src == "random_encoder":
    torch.manual_seed(0); m = ns["BigJEPA"]("separate"); m.eval(); X = encode(m)
else:
    o = obs.numpy().astype(np.float64)                                   # (N, W=32, D=128)
    if src == "raw_stats":
        mu, sd_ = o.mean(1), o.std(1) + 1e-6
        ac = ((o[:, 1:] - mu[:, None]) * (o[:, :-1] - mu[:, None])).mean(1) / sd_ ** 2
        X = np.concatenate([mu, np.log(sd_), ac], 1)
    elif src == "raw_spectra":
        X = np.log(np.abs(np.fft.rfft(o - o.mean(1, keepdims=True), axis=1)[:, 1:]) ** 2 + 1e-3).reshape(len(o), -1)
    elif src == "raw_flat":
        X = o.reshape(len(o), -1)
    del o
if src == "step55000":
    X = np.load("regime_readout/gap1_latest_Z.npz")["Z"].reshape(-1, 124)
X = X.reshape(-1, X.shape[-1]); X = X - X.mean(0)
if X.shape[1] > 124:                                                     # reduce to the JEPA's latent size
    X = PCA(124, random_state=0, svd_solver="randomized").fit_transform(X)
d = X.shape[1]; Zw = (X @ whitener(X)).reshape(R_, T_, d); Sf = S.reshape(-1)
n2 = R_ // 2
probe = balanced_accuracy_score(S[n2:].ravel(), LogisticRegression(max_iter=3000, C=0.1).fit(Zw[:n2].reshape(-1, d), S[:n2].ravel()).predict(Zw[n2:].reshape(-1, d)))
Mreg = np.array([Zw.reshape(-1, d)[Sf == k].mean(0) for k in range(K)]); Q = np.linalg.qr((Mreg - Mreg.mean(0)).T)[0][:, :K - 1]

def components(Zw):
    D = contrasts(Zw, 1); Wd_ = whitener(D)
    ica = FastICA(n_components=d, whiten=False, fun="cube", max_iter=400, tol=1e-4, random_state=0).fit((D - D.mean(0)) @ Wd_)
    F = Wd_ @ ica.components_.T; F /= np.linalg.norm(F, axis=0)
    C = Zw.reshape(-1, d) @ F
    mk = kurt(C); ck = kurt(contrasts(C.reshape(R_, T_, d), 1))
    return F, ck - np.abs(mk), ck, mk
F, ex, ck, mk = components(Zw)
order = np.argsort(-ex); sel = []
for j in order:
    if all(abs(F[:, j] @ F[:, i]) < 0.8 for i in sel): sel.append(j)
share = ((Q.T @ F) ** 2).sum(0)
_, ex_null, _, _ = components(phase_surrogate(Zw, rng)); thr = ex_null.max()
above = [j for j in sel if ex[j] > thr]
def readout(idx):
    if not idx: return float("nan"), float("nan")
    P = Zw.reshape(-1, d) @ np.linalg.qr(F[:, idx])[0]; P = (P - P.mean(0)) / P.std(0)
    lg = GaussianMixture(K, n_init=5, random_state=0).fit(P).predict(P); lh = hmm_regression(None, P, K, [T_] * R_, init=lg)
    return best_perm_acc(Sf, lg, K), best_perm_acc(Sf, lh, K)
g3, h3 = readout(sel[:3]); ga, ha = readout(above)
res = dict(source=src, supervised_linear_probe=probe, top_excess=[round(float(ex[j]), 2) for j in sel[:8]],
           top_share=[round(float(share[j]), 2) for j in sel[:8]], top_marg_kurt=[round(float(mk[j]), 2) for j in sel[:8]],
           null_max=round(float(thr), 2), n_above_null=len(above), gmm_top3=g3, hmm_top3=h3, gmm_above=ga, hmm_above=ha)
print(json.dumps(res), flush=True)
