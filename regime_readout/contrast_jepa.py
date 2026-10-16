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
ns["phase_surrogate"] = None
best_perm_acc, hmm_regression = ns["best_perm_acc"], ns["hmm_regression"]
m = ns["BigJEPA"]("separate"); ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m.load_state_dict(ck["model"]); m.eval(); print("checkpoint step", ck["step"], flush=True)
with torch.no_grad():
    Z = torch.cat([m.ctx(obs[k:k + 500]) for k in range(0, len(obs), 500)]).numpy().astype(np.float64).reshape(len(SER), 2000, -1)
np.savez_compressed("/private/tmp/claude-503/-Users-jq23948-Documents-JEPA-LINEAR/31242d0a-0d8a-4ab6-9dae-0e3f9e3e462f/scratchpad/gap1_latest_Z.npz", Z=Z, S=S)
R_, T_, d = Z.shape; K = 4
runs = [np.diff(np.flatnonzero(np.diff(np.r_[-1, s, -1]))) for s in S]
print(f"series {R_} x {T_} windows, d={d}; mean regime dwell {np.mean(np.concatenate(runs)):.1f} windows", flush=True)

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

rng = np.random.default_rng(0)
Zf = Z.reshape(-1, d); Zw = ((Zf - Zf.mean(0)) @ whitener(Zf)).reshape(R_, T_, d)
Zsur = phase_surrogate(Zw, rng)
# label-only diagnostics: regime-mean subspace in whitened coordinates
Mreg = np.array([Zw.reshape(-1, d)[S.reshape(-1) == k].mean(0) for k in range(K)])
Mreg = (Mreg - Mreg.mean(0)).T                                          # d x K, rank K-1
print("marginal excess kurtosis of whitened Z: max", kurt(Zw.reshape(-1, d)).max().round(3),
      "| along regime-mean directions:", np.round(kurt(Zw.reshape(-1, d) @ np.linalg.qr(Mreg)[0][:, :3]), 3), flush=True)
for k_ in range(K): pass

# ---- (1) scan m: contrast kurtosis spectrum, held-out replication, surrogate null
A, B = slice(0, R_ // 2), slice(R_ // 2, R_)
r = K - 1
print("\n(1) whitened-contrast max-kurtosis directions; fit on half the series, kurtosis re-measured on the other half")
print(f"{'m':>3} {'fit k1':>7} {'k2':>6} {'k3':>6} {'held k1':>8} {'k2':>6} {'k3':>6} | {'surr held k1':>12} {'k2':>6} {'k3':>6} | "
      f"{'angles to regime-mean subspace (labels, diagnostic)':>30}")
res = {}
for mm in (1, 2, 3, 4, 6, 8, 12, 16, 24):
    row = []
    for src in (Zw, Zsur):
        Dfit, Dhel = contrasts(src[A], mm), contrasts(src[B], mm)
        Wfit = whitener(Dfit)
        Wd = max_kurt_dirs((Dfit - Dfit.mean(0)) @ Wfit, r, rng)
        filt = Wfit @ Wd                                                 # directions as filters on Zw
        held = kurt((Dhel - Dhel.mean(0)) @ filt)
        row.append((kurt((Dfit - Dfit.mean(0)) @ filt), np.sort(held)[::-1], filt))
    (kf, kh, filt), (_, ks, _) = row
    res[mm] = (kh.mean() - ks.mean(), filt)
    print(f"{mm:3d} {kf[0]:7.3f} {kf[1]:6.3f} {kf[2]:6.3f} {kh[0]:8.3f} {kh[1]:6.3f} {kh[2]:6.3f} | {ks[0]:12.3f} {ks[1]:6.3f} {ks[2]:6.3f} | "
          + " ".join(f"{a:5.1f}" for a in angles(filt, Mreg[:, :3])), flush=True)

# ---- (2) pick m without labels: largest held-out excess over the surrogate; refit on all series
m_sel = max(res, key=lambda k: res[k][0])
D = contrasts(Zw, m_sel); Wfit = whitener(D)
filt = Wfit @ max_kurt_dirs((D - D.mean(0)) @ Wfit, r, rng)
P = Zw @ filt                                                            # (R, T, 3) projected latent
P = (P - P.reshape(-1, r).mean(0)) / P.reshape(-1, r).std(0)
print(f"\n(2) selected m = {m_sel} (label-free); angles to regime-mean subspace: {np.round(angles(filt, Mreg[:, :3]), 1)}")
Sf = S.reshape(-1); L = [T_] * R_
lab_g = GaussianMixture(K, n_init=5, random_state=0).fit(P.reshape(-1, r)).predict(P.reshape(-1, r))
lab_h = hmm_regression(None, P.reshape(-1, r), K, L, init=lab_g)
print(f"  per-window GMM on 3-D projection        {best_perm_acc(Sf, lab_g, K):.3f}")
print(f"  HMM on 3-D projection                   {best_perm_acc(Sf, lab_h, K):.3f}")
for wb in (2, 3, 4):
    WM = P[:, : T_ // wb * wb].reshape(R_, -1, wb, r).mean(2)
    tw = S[:, : T_ // wb * wb].reshape(R_, -1, wb)[..., 0].reshape(-1)  # label of first window in block
    print(f"  k-means on non-overlapping {wb}-window means {best_perm_acc(tw, kmeans(WM.reshape(-1, r), K, rng), K):.3f}")
# full-space references
lab_full = GaussianMixture(K, n_init=5, random_state=0, covariance_type="diag").fit(Zw.reshape(-1, d)).predict(Zw.reshape(-1, d))
print(f"  reference: GMM on full whitened 124-D Z {best_perm_acc(Sf, lab_full, K):.3f}")
Ph = lab_h.reshape(R_, T_)
print(f"  scaling exponent of HMM labels {scaling_exponent(Ph):.2f}   true S {scaling_exponent(S):.2f}")
n2 = R_ // 2
lp = lambda X: balanced_accuracy_score(S[n2:].reshape(-1), LogisticRegression(max_iter=3000).fit(X[:n2].reshape(-1, X.shape[-1]), S[:n2].reshape(-1)).predict(X[n2:].reshape(-1, X.shape[-1])))
print(f"  (labels, diagnostic) linear probe on the 3-D projection {lp(P):.3f}   on full Z {lp(Zw):.3f}")
