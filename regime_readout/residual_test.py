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

from sklearn.decomposition import FastICA
from sklearn.metrics import roc_auc_score
import warnings; warnings.filterwarnings("ignore")
best_perm_acc, hmm_regression = ns["best_perm_acc"], ns["hmm_regression"]
m = ns["BigJEPA"]("separate"); ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False); m.load_state_dict(ck["model"]); m.eval()
with torch.no_grad():
    enc = lambda f: torch.cat([f(obs[k:k+500]) for k in range(0, len(obs), 500)]).numpy().astype(np.float64).reshape(len(SER), 2000, -1)
    Z, Y = enc(m.ctx), enc(m.tgt); A = m.pred.weight.detach().numpy().astype(np.float64)      # y_hat = A z
R_, T_, d = Z.shape; K, h = 4, 2; Sf = S.reshape(-1); rng = np.random.default_rng(0)
Zc, Yc = Z - Z.reshape(-1, d).mean(0), Y - Y.reshape(-1, d).mean(0)
def regime_basis(X):
    M = np.array([X.reshape(-1, d)[Sf == k].mean(0) for k in range(K)]); return np.linalg.qr((M - M.mean(0)).T)[0][:, :K-1]
QZ, QY = regime_basis(Zc), regime_basis(Yc)                    # labels: diagnostics only
n2 = R_ // 2
def probe(X, Sl):
    X = X.reshape(R_, -1, X.shape[-1]); Sl = Sl.reshape(R_, -1)
    return balanced_accuracy_score(Sl[n2:].ravel(), LogisticRegression(max_iter=3000, C=0.1).fit(X[:n2].reshape(-1, X.shape[-1]), Sl[:n2].ravel()).predict(X[n2:].reshape(-1, X.shape[-1])))

print(f"step {ck['step']}, {R_} series x {T_} windows, d={d}, lag h={h}")
# ================= 1. same time step: CCA between heads, and where the regime sits
Wz, Wy = whitener(Zc.reshape(-1, d)), whitener(Yc.reshape(-1, d))
U_, cc, Vt_ = np.linalg.svd((Zc.reshape(-1, d) @ Wz).T @ (Yc.reshape(-1, d) @ Wy) / (R_ * T_))
cz = Zc.reshape(-1, d) @ Wz @ U_                               # canonical coordinates of Z, sorted by agreement with Y_t
print("\n1. SAME-TIME (Z_t, Y_t)")
print(f"   canonical correlations: >0.9: {(cc>0.9).sum()}  0.5-0.9: {((cc>0.5)&(cc<=0.9)).sum()}  <0.5: {(cc<=0.5).sum()}   (min {cc.min():.3f})")
for lo, hi, nm in ((0, 31, "agreeing  (top 31)"), (31, 93, "middle    (32-93)"), (93, 124, "disagreeing (last 31)")):
    print(f"   regime probe on {nm:<22} canonical dims: {probe(cz[:, lo:hi], Sf):.3f}")

# ================= 2. lagged pair in SVD(A) coordinates
Ua, sa, Vat = np.linalg.svd(A)
c = Zc[:, :-h] @ Vat.T                                         # context coords, (R, T-h, d)
u = Yc[:, h:] @ Ua                                             # target coords
r = u - c * sa                                                 # residual per singular direction
S0, Sh = S[:, :-h], S[:, h:]; sw = (S0 != Sh).ravel()
print(f"\n2. LAGGED PAIR (Z_t, Y_t+{h}) in SVD(A) coordinates; regime switches in {sw.mean():.1%} of pairs")
cor = np.array([np.corrcoef(c[..., i].ravel(), u[..., i].ravel())[0, 1] for i in range(d)])
rk = kurt(r.reshape(-1, d)); mk = kurt(c.reshape(-1, d))
shZ = ((QZ.T @ Vat.T) ** 2).sum(0); shY = ((QY.T @ Ua) ** 2).sum(0)
print(f"   sigma_i: top {np.round(sa[:4],2)}, #>0.5 {(sa>0.5).sum()};  corr(c_i,u_i): top {np.round(cor[:4],2)}")
# surrogate null: common-phase randomisation of the joint (Z,Y) series keeps all auto/cross covariances
J = phase_surrogate(np.concatenate([Zc, Yc], -1), rng); Zs, Ys = J[..., :d], J[..., d:]
rs = Ys[:, h:] @ Ua - (Zs[:, :-h] @ Vat.T) * sa; null_rk = kurt(rs.reshape(-1, d))
print(f"   residual excess kurtosis: real max {rk.max():.2f}, #>surrogate max ({np.abs(null_rk).max():.2f}): {(rk > np.abs(null_rk).max()).sum()}")
df = pd.DataFrame(dict(sigma=sa, corr=cor, resid_kurt=rk, ctx_marg_kurt=mk, share_ctx=shZ, share_tgt=shY))
pd.set_option("display.width", 200)
print("   top 8 singular directions by residual kurtosis:"); print(df.sort_values("resid_kurt", ascending=False).head(8).round(3).to_string())
print("   singular directions with >30% in the regime subspace (labels):"); print(df[(df.share_ctx > 0.3) | (df.share_tgt > 0.3)].round(3).to_string())
print(f"   (labels) regime-subspace share captured by top-k singular directions of A: "
      + "  ".join(f"k={k}: ctx {shZ[:k].sum()/3:.2f} tgt {shY[:k].sum()/3:.2f}" for k in (8, 16, 32, 64)))

# ---- 2b. readout: full ICA on whitened residuals, rank by residual kurtosis - |marginal kurtosis| of the matching state coordinate
rf = r.reshape(-1, d); Wr = whitener(rf)
ica = FastICA(n_components=d, whiten=False, fun="cube", max_iter=400, tol=1e-4, random_state=0).fit((rf - rf.mean(0)) @ Wr)
G = Wr @ ica.components_.T; G /= np.linalg.norm(G, axis=0)    # filters on residual coords (SVD-target basis)
Gy = Ua @ G                                                    # same filters as directions in Y-space
ykurt = kurt(Yc.reshape(-1, d) @ Gy); rkurt = kurt((rf - rf.mean(0)) @ G)
ex = rkurt - np.abs(ykurt); order = np.argsort(-ex)
shG = ((QY.T @ Gy) ** 2).sum(0)
print("\n   residual ICA, top 8 by excess (residual kurtosis − |marginal kurtosis of Y along it|):")
print(pd.DataFrame(dict(resid_kurt=rkurt[order[:8]], y_marg_kurt=ykurt[order[:8]], excess=ex[order[:8]], share_tgt=shG[order[:8]])).round(3).to_string())
def readout(Dirs, X, name):
    P = X.reshape(-1, d) @ np.linalg.qr(Dirs)[0]; P = (P - P.mean(0)) / P.std(0)
    lg = GaussianMixture(K, n_init=5, random_state=0).fit(P).predict(P); lh = hmm_regression(None, P, K, [X.shape[1]] * R_, init=lg)
    print(f"   {name:<52} GMM {best_perm_acc(S.reshape(-1)[:len(lg)] if X.shape[1]==T_ else S0.ravel(), lg, K):.3f}  HMM {best_perm_acc(S.reshape(-1) if X.shape[1]==T_ else S0.ravel(), lh, K):.3f}")
readout(Gy[:, order[:3]], Yc, "Y projected on top-3 residual-ICA directions")
readout(Ua[:, np.argsort(-rk)[:3]], Yc, "Y on top-3 singular directions by residual kurtosis")

# ---- 2c. change-point statistic: does the residual flag the pairs that straddle a switch?
rw = (rf - rf.mean(0)) @ Wr
stats = {"||whitened residual||² (all 124)": (rw ** 2).sum(1),
         "residual² on top-3 residual-ICA dirs": (((rf - rf.mean(0)) @ G[:, order[:3]]) ** 2).sum(1),
         "residual² on top-3 kurtosis singular dirs": (r.reshape(-1, d)[:, np.argsort(-rk)[:3]] ** 2).sum(1)}
Zw_ = (Zc.reshape(-1, d) @ Wz).reshape(R_, T_, d)
stats["baseline ||Z_t+2 − Z_t||² (whitened, no predictor)"] = ((Zw_[:, h:] - Zw_[:, :-h]) ** 2).sum(-1).ravel()
print("\n   change-point detection, AUC for 'regime switched between t and t+2' (labels used only to score):")
for k_, v in stats.items(): print(f"     {k_:<52} {roc_auc_score(sw, v):.3f}")

# ---- 2d. the confound: non-Gaussian residual directions that are NOT regime
nonreg = [i for i in order[:20] if shG[i] < 0.1]
print(f"\n   confound check: of the top-20 residual-ICA components by excess, {len(nonreg)} have <10% regime share;"
      f" their residual kurtosis {np.round(rkurt[nonreg][:6],1)} vs regime ones {np.round(rkurt[[i for i in order[:20] if shG[i] >= 0.5]],1)}")
