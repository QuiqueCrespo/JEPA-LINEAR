"""Super-long continuation of the large-scale linear-predictor LeJEPA (d=124), watching for a late phase change in
regime decodability. Generated from the same code as regime_large_scale.ipynb.

    caffeinate -i python long_run.py --design shared --steps 100000 --lam 0.5 --lr 3e-4
    caffeinate -i python long_run.py --design separate ...

Resumable: re-running the same command continues from long_runs/<tag>_ckpt.pt (saved every 5000 steps).
Logs: long_runs/<tag>_log.jsonl (losses every 100 steps, diagnostics every 2000, MLP probe + R² every 10000);
      long_runs/progress.log (one line per evaluation). Plot with regime_long_run_monitor.ipynb.
"""
import os, sys
os.environ.setdefault("MPLBACKEND", "Agg")
def display(x):
    print(x)

import os
for _v in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ.setdefault(_v, "4")      # small matrices: many threads only oversubscribe the CPU
import math, time, itertools, warnings
from dataclasses import dataclass, field, replace
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neural_network import MLPClassifier
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
torch.set_num_threads(4)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

torch.manual_seed(0)

FAST = '--fast' in sys.argv
EMBED_DIM = 124                      # latent dimension d of each encoder ("hidden dim 124")
HIDDEN = 256                         # backbone width
STEPS = 2000 if FAST else 20000      # long run
BATCH = 256
LR = 1e-3
LAM = 0.1                            # SIGReg weight (λ=0.1 reached the theory in the small notebook; λ=1 stalled)
EVAL_EVERY = 500 if FAST else 2000   # learning-curve checkpoints
SERIES = (8, 4) if FAST else (64, 16)
T_WIN = 1000 if FAST else 2000       # windows per series
OUT = "large_scale_runs"             # checkpoints + results (re-running loads them instead of retraining)
import os, json, pickle
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
os.makedirs(OUT, exist_ok=True)
C_SH, C_SEP, C_TH, C_B = "#2a78d6", "#eb6834", "#52514e", "#1baf7a"
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.alpha": 0.25, "lines.linewidth": 2, "figure.dpi": 110, "font.size": 9})

class SIGReg(torch.nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU!)

    Verbatim from lucas-maes/le-wm, module.py (the LeWM implementation of LeJEPA's SIGReg):
    Epps-Pulley test of every random 1-D projection against N(0,1), on 17 knots in [0,3].
    """

    def __init__(self, knots=17, num_proj=1024):
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        """
        proj: (T, B, D)
        """
        # sample random projections
        A = torch.randn(proj.size(-1), self.num_proj, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        # compute the epps-pulley statistic
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()  # average over projections and time


def batch_whiten(z, eps=1e-4):
    """Exact (in-batch) Cholesky whitening: the idealised constraint of the theory (W-MSE / VAMPnet style)."""
    zc = z - z.mean(0, keepdim=True)
    cov = zc.T @ zc / (z.shape[0] - 1) + eps * torch.eye(z.shape[1], device=z.device)
    L = torch.linalg.cholesky(cov.cpu()).to(z.device)  # cholesky is not implemented on MPS
    return torch.linalg.solve_triangular(L, zc.T, upper=False).T

def predictable_variance(Z, Y, shared):
    """||C||_F^2 with C = E[Y_{t+1} Z_t^T] after exact (empirical) whitening of each encoder's outputs.
    For the shared design the same whitening map is used for both sides (it is one encoder)."""
    d = Z.shape[-1]
    if shared:
        Wd = whiten_map(Z.reshape(-1, d))
        Zw = Yw = (Z - Z.reshape(-1, d).mean(0)) @ Wd
    else:
        Zw = (Z - Z.reshape(-1, d).mean(0)) @ whiten_map(Z.reshape(-1, d))
        Yw = (Y - Y.reshape(-1, d).mean(0)) @ whiten_map(Y.reshape(-1, d))
    a, b = Zw[:, :-1].reshape(-1, Zw.shape[-1]), Yw[:, 1:].reshape(-1, Yw.shape[-1])
    C = b.T @ a / len(a)
    return float((C ** 2).sum()), np.linalg.svd(C, compute_uv=False)


def whiten_map(A, eps=1e-6):
    A = A - A.mean(0)
    C = A.T @ A / len(A)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, eps * max(w.max(), eps))
    return V / np.sqrt(w)


def gt_features(spec, lat):
    """Named ground-truth features per window, each (N,T,k). These are the candidate features of the theory."""
    S, x, n = lat["S"], lat["x"], lat["n"]
    K, pi = spec.K, spec.pi
    onehot = np.eye(K)[S]
    f = {"regime": onehot[..., : K - 1]} if K > 1 else {}
    f["x"] = x
    f["g (x·1[S=k])"] = np.concatenate([x * onehot[..., k: k + 1] / math.sqrt(pi[k]) for k in range(K)], -1)
    f["|x|²"] = (x ** 2).sum(-1, keepdims=True)
    if spec.m == 2:
        f["x quad. (x1²-x2², x1x2)"] = np.stack([x[..., 0] ** 2 - x[..., 1] ** 2, x[..., 0] * x[..., 1]], -1)
    if spec.q:
        f["nuisance n"] = n
        quad = [n[..., i] * n[..., j] - (i == j) for i in range(spec.q) for j in range(i, spec.q)]
        f["nuisance n²"] = np.stack(quad, -1)
    return f


def r2_table(Lat, feats, n_fit):
    """Linear R^2 of each ground-truth feature from the latent (fit on first n_fit windows, test on the rest)."""
    out = {}
    L = Lat.reshape(-1, Lat.shape[-1])
    for name, F_ in feats.items():
        Fm = F_.reshape(-1, F_.shape[-1])
        r2s = []
        for j in range(Fm.shape[1]):
            reg = LinearRegression().fit(L[:n_fit], Fm[:n_fit, j])
            res = Fm[n_fit:, j] - reg.predict(L[n_fit:])
            r2s.append(1 - res.var() / (Fm[n_fit:, j].var() + 1e-12))
        out[name] = float(np.clip(np.mean(r2s), 0, 1))
    return out


def best_perm_acc(true, pred, K):
    M = np.zeros((K, K))
    for a, b in zip(true, pred):
        M[a, b] += 1
    r, c = linear_sum_assignment(-M)
    mapping = dict(zip(c, r))
    return balanced_accuracy_score(true, [mapping.get(p, -1) for p in pred])


def switching_regression(X, Y, K, restarts=5, iters=50, seed=0):
    """Unsupervised mixture of K linear regressions Y ~ A_k [X,1] (EM). Returns hard labels."""
    rng = np.random.default_rng(seed)
    N, dy = Y.shape
    Xd = np.c_[X, np.ones(N)]
    best, best_ll = None, -np.inf
    for r in range(restarts):
        R = rng.dirichlet(np.ones(K), N)
        for it in range(iters):
            logp = np.zeros((N, K))
            for k in range(K):
                w = R[:, k] + 1e-9
                A = np.linalg.solve(Xd.T @ (Xd * w[:, None]) + 1e-6 * np.eye(Xd.shape[1]), Xd.T @ (Y * w[:, None]))
                E = Y - Xd @ A
                Sig = (E * w[:, None]).T @ E / w.sum() + 1e-4 * np.eye(dy)
                Si = np.linalg.inv(Sig)
                logp[:, k] = (np.log(w.mean()) - 0.5 * np.linalg.slogdet(Sig)[1]
                              - 0.5 * np.einsum("ni,ij,nj->n", E, Si, E))
            mx = logp.max(1, keepdims=True)
            ll = float((mx[:, 0] + np.log(np.exp(logp - mx).sum(1))).sum())
            R = np.exp(logp - mx); R /= R.sum(1, keepdims=True)
        if ll > best_ll:
            best, best_ll = R.argmax(1), ll
    return best


def regime_readouts(Z, Ysame, S, K, shared, n_fit, seed=0):
    """How the regime can be read out of the latents (balanced accuracy, chance = 1/K)."""
    d = Z.shape[-1]
    Zf, Yf, Sf = Z.reshape(-1, d), Ysame.reshape(-1, d), S.reshape(-1)
    sc = StandardScaler().fit(Zf[:n_fit]); Zs = sc.transform(Zf)
    scy = StandardScaler().fit(Yf[:n_fit]); Ys = scy.transform(Yf)
    out = {}
    lin = LogisticRegression(max_iter=2000).fit(Zs[:n_fit], Sf[:n_fit])
    out["probe lin(Z)"] = balanced_accuracy_score(Sf[n_fit:], lin.predict(Zs[n_fit:]))
    mlp = MLPClassifier((64, 64), max_iter=200, early_stopping=True, random_state=seed).fit(Zs[:n_fit], Sf[:n_fit])
    out["probe MLP(Z)"] = balanced_accuracy_score(Sf[n_fit:], mlp.predict(Zs[n_fit:]))
    gm = GaussianMixture(K, n_init=3, random_state=seed).fit(Zs[:n_fit])
    lab = gm.predict(Zs[n_fit:])
    out["GMM(Z) acc"] = best_perm_acc(Sf[n_fit:], lab, K)
    if not shared:
        mlp = MLPClassifier((64, 64), max_iter=200, early_stopping=True, random_state=seed).fit(Ys[:n_fit], Sf[:n_fit])
        out["probe MLP(Y)"] = balanced_accuracy_score(Sf[n_fit:], mlp.predict(Ys[n_fit:]))
        ZY = np.c_[Zs, Ys]
        mlp = MLPClassifier((64, 64), max_iter=200, early_stopping=True, random_state=seed).fit(ZY[:n_fit], Sf[:n_fit])
        out["probe MLP(Z,Y)"] = balanced_accuracy_score(Sf[n_fit:], mlp.predict(ZY[n_fit:]))
        lab = switching_regression(Zs[n_fit:], Ys[n_fit:], K, seed=seed)
        out["switch-reg(Z→Y)"] = best_perm_acc(Sf[n_fit:], lab, K)
        if d == 1 and K == 2:
            out["sign(z·y)"] = best_perm_acc(Sf[n_fit:], (Zs[n_fit:, 0] * Ys[n_fit:, 0] > 0).astype(int), 2)
    return out


def evaluate_latents(Z, Y, spec, lat, shared, n_eval=8000, seed=0, clusters=True):
    """Z, Y: (N,T,d) context / target-encoder outputs on the SAME windows."""
    score, sv = predictable_variance(Z, Y, shared)
    feats = gt_features(spec, lat)
    N, T, d = Z.shape
    take = min(N * T, n_eval)
    idx = np.random.default_rng(seed).permutation(N * T)[:take]
    sub = lambda A: A.reshape(N * T, -1)[idx][None]
    n_fit = int(0.6 * take)
    fsub = {k: sub(v) for k, v in feats.items()}
    res = dict(score=score, canon=np.round(sv, 3))
    res.update({f"R²Z {k}": v for k, v in r2_table(sub(Z), fsub, n_fit).items()})
    if not shared:
        res.update({f"R²Y {k}": v for k, v in r2_table(sub(Y), fsub, n_fit).items()})
    if spec.K > 1:
        res.update(regime_readouts(sub(Z), sub(Y), sub(lat["S"][..., None])[..., 0], spec.K, shared, n_fit, seed))
        if clusters:
            res.update({f"clu {k}": v for k, v in cluster_battery(Z, lat["S"], spec.K, None if shared else Y, seed=seed).items()})
    cov = np.cov(Z.reshape(-1, d).T).reshape(d, d)
    ev = np.linalg.eigvalsh(cov)
    res["latent cov eig (min,max)"] = (round(float(ev.min()), 2), round(float(ev.max()), 2))
    return res

@dataclass
class BigSpec:
    K: int = 4
    p_stay: float = 0.9
    W: int = 32                                   # samples per window
    D: int = 128                                  # observed channels
    # regime-dependent state: 3 rotation planes; regime k sets each plane's frequency (cycles/sample) and damping
    reg_freqs: tuple = ((0.05, 0.12, 0.20), (0.12, 0.20, 0.05), (0.20, 0.05, 0.12), (0.08, 0.16, 0.26))
    reg_damp: tuple = (0.99, 0.99, 0.99, 0.97)
    # regime-independent features of different types
    slow_lams: tuple = (0.9, 0.95, 0.97)          # slow AR factors (window-level autocorrelation)
    osc_freqs: tuple = (0.03, 0.31)               # fixed-frequency oscillators (2 planes)
    osc_damp: float = 0.995
    vol_lam: float = 0.95                         # stochastic volatility (log-vol AR, window-level)
    vol_scale: float = 0.8
    gain_lam: float = 0.98                        # multiplicative sensor-gain drift
    gain_scale: float = 0.25
    n_distract: int = 4                           # white distractors (unpredictable)
    burst_rate: float = 0.02                      # heavy-tailed bursts (unpredictable)
    # corruption
    couplings: int = 4                            # invertible affine coupling layers
    coupling_strength: float = 1.0
    saturation: float = 3.0                       # soft clip a·tanh(o/a)
    lowpass_max: float = 0.6                      # per-channel first-order sensor dynamics within a window
    obs_noise: float = 0.2
    outliers: float = 0.01
    dropout: float = 0.05                         # whole channel missing in a window
    seed: int = 0

    @property
    def n_lat(self):
        return 6 + len(self.slow_lams) + 2 * len(self.osc_freqs) + 1 + self.n_distract + 1


def _rot(f, r):
    a = 2 * np.pi * f
    return r * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])


def simulate_big(spec, n_series, T, rng):
    """Sample-level latent process. Regimes switch at window boundaries (Markov, p_stay); every component has unit
    stationary variance. Returns per-sample latents u (N,T,W,n_lat), log-gain (N,T,W) and window-level labels."""
    K, W = spec.K, spec.W
    P = np.full((K, K), (1 - spec.p_stay) / (K - 1)); np.fill_diagonal(P, spec.p_stay)
    S = np.zeros((n_series, T), int); S[:, 0] = rng.integers(K, size=n_series)
    cum = P.cumsum(1)
    for t in range(1, T):
        S[:, t] = (rng.random(n_series)[:, None] > cum[S[:, t - 1]]).sum(1)
    B = np.stack([np.block([
        [_rot(spec.reg_freqs[k][0], spec.reg_damp[k]), np.zeros((2, 4))],
        [np.zeros((2, 2)), _rot(spec.reg_freqs[k][1], spec.reg_damp[k]), np.zeros((2, 2))],
        [np.zeros((2, 4)), _rot(spec.reg_freqs[k][2], spec.reg_damp[k])]]) for k in range(K)])
    sd_reg = np.sqrt(1 - np.array(spec.reg_damp) ** 2)
    a_slow = np.array(spec.slow_lams) ** (1 / W)
    Bo = [_rot(f, spec.osc_damp) for f in spec.osc_freqs]
    a_vol, a_gain = spec.vol_lam ** (1 / W), spec.gain_lam ** (1 / W)
    n_s, n_o = len(spec.slow_lams), 2 * len(spec.osc_freqs)
    N, TW = n_series, T * W
    x = rng.standard_normal((N, 6)); slow = rng.standard_normal((N, n_s)); osc = rng.standard_normal((N, n_o))
    h = rng.standard_normal(N); G = rng.standard_normal(N)
    X = np.zeros((N, TW, 6), np.float32); SL = np.zeros((N, TW, n_s), np.float32)
    OS = np.zeros((N, TW, n_o), np.float32); H = np.zeros((N, TW), np.float32); GG = np.zeros((N, TW), np.float32)
    for s in range(TW):
        k = S[:, s // W]
        x = np.einsum("nij,nj->ni", B[k], x) + sd_reg[k][:, None] * rng.standard_normal((N, 6))
        slow = a_slow * slow + np.sqrt(1 - a_slow ** 2) * rng.standard_normal((N, n_s))
        osc = np.concatenate([osc[:, 2 * i: 2 * i + 2] @ Bo[i].T for i in range(len(Bo))], 1) \
            + math.sqrt(1 - spec.osc_damp ** 2) * rng.standard_normal((N, n_o))
        h = a_vol * h + math.sqrt(1 - a_vol ** 2) * rng.standard_normal(N)
        G = a_gain * G + math.sqrt(1 - a_gain ** 2) * rng.standard_normal(N)
        X[:, s], SL[:, s], OS[:, s], H[:, s], GG[:, s] = x, slow, osc, h, G
    vol = np.exp(spec.vol_scale * H - spec.vol_scale ** 2) * rng.standard_normal((N, TW))
    distract = rng.standard_normal((N, TW, spec.n_distract))
    burst = (rng.random((N, TW)) < spec.burst_rate) * rng.standard_t(3, (N, TW)) * 3
    u = np.concatenate([X, SL, OS, vol[..., None], distract, burst[..., None]], -1).astype(np.float32)
    r = lambda A: A.reshape(N, T, W, *A.shape[2:])
    return dict(S=S, u=r(u), logg=r(spec.gain_scale * GG), x=r(X), slow=r(SL), osc=r(OS), h=r(H), G=r(GG))


class Corruptor(nn.Module):
    """Fixed random map latent (n_lat) -> D channels: up-projection, invertible affine couplings, soft saturation,
    multiplicative gain drift, per-channel low-pass sensor dynamics, noise, outliers, channel dropout.
    Everything before the noise is invertible (couplings) or monotone (saturation, gain, causal filter)."""

    def __init__(self, spec):
        super().__init__()
        g = torch.Generator().manual_seed(1000 + spec.seed)
        D, L = spec.D, spec.n_lat
        Q, _ = torch.linalg.qr(torch.randn(D, L, generator=g))
        self.register_buffer("P", Q * math.sqrt(D / L))                    # orthonormal columns -> injective
        self.perms = [torch.randperm(D, generator=g) for _ in range(spec.couplings)]
        h = D // 2
        self.A = nn.ParameterList([nn.Parameter(torch.randn(h, h, generator=g) * 1.5 / math.sqrt(h), requires_grad=False) for _ in range(spec.couplings)])
        self.C = nn.ParameterList([nn.Parameter(torch.randn(h, h, generator=g) * 1.5 / math.sqrt(h), requires_grad=False) for _ in range(spec.couplings)])
        self.register_buffer("alpha", torch.rand(D, generator=g) * spec.lowpass_max)
        self.spec = spec
        self.register_buffer("mu", torch.zeros(D)); self.register_buffer("sd", torch.ones(D))

    def clean(self, u, logg):
        s = self.spec
        o = u @ self.P.T
        h = s.D // 2
        for perm, A, C in zip(self.perms, self.A, self.C):
            o = o[..., perm.to(o.device)]
            a, b = o[..., :h], o[..., h:]
            b = b * torch.exp(0.5 * s.coupling_strength * torch.tanh(a @ A.T)) + s.coupling_strength * torch.tanh(a @ C.T)
            o = torch.cat([a, b], -1)
        o = s.saturation * torch.tanh(o / s.saturation)
        o = o * torch.exp(logg)[..., None]
        out, prev = [], o[:, 0]
        for t in range(o.shape[1]):                                       # filter within the window only
            prev = self.alpha * prev + (1 - self.alpha) * o[:, t]
            out.append(prev)
        return torch.stack(out, 1)

    def forward(self, u, logg, gen=None):
        s = self.spec
        o = self.clean(u, logg)
        o = o + s.obs_noise * torch.randn(o.shape, device=o.device, generator=gen)
        hit = torch.rand(o.shape, device=o.device, generator=gen) < s.outliers
        o = torch.where(hit, o + 5 * torch.randn(o.shape, device=o.device, generator=gen), o)
        keep = (torch.rand(o.shape[0], 1, o.shape[2], device=o.device, generator=gen) > s.dropout).float()
        return (o * keep - self.mu) / self.sd


def window_features(spec, lat):
    """Window-level ground truth, grouped by type (used for R² and the oracle)."""
    S, K = lat["S"], spec.K
    oh = np.eye(K)[S]
    x_end = lat["x"][:, :, -1]
    f = dict()
    f["regime indicator"] = oh[..., : K - 1]
    f["regime state x"] = x_end
    f["g = x·1[S=k]"] = np.concatenate([x_end * oh[..., k: k + 1] for k in range(K)], -1)
    xs = lat["x"]
    f["state energy per plane"] = np.stack([(xs[..., 2 * i: 2 * i + 2] ** 2).sum(-1).mean(-1) for i in range(3)], -1)
    f["slow factors"] = lat["slow"].mean(2)
    sl = f["slow factors"]
    f["slow factors²"] = np.stack([sl[..., i] * sl[..., j] for i in range(sl.shape[-1]) for j in range(i, sl.shape[-1])], -1)
    f["oscillators"] = lat["osc"][:, :, -1]
    f["log-volatility"] = lat["h"].mean(2)[..., None]
    f["gain drift"] = lat["G"].mean(2)[..., None]
    f["distractors"] = lat["u"][..., -1 - spec.n_distract: -1].mean(2)
    return f

spec = BigSpec()
rng = np.random.default_rng(spec.seed)
t0 = time.time()
train_lat = simulate_big(spec, SERIES[0], T_WIN, rng)
test_lat = simulate_big(spec, SERIES[1], T_WIN, rng)
print(f"simulated in {time.time()-t0:.0f}s: train {train_lat['u'].shape}, test {test_lat['u'].shape}")
corr = Corruptor(spec).to(DEVICE)
U_tr = torch.tensor(train_lat["u"], device=DEVICE); LG_tr = torch.tensor(train_lat["logg"], device=DEVICE)
with torch.no_grad():                                   # channel standardisation from a training sample
    i = torch.randint(SERIES[0], (2048,)); t = torch.randint(T_WIN, (2048,))
    o = corr(U_tr[i, t], LG_tr[i, t])
    corr.mu.copy_(o.mean((0, 1))); corr.sd.copy_(o.std((0, 1)) + 1e-6)

def render_test(lat, idx_series, idx_t, chunk=0):
    """Deterministic rendering of test windows (CPU generator seeded by chunk)."""
    g = torch.Generator().manual_seed(10_000 + chunk)
    u = torch.tensor(lat["u"][idx_series, idx_t]); lg = torch.tensor(lat["logg"][idx_series, idx_t])
    corr_cpu = corr.to("cpu")
    with torch.no_grad():
        o = corr_cpu(u, lg, gen=g)
    corr.to(DEVICE)
    return o

# what the encoder sees
o = render_test(test_lat, np.zeros(6, int), np.arange(6))
fig, axs = plt.subplots(1, 3, figsize=(13, 3.2), gridspec_kw=dict(width_ratios=[3, 1.2, 1.2]))
axs[0].imshow(o.reshape(-1, spec.D).T.numpy(), aspect="auto", cmap="Greys", interpolation="nearest")
axs[0].set_title(f"6 consecutive windows (regimes {test_lat['S'][0,:6].tolist()}), 128 channels × 32 samples"); axs[0].grid(False)
xs = test_lat["x"][0, :200].reshape(-1, 6)
axs[1].plot(xs[:400, 0], xs[:400, 1], lw=0.8, color=C_SH); axs[1].set_title("regime state, plane 1")
on = torch.tensor(test_lat["u"][0, :6]).reshape(-1, spec.n_lat) @ corr.P.cpu().T
axs[2].scatter(on[:, 0], o.reshape(-1, spec.D)[:, corr.perms[0][0]], s=3, color=C_TH)
axs[2].set_title("a channel before vs after corruption"); axs[2].set_xlabel("linear projection"); axs[2].set_ylabel("observed")
plt.tight_layout(); plt.show()

class BigTCN(nn.Module):
    def __init__(self, D, H, levels=5, k=3):
        super().__init__()
        self.inp = nn.Conv1d(D, H, 1)
        self.blocks = nn.ModuleList([nn.Sequential(
            nn.Conv1d(H, H, k, dilation=2 ** i, padding=2 ** i), nn.GroupNorm(8, H), nn.GELU(),
            nn.Conv1d(H, H, k, dilation=2 ** i, padding=2 ** i)) for i in range(levels)])
        self.out_dim = 2 * H

    def forward(self, x):
        h = self.inp(x.transpose(1, 2))
        for b in self.blocks:
            h = h + b(F.gelu(h))
        h = F.gelu(h)
        return torch.cat([h.mean(-1), h.amax(-1)], -1)


class BigEncoder(nn.Module):
    def __init__(self, D, d, H):
        super().__init__()
        self.phi = BigTCN(D, H)
        self.proj = nn.Sequential(nn.Linear(self.phi.out_dim, 2 * H), nn.BatchNorm1d(2 * H), nn.GELU(), nn.Linear(2 * H, d))

    def forward(self, x):
        return self.proj(self.phi(x))


class BigJEPA(nn.Module):
    def __init__(self, design, d=EMBED_DIM, H=HIDDEN, predictor="linear"):
        super().__init__()
        self.design = design
        self.ctx = BigEncoder(spec.D, d, H)
        self.tgt = self.ctx if design == "shared" else BigEncoder(spec.D, d, H)
        self.pred = nn.Linear(d, d, bias=False) if predictor == "linear" else nn.Sequential(
            nn.Linear(d, 512), nn.GELU(), nn.Linear(512, d))
        self.sigreg = SIGReg()

    def losses(self, xc, xn):
        if self.design == "shared":
            out = self.ctx(torch.cat([xc, xn])); z, y = out.chunk(2); reg = self.sigreg(out[None])
        else:
            z, y = self.ctx(xc), self.tgt(xn); reg = 0.5 * (self.sigreg(z[None]) + self.sigreg(y[None]))
        return (y - self.pred(z)).square().mean(), reg

EVAL_S, EVAL_T = np.repeat(np.arange(SERIES[1]), T_WIN), np.tile(np.arange(T_WIN), SERIES[1])
test_feats = window_features(spec, test_lat)

@torch.no_grad()
def encode_test(fn, n_series=None, chunk=2000):
    """Encode test windows of the first n_series test series (all windows, in order) -> (n, T, d) arrays."""
    n_series = n_series or SERIES[1]
    outs = []
    sel = np.arange(n_series * T_WIN)
    for c, i in enumerate(range(0, len(sel), chunk)):
        j = sel[i: i + chunk]
        o = render_test(test_lat, EVAL_S[j], EVAL_T[j], chunk=c).to(DEVICE)
        outs.append([a.cpu().numpy().astype(np.float64) for a in fn(o)])
    return [np.concatenate([o[k] for o in outs]).reshape(n_series, T_WIN, -1) for k in range(len(outs[0]))]

def quick_eval(model, n_series=4):
    n_series = min(n_series, SERIES[1])
    model.eval()
    Z, Y = encode_test(lambda o: (model.ctx(o), model.tgt(o)), n_series)
    model.train()
    score, sv = predictable_variance(Z, Y, model.design == "shared")
    Zf, S = Z.reshape(-1, Z.shape[-1]), test_lat["S"][:n_series].reshape(-1)
    n = len(S) // 2
    sc = StandardScaler().fit(Zf[:n])
    lin = LogisticRegression(max_iter=1000, C=0.1).fit(sc.transform(Zf[:n]), S[:n])
    return dict(score=score, top_canon=float(sv[0]), lin_probe=balanced_accuracy_score(S[n:], lin.predict(sc.transform(Zf[n:]))))

def train_big(design, predictor="linear", steps=STEPS, seed=0, tag=None):
    tag = tag or f"{design}_{predictor}_d{EMBED_DIM}"
    path = f"{OUT}/{tag}.pt"
    model = BigJEPA(design, predictor=predictor).to(DEVICE)
    if os.path.exists(path):
        ck = torch.load(path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ck["model"]); model.eval()
        print(f"loaded {path}")
        return model, ck["curve"]
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, LR, total_steps=steps, pct_start=0.05)
    curve, t0 = [], time.time()
    for step in range(steps + 1):
        if step % EVAL_EVERY == 0:
            ev = quick_eval(model)
            curve.append(dict(step=step, **ev, pred=float(pred) if step else np.nan, reg=float(rg) if step else np.nan))
            msg = (f"[{tag}] step {step:6d}  {time.time()-t0:6.0f}s  score {ev['score']:.2f}  "
                   f"lin-probe {ev['lin_probe']:.3f}  top canon {ev['top_canon']:.3f}")
            print(msg, flush=True)
            with open(f"{OUT}/progress.log", "a") as f:   # follow with: tail -f large_scale_runs/progress.log
                f.write(time.strftime("%H:%M:%S ") + msg + "\n")
        if step == steps:
            break
        i = torch.randint(SERIES[0], (BATCH,), device=DEVICE); t = torch.randint(T_WIN - 1, (BATCH,), device=DEVICE)
        with torch.no_grad():
            xc, xn = corr(U_tr[i, t], LG_tr[i, t]), corr(U_tr[i, t + 1], LG_tr[i, t + 1])
        pred, rg = model.losses(xc, xn)
        loss = pred + LAM * rg
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step(); sched.step()
    model.eval()
    torch.save(dict(model=model.state_dict(), curve=curve), path)
    return model, curve

def regime_report(Z, Y=None, name="", n_eval=16000, seed=0):
    """Regime recoverability from a latent (and, for separate encoders, from the pair on the same input)."""
    N, T, d = Z.shape
    S = test_lat["S"][:N]
    take = min(N * T, n_eval)
    Zf, Sf = Z.reshape(-1, d)[:take], S.reshape(-1)[:take]           # keep temporal order for smoothing
    n = int(0.6 * take)
    sc = StandardScaler().fit(Zf[:n]); Zs = sc.transform(Zf)
    K = spec.K
    r = dict(latent=name)
    lin = LogisticRegression(max_iter=2000, C=0.1).fit(Zs[:n], Sf[:n])
    r["linear probe"] = balanced_accuracy_score(Sf[n:], lin.predict(Zs[n:]))
    mlp = MLPClassifier((256, 256), max_iter=200, early_stopping=True, random_state=seed).fit(Zs[:n], Sf[:n])
    r["MLP probe"] = balanced_accuracy_score(Sf[n:], mlp.predict(Zs[n:]))
    from sklearn.decomposition import PCA
    for k in [8, 32]:
        pcs = PCA(min(k, d), random_state=seed).fit(Zs[:n]).transform(Zs)
        gm = GaussianMixture(K, n_init=3, random_state=seed).fit(pcs[:n])
        r[f"GMM on PCA-{k}"] = best_perm_acc(Sf[n:], gm.predict(pcs[n:]), K)
        if k == 8:   # unsupervised + temporal smoothing of the posteriors over ±2 windows (change-point style)
            post = gm.predict_proba(pcs[n:])
            sm = np.stack([np.convolve(post[:, j], np.ones(5) / 5, mode="same") for j in range(K)], 1)
            r["GMM PCA-8 + smoothing"] = best_perm_acc(Sf[n:], sm.argmax(1), K)
    if Y is not None:
        Yf = Y.reshape(-1, d)[:take]; Ys = StandardScaler().fit(Yf[:n]).transform(Yf)
        from sklearn.cross_decomposition import CCA
        cca = CCA(8, max_iter=1000).fit(Zs[:n], Ys[:n])
        a, b = cca.transform(Zs[n:], Ys[n:])
        r["switch-reg on top-8 CCA(Z,Y)"] = best_perm_acc(Sf[n:], switching_regression(a, b, K, seed=seed), K)
        mlp = MLPClassifier((256, 256), max_iter=200, early_stopping=True, random_state=seed).fit(np.c_[Zs, Ys][:n], Sf[:n])
        r["MLP probe (Z,Y)"] = balanced_accuracy_score(Sf[n:], mlp.predict(np.c_[Zs, Ys][n:]))
    r.update({f"clu {k}": v for k, v in cluster_battery(Z, S, K, Y, seed=seed).items()})
    return r

def r2_by_type(Z, name, n_eval=16000):
    N = Z.shape[0]
    feats = {k: v[:N].reshape(N * T_WIN, -1)[:n_eval][None] for k, v in test_feats.items()}
    out = r2_table(Z.reshape(-1, Z.shape[-1])[:n_eval][None], feats, int(0.6 * min(n_eval, N * T_WIN)))
    return dict(latent=name, **out)

# ---- super-long continuation run ------------------------------------------------------------------
import json, argparse, matplotlib
matplotlib.use("Agg")
ap = argparse.ArgumentParser()
ap.add_argument("--design", default="shared")
ap.add_argument("--steps", type=int, default=100_000)        # total steps of this run (resumable)
ap.add_argument("--lam", type=float, default=0.5)
ap.add_argument("--lr", type=float, default=3e-4)            # constant after warm-up: no annealing, so late dynamics stay visible
ap.add_argument("--warmup", type=int, default=1000)
ap.add_argument("--init", default=None)                      # default: the λ=0.5 resume checkpoint of the same design
ap.add_argument("--fast", action="store_true")
ap.add_argument("--fresh", action="store_true")              # simulate new latent trajectories for every batch (no reuse)
ap.add_argument("--gap", type=int, default=0)                # windows skipped between context and target (needs --fresh)
ap.add_argument("--scratch", action="store_true")            # start from a random initialisation instead of a checkpoint
args, _ = ap.parse_known_args()
LONG = "long_runs"
os.makedirs(LONG, exist_ok=True)
assert args.gap == 0 or args.fresh, "--gap requires --fresh"
TAG = (f"{args.design}_lam{args.lam}_lr{args.lr}" + ("_fresh" if args.fresh else "") + (f"_gap{args.gap}" if args.gap else "")
       + ("_scratch" if args.scratch else ""))
CKPT, LOG = f"{LONG}/{TAG}_ckpt.pt", f"{LONG}/{TAG}_log.jsonl"
DENSE_EVERY, EVAL_EVERY_L, DEEP_EVERY, SAVE_EVERY = (10, 50, 100, 100) if args.fast else (100, 2000, 10_000, 5_000)

def log(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec) + "\n")

THETA = 0.867   # predictability of the regime indicator (window level)

@torch.no_grad()
def probe_eval(model, deep=False):
    """Regime and predictability diagnostics on 4 test series (8000 consecutive windows)."""
    model.eval()
    Z, Y = encode_test(lambda o: (model.ctx(o), model.tgt(o)), min(4, SERIES[1]))
    model.train()
    shared = model.design == "shared"
    Zl, Yl = Z[:, :: args.gap + 1], Y[:, :: args.gap + 1]          # consecutive entries are gap+1 windows apart
    score, sv = predictable_variance(Zl, Yl, shared)
    Zf = Z.reshape(-1, Z.shape[-1]); S = test_lat["S"][: Z.shape[0]].reshape(-1)
    n = len(S) // 2
    sc = StandardScaler().fit(Zf[:n]); Zs = sc.transform(Zf)
    lin = LogisticRegression(max_iter=1000, C=0.1).fit(Zs[:n], S[:n])
    ev = np.linalg.eigvalsh(np.cov(Zf.T))
    onehot = np.eye(spec.K)[S][:, : spec.K - 1]
    r2_reg = r2_table(Zf[None], {"regime": onehot[None]}, n)["regime"]
    rec = dict(score=score, top_canon=float(sv[0]), canon_above_theta=int((sv > THETA).sum()),
               lin_probe=balanced_accuracy_score(S[n:], lin.predict(Zs[n:])), r2_regime=r2_reg,
               dirs_var_below_0_1=int((ev < 0.1).sum()), cov_eig_max=float(ev.max()))
    if deep:
        mlp = MLPClassifier((256, 256), max_iter=200, early_stopping=True, random_state=0).fit(Zs[:n], S[:n])
        rec["mlp_probe"] = balanced_accuracy_score(S[n:], mlp.predict(Zs[n:]))
        rec.update({f"r2 {k}": v for k, v in r2_by_type(Z, "x").items() if k != "latent"})
    return rec

model = BigJEPA(args.design).to(DEVICE)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
lr_at = lambda s: args.lr * min(1.0, (s + 1) / args.warmup)
start = 0
if os.path.exists(CKPT):                                  # resume an interrupted run
    ck = torch.load(CKPT, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); start = ck["step"]
    torch.set_rng_state(ck["rng"].cpu())
    print(f"resumed {CKPT} at step {start}", flush=True)
else:
    if args.scratch:
        init = "random initialisation"
        torch.manual_seed(0); model = BigJEPA(args.design).to(DEVICE)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    else:
        init = args.init or f"{OUT}/{args.design}_linear_d{EMBED_DIM}_resume_lam0.5.pt"
        model.load_state_dict(torch.load(init, map_location=DEVICE, weights_only=False)["model"])
    print(f"initialised from {init}", flush=True)
    log(dict(kind="meta", tag=TAG, **{**vars(args), "init": init}, time=time.time()))
model.train()
acc, t0 = np.zeros(3), time.time()
for step in range(start, args.steps + 1):
    if step % EVAL_EVERY_L == 0:
        deep = step % DEEP_EVERY == 0
        rec = probe_eval(model, deep)
        log(dict(kind="eval", step=step, deep=deep, time=time.time(), **rec))
        msg = (f"[long {TAG}] step {step:7d} {time.time()-t0:7.0f}s score {rec['score']:.2f} lin-probe {rec['lin_probe']:.3f} "
               f"R²(regime) {rec['r2_regime']:.3f} canon>θ {rec['canon_above_theta']} collapsed {rec['dirs_var_below_0_1']}"
               + (f" MLP-probe {rec['mlp_probe']:.3f}" if deep else ""))
        print(msg, flush=True)
        with open(f"{LONG}/progress.log", "a") as f:
            f.write(time.strftime("%H:%M:%S ") + msg + ("   <<< regime appearing" if rec["lin_probe"] > 0.4 else "") + "\n")
    if step % SAVE_EVERY == 0 and step > start:
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), step=step, rng=torch.get_rng_state()), CKPT)
        if step % DEEP_EVERY == 0:
            torch.save(model.state_dict(), f"{LONG}/{TAG}_step{step}.pt")     # snapshots for later inspection
    if step == args.steps:
        break
    for gr in opt.param_groups:
        gr["lr"] = lr_at(step)
    with torch.no_grad():
        if args.fresh:
            # every component starts from its stationary law, so two simulated windows are an exact stationary pair
            g = args.gap
            fr = simulate_big(spec, BATCH, g + 2, np.random.default_rng([12345, step]))
            u = torch.tensor(fr["u"][:, [0, g + 1]], device=DEVICE); lg = torch.tensor(fr["logg"][:, [0, g + 1]], device=DEVICE)
            xc, xn = corr(u[:, 0], lg[:, 0]), corr(u[:, 1], lg[:, 1])
        else:
            i = torch.randint(SERIES[0], (BATCH,), device=DEVICE); t = torch.randint(T_WIN - 1, (BATCH,), device=DEVICE)
            xc, xn = corr(U_tr[i, t], LG_tr[i, t]), corr(U_tr[i, t + 1], LG_tr[i, t + 1])
    pred, rg = model.losses(xc, xn)
    loss = pred + args.lam * rg
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    opt.step()
    acc += [pred.item(), rg.item(), loss.item()]
    if (step + 1) % DENSE_EVERY == 0:
        log(dict(kind="loss", step=step + 1, pred=acc[0] / DENSE_EVERY, sigreg=acc[1] / DENSE_EVERY,
                 total=acc[2] / DENSE_EVERY, lr=lr_at(step)))
        acc[:] = 0
print("done", flush=True)
