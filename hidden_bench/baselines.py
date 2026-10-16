"""Acceptance test for the hidden-transition benchmark: cheap baselines must fail, the oracle must succeed.
usage: python hidden_bench/baselines.py <feature>   feature in {oracle, flat, stats, spectra, cov, bispec}"""
import os, sys, json, time, warnings
os.environ.setdefault("OMP_NUM_THREADS", "2")
import numpy as np, nbformat
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linear_sum_assignment
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from hidden_bench import HiddenSpec, simulate_hidden

ns = dict(np=np, GaussianMixture=GaussianMixture, linear_sum_assignment=linear_sum_assignment, PCA=PCA, StandardScaler=StandardScaler,
          LogisticRegression=LogisticRegression, LinearRegression=LinearRegression, balanced_accuracy_score=balanced_accuracy_score,
          MLPClassifier=MLPClassifier)
for c in nbformat.read("regime_large_scale.ipynb", 4).cells:
    if c.cell_type == "code" and ("def cluster_battery" in c.source or "def predictable_variance" in c.source):
        try: exec(c.source, ns)
        except Exception: pass
best_perm_acc, hmm_regression = ns["best_perm_acc"], ns["hmm_regression"]

feat = sys.argv[1]; R, T = 8, 2000; K = 4
spec = HiddenSpec()
t0 = time.time()
d = simulate_hidden(spec, R, T, np.random.default_rng(12345), return_obs=(feat != "oracle"))
S = d["S"]; N = R * T

def whiten(F, k):
    F = F - F.mean(0)
    P = PCA(k, random_state=0, svd_solver="randomized").fit_transform(F) if F.shape[1] > k else F
    return (P - P.mean(0)) / P.std(0)

if feat == "oracle":
    F = d["x"].reshape(N, -1)
else:
    o = d["obs"].reshape(N, spec.W, spec.D).astype(np.float64)
    if feat == "flat":
        F = o.reshape(N, -1)
    elif feat == "stats":
        mu, sd = o.mean(1), o.std(1) + 1e-6
        ac = ((o[:, 1:] - mu[:, None]) * (o[:, :-1] - mu[:, None])).mean(1) / sd ** 2
        F = np.concatenate([mu, np.log(sd), ac], 1)
    elif feat in ("spectra", "bispec"):
        X = np.fft.rfft(o - o.mean(1, keepdims=True), axis=1)            # (N, 33, D)
        if feat == "spectra":
            F = np.log(np.abs(X[:, 1:]) ** 2 + 1e-3).reshape(N, -1)
        else:
            nb = X.shape[1] - 1; pairs = [(a, b) for a in range(1, nb) for b in range(a, nb) if a + b <= nb]
            Xn = X / (np.sqrt((np.abs(X[:, 1:]) ** 2).mean((1, 2)))[:, None, None] + 1e-9)   # remove per-window volatility
            B = np.stack([(Xn[:, a] * Xn[:, b] * np.conj(Xn[:, a + b])).sum(-1) for a, b in pairs], 1)
            F = np.concatenate([B.real, B.imag], 1)
    elif feat == "cov":
        oc = o - o.mean(1, keepdims=True); C = np.einsum("nwi,nwj->nij", oc, oc) / spec.W
        iu = np.triu_indices(spec.D); F = C[:, iu[0], iu[1]]
    del o
Fw = whiten(F, min(64, F.shape[1])).reshape(R, T, -1)
res = dict(feature=feat, dim=int(F.shape[1]), gen_s=round(time.time() - t0))
Sf = S.reshape(-1); St = S[:, 1:].reshape(-1)
k1 = min(16, Fw.shape[-1])

# ---- unsupervised, single window
P1 = Fw[..., :k1].reshape(N, -1)
lg = GaussianMixture(K, n_init=3, random_state=0).fit(P1).predict(P1)
res["GMM single"] = best_perm_acc(Sf, lg, K)
res["HMM single"] = best_perm_acc(Sf, hmm_regression(None, P1, K, [T] * R, init=lg), K)
# ---- unsupervised, cross-window
k2 = min(8, Fw.shape[-1])
pair = np.concatenate([Fw[:, :-1, :k2], Fw[:, 1:, :k2]], -1).reshape(-1, 2 * k2)
lp = GaussianMixture(K, n_init=3, random_state=0).fit(pair).predict(pair)
res["GMM pair"] = best_perm_acc(St, lp, K)
for k in sorted({min(4, Fw.shape[-1]), k2, k1}):
    Xa, Ya = Fw[:, :-1, :k].reshape(-1, k), Fw[:, 1:, :k].reshape(-1, k)
    res[f"AR-HMM top{k}"] = best_perm_acc(St, hmm_regression(Xa, Ya, K, [T - 1] * R), K)
# ---- supervised diagnostics (labels): is the regime there at all?
h = R // 2
def sup(X, y, mlp=False):
    X = X.reshape(R, -1, X.shape[-1]); y = y.reshape(R, -1)
    clf = MLPClassifier((256, 256), max_iter=200, early_stopping=True, random_state=0) if mlp else LogisticRegression(max_iter=3000, C=0.1)
    sc = StandardScaler().fit(X[:h].reshape(-1, X.shape[-1]))
    return balanced_accuracy_score(y[h:].ravel(), clf.fit(sc.transform(X[:h].reshape(-1, X.shape[-1])), y[:h].ravel()).predict(sc.transform(X[h:].reshape(-1, X.shape[-1]))))
kk = min(32, Fw.shape[-1])
res["sup linear single"] = sup(Fw[..., :kk], S)
pairs = np.concatenate([Fw[:, :-1, :kk], Fw[:, 1:, :kk]], -1)
res["sup linear pair"] = sup(pairs, S[:, 1:])
res["sup MLP pair"] = sup(pairs, S[:, 1:], mlp=True)
if feat != "oracle":   # can x itself be read from these features?  (ridge R^2 on held-out series)
    Xf = Fw.reshape(R, T, -1); xs = d["x"]
    tgt = np.concatenate([np.cos(2 * np.arctan(xs)), np.sin(2 * np.arctan(xs))], -1)
    m = Ridge(1.0).fit(Xf[:h].reshape(-1, Xf.shape[-1]), tgt[:h].reshape(-1, tgt.shape[-1]))
    pr = m.predict(Xf[h:].reshape(-1, Xf.shape[-1])); tt = tgt[h:].reshape(-1, tgt.shape[-1])
    res["R2 of phase code (linear)"] = float(1 - ((pr - tt) ** 2).sum() / ((tt - tt.mean(0)) ** 2).sum())
res["total_s"] = round(time.time() - t0)
print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()}), flush=True)
