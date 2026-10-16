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


from hidden_bench import make_mixing
spec = HiddenSpec(); mix = make_mixing(spec); R, T = 8, 2000
d = simulate_hidden(spec, R, T, np.random.default_rng(12345), mix=mix)
o = d["obs"].astype(np.float64).reshape(R * T, spec.W, spec.D); src = o @ np.linalg.pinv(mix["M"]).T
tau = np.arange(spec.W); win = np.hanning(spec.W); est = []
for i, (fa, fb) in enumerate(spec.triplets):
    z = [src[:, :, 3 * i + j] * np.exp(-2j * np.pi * f * tau) for j, f in enumerate((fa, fb, fa + fb))]
    z = [np.apply_along_axis(lambda v: np.convolve(v, np.ones(8) / 8, "same"), 1, zz) for zz in z]
    est.append(np.angle((z[2] * np.conj(z[0] * z[1]) * win).sum(1)))
psi = np.stack(est, 1); S = d["S"]; St = S[:, 1:].ravel()
xh = np.clip(np.tan(psi / 2), -4, 4).reshape(R, T, -1)
lab = hmm_regression(xh[:, :-1].reshape(-1, 4), xh[:, 1:].reshape(-1, 4), 4, [T - 1] * R)
print(f"known-source decoder -> AR-HMM on decoded hidden state x_hat: regime accuracy {best_perm_acc(St, lab, 4):.3f}")
xt = d["x"]; lab0 = hmm_regression(xt[:, :-1].reshape(-1, 4), xt[:, 1:].reshape(-1, 4), 4, [T - 1] * R)
print(f"reference: AR-HMM on the true hidden state x:                  {best_perm_acc(St, lab0, 4):.3f}")
