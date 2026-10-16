"""Regime-clustering detectors (labels per window, plus the posterior switch probability as a change score)."""
import numpy as np
from sklearn.decomposition import FastICA
from sklearn.mixture import GaussianMixture

from .base import Detector, register_detector, reduce, flat, persistence
from .hmm import hmm_regression, switch_probability

SCALINGS = ("whiten", "keep", "zscore")


def mixture_candidates(Z, K, k, seed, scalings=SCALINGS, seeds=3):
    """Gaussian mixtures on several preprocessings and initialisations: [(persistence, features, posteriors)], most
    persistent first.  Regimes are persistent by assumption, while partitions that isolate e.g. the windows straddling
    a switch flip constantly; ranking by persistence uses no labels."""
    R, out = Z.shape[0], []
    for sc in scalings:
        P = reduce(flat(Z), k, seed, scaling=sc)
        for s in range(seeds):
            post = GaussianMixture(K, n_init=2, random_state=seed + s, reg_covar=1e-3).fit(P).predict_proba(P)
            out.append((persistence(post.argmax(1), R), P, post))
    return sorted(out, key=lambda c: -c[0])


def persistent_mixture(Z, K, k, seed, scalings=SCALINGS, seeds=3):
    """(features, posteriors) of the most persistent mixture partition."""
    _, P, post = mixture_candidates(Z, K, k, seed, scalings, seeds)[0]
    return P, post


def persistent_hmm(Z, K, k, seed, select, restarts, autoregressive=False, top=3, return_switch=False):
    """Baum-Welch started from the `top` most persistent mixture partitions (only the best one when select="whiten");
    EM can drift towards higher-likelihood but non-persistent partitions, so the final labels are again chosen by
    persistence.  Returns labels and posteriors (for t >= 1 when autoregressive)."""
    R, T, _ = Z.shape
    cands = mixture_candidates(Z, K, k, seed, SCALINGS if select == "persistence" else ("whiten",), 3 if select == "persistence" else 1)
    best = None
    for _, P, post0 in cands[: top if select == "persistence" else 1]:
        init = post0.argmax(1)
        if autoregressive:
            P3 = P.reshape(R, T, -1)
            lab, post, switches = hmm_regression(flat(P3[:, :-1]), flat(P3[:, 1:]), K, [T - 1] * R, init=init.reshape(R, T)[:, 1:].ravel(),
                                       restarts=restarts, return_post=True, return_switch=True)
        else:
            lab, post, switches = hmm_regression(None, P, K, [T] * R, init=init, restarts=restarts, return_post=True, return_switch=True)
        score = persistence(lab, R)
        if best is None or score > best[0]:
            best = (score, lab, post, switches)
    return best[1:] if return_switch else (best[1], best[2])


def _pad_first(lab, R):
    lab = lab.reshape(R, -1)
    return np.concatenate([lab[:, :1], lab], 1)


@register_detector("gmm")
class GMM(Detector):
    """Gaussian mixture on PCA features (no temporal model).  select="persistence": the most persistent partition
    among several preprocessings and initialisations (label-free); select="whiten": one fit on whitened features."""
    defaults = dict(k=16, select="persistence")

    def detect(self, Z, K, seed=0):
        R, T, _ = Z.shape
        _, post = persistent_mixture(Z, K, self.k, seed, SCALINGS if self.select == "persistence" else ("whiten",), 3 if self.select == "persistence" else 1)
        return dict(labels=post.argmax(1).reshape(R, T), cp_score=switch_probability(post, [T] * R).reshape(R, T))


@register_detector("hmm")
class HMM(Detector):
    """Gaussian HMM on PCA features, started from the most persistent mixture partitions; most persistent result."""
    defaults = dict(k=16, restarts=1, select="persistence")       # restarts beyond the first are random inits

    def detect(self, Z, K, seed=0):
        R, T, _ = Z.shape
        lab, post, sw = persistent_hmm(Z, K, self.k, seed, self.select, self.restarts, return_switch=True)
        return dict(labels=lab.reshape(R, T), cp_score=sw.reshape(R, T))


@register_detector("arhmm")
class ARHMM(Detector):
    """Switching linear dynamics z_t ~ A_k z_{t-1} + b_k: for regimes that live in transitions.  Started and selected
    by persistence as HMM; restarts beyond the first are random initialisations."""
    defaults = dict(k=8, restarts=1, select="persistence")

    def detect(self, Z, K, seed=0):
        R, T, _ = Z.shape
        lab, post, sw = persistent_hmm(Z, K, self.k, seed, self.select, self.restarts, autoregressive=True, return_switch=True)
        sw = sw.reshape(R, T - 1)
        return dict(labels=_pad_first(lab, R), cp_score=np.concatenate([np.zeros((R, 1)), sw], 1))


def kurtosis(x):
    """Excess kurtosis of each column."""
    x = x - x.mean(0)
    return (x ** 4).mean(0) / ((x ** 2).mean(0) ** 2 + 1e-24) - 3


def window_contrasts(P, m):
    """P (R, T, d) -> (R, T-2m+1, d): mean of the m windows from t minus the mean of the m windows before t."""
    cs = np.concatenate([np.zeros((P.shape[0], 1, P.shape[2])), np.cumsum(P, 1)], 1)
    t = np.arange(m, P.shape[1] - m + 1)
    return (cs[:, t + m] - 2 * cs[:, t] + cs[:, t - m]) / m


def phase_surrogate(P, rng):
    """Same power spectrum (and cross-spectra) per series, random Fourier phases shared across dimensions: linear
    Gaussian dynamics with the second-order structure of P and none of its non-Gaussian temporal structure."""
    Fz = np.fft.rfft(P, axis=1)
    ph = np.exp(2j * np.pi * rng.random((P.shape[0], Fz.shape[1], 1))); ph[:, 0] = 1
    return np.fft.irfft(Fz * ph, n=P.shape[1], axis=1)


def _whitener(X):
    w, V = np.linalg.eigh(np.atleast_2d(np.cov(X.T))); w = np.maximum(w, 1e-8 * w.max())
    return V @ np.diag(w ** -0.5) @ V.T


def contrast_components(P, ms, seed):
    """Unit-norm directions (d, len(ms) * d): FastICA on the whitened window contrasts at each scale m."""
    F = []
    for m in [m for m in ms if P.shape[1] > 2 * m]:
        D = flat(window_contrasts(P, m)); D = D - D.mean(0); Wd = _whitener(D)
        ica = FastICA(whiten=False, fun="cube", max_iter=400, tol=1e-4, random_state=seed).fit(D @ Wd)
        F.append(Wd @ ica.components_.T)
    F = np.concatenate(F, 1)
    return F / np.linalg.norm(F, axis=0)


def temporal_excess(P, F, ms=(1,)):
    """max over scales m of kurtosis(window contrast at m) minus |marginal kurtosis| along each direction: large for
    piecewise-constant (regime-like) directions, ~0 for Gaussian ones, negative for jump-like nuisances that are
    heavy-tailed in both.  Averaging m windows on each side divides white within-regime noise by m and keeps the jumps,
    so larger m reveal regime directions whose jumps are small next to the within-regime spread."""
    ms = [m for m in ms if P.shape[1] > 2 * m]
    ck = np.max([kurtosis(flat(window_contrasts(P, m)) @ F) for m in ms], 0)
    return ck - np.abs(kurtosis(flat(P) @ F))


def distinct(F, order, n=None, above=None, excess=None, cos=0.8):
    """Directions in `order`, skipping near-duplicates (|cos| >= cos), until n are chosen or the excess drops to `above`."""
    sel = []
    for j in order:
        if above is not None and excess[j] <= above:
            break
        if all(abs(F[:, j] @ F[:, i]) < cos for i in sel):
            sel.append(j)
        if n is not None and len(sel) == n:
            break
    return sel


@register_detector("temporal_excess")
class TemporalExcess(Detector):
    """Regime directions as the components of the window contrasts that are heavy-tailed in time but not in the
    marginal (regime_readout/temporal_excess_method.html, contrast_rank2.py).
    - candidates: FastICA on the whitened contrasts at each scale in `ms`;
    - ranking: temporal excess = max over m in `ms` of kurtosis(contrast at m) - |marginal kurtosis|;
    - how many directions: n="K-1" (the top K-1 distinct), n="null" (every distinct one whose excess beats the largest
      excess of a phase surrogate of the features: same spectra, no non-Gaussian dynamics; label-free), or an integer;
    - readout: GMM then HMM on the chosen directions (constant labels if none is found).
    The decisions also carry n_dirs, the top excesses and the null threshold (diagnostics)."""
    defaults = dict(k=32, ms=[1, 2, 4, 8], n="null", restarts=2)

    def init(self):
        if not (self.n in ("K-1", "null") or isinstance(self.n, int)):
            raise ValueError(f"temporal_excess: n must be 'K-1', 'null' or an integer, got {self.n!r}")

    def detect(self, Z, K, seed=0):
        R, T, _ = Z.shape
        P = reduce(flat(Z), self.k, seed).reshape(R, T, -1)
        F = contrast_components(P, self.ms, seed)
        excess = temporal_excess(P, F, self.ms); order = np.argsort(-excess)
        thr = None
        if self.n == "null":
            Ps = phase_surrogate(P, np.random.default_rng(seed))
            thr = float(temporal_excess(Ps, contrast_components(Ps, self.ms, seed), self.ms).max())
            sel = distinct(F, order, above=thr, excess=excess)
        else:
            sel = distinct(F, order, n=K - 1 if self.n == "K-1" else self.n)
        info = dict(n_dirs=len(sel), excess=np.round(excess[order[:8]], 2).tolist(), null=thr)
        if not sel:
            return dict(labels=np.zeros((R, T), int), cp_score=np.zeros((R, T)), **info)
        C = flat(P) @ F[:, sel]
        Q = (C - C.mean(0)) / C.std(0)
        init = GaussianMixture(K, n_init=3, random_state=seed).fit(Q).predict(Q)
        lab, post, switches = hmm_regression(None, Q, K, [T] * R, init=init, restarts=self.restarts, return_post=True, return_switch=True)
        return dict(labels=lab.reshape(R, T), cp_score=switches.reshape(R, T), **info)
