"""Raw-window feature functions: windows (N, W, D) -> (N, F)."""
import numpy as np
from sklearn.decomposition import PCA


def flat(o):
    return o.reshape(len(o), -1)


def mean(o):
    return o.mean(1)


def stats(o):
    mu, sd = o.mean(1), o.std(1) + 1e-6
    ac = ((o[:, 1:] - mu[:, None]) * (o[:, :-1] - mu[:, None])).mean(1) / sd ** 2
    return np.concatenate([mu, np.log(sd), ac], 1)


def spectra(o):
    X = np.fft.rfft(o - o.mean(1, keepdims=True), axis=1)
    return np.log(np.abs(X[:, 1:]) ** 2 + 1e-3).reshape(len(o), -1)


def cov(o):
    oc = o - o.mean(1, keepdims=True)
    C = np.einsum("nwi,nwj->nij", oc, oc) / o.shape[1]
    iu = np.triu_indices(o.shape[2])
    return C[:, iu[0], iu[1]]


def lagcov(o, lags=(1, 2, 4)):
    """Antisymmetric part of lagged cross-covariances: sensitive to which channel leads."""
    oc = o - o.mean(1, keepdims=True); iu = np.triu_indices(o.shape[2], 1); out = []
    for L in lags:
        C = np.einsum("nwi,nwj->nij", oc[:, L:], oc[:, :-L]) / (o.shape[1] - L)
        A = C - C.transpose(0, 2, 1)
        out.append(A[:, iu[0], iu[1]])
    return np.concatenate(out, 1)


def coskew(o):
    """Third-order co-moments E[x_i^2 x_j] of standardised channels (odd moments: sign-sensitive)."""
    oc = o - o.mean(1, keepdims=True); oc = oc / (oc.std(1, keepdims=True) + 1e-6)
    return (np.einsum("nwi,nwj->nij", oc ** 2, oc) / o.shape[1]).reshape(len(o), -1)


def bispec(o):
    X = np.fft.rfft(o - o.mean(1, keepdims=True), axis=1)
    nb = X.shape[1] - 1
    Xn = X / (np.sqrt((np.abs(X[:, 1:]) ** 2).mean((1, 2)))[:, None, None] + 1e-9)
    pairs = [(a, b) for a in range(1, nb) for b in range(a, nb) if a + b <= nb]
    B = np.stack([(Xn[:, a] * Xn[:, b] * np.conj(Xn[:, a + b])).sum(-1) for a, b in pairs], 1)
    return np.concatenate([B.real, B.imag], 1)


def asym(o, lags=(1, 2, 4, 8)):
    """Fourth-order time asymmetry E[x_t^3 x_{t+l} - x_t x_{t+l}^3] per channel and lag (standardised windows): zero
    for any time-reversible process, flips sign under time reversal; blind to every second- and third-order statistic."""
    oc = o - o.mean(1, keepdims=True); oc = oc / (oc.std(1, keepdims=True) + 1e-6)
    return np.concatenate([(oc[:, :-l] ** 3 * oc[:, l:] - oc[:, :-l] * oc[:, l:] ** 3).mean(1) for l in lags], 1)


RAW = dict(flat=flat, mean=mean, stats=stats, spectra=spectra, cov=cov, lagcov=lagcov, coskew=coskew, bispec=bispec, asym=asym)
