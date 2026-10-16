"""HMMs with Gaussian / linear-regression emissions (from regime_large_scale.ipynb, unchanged maths), returning
posteriors as well so that change-point scores can be read from them."""
import numpy as np


def forward_backward(logB, A, pi, lengths, return_switch=False):
    """Scaled forward-backward over several sequences. Returns posteriors, expected transition counts, log-lik."""
    N, K = logB.shape
    switches = np.zeros(N)
    gamma, xi, ll, start = np.zeros((N, K)), np.zeros((K, K)), 0.0, 0
    for L in lengths:
        lb = logB[start: start + L]
        m = lb.max(1, keepdims=True)
        B = np.exp(lb - m)
        alpha, c = np.zeros((L, K)), np.zeros(L)
        a = pi * B[0]; c[0] = a.sum() + 1e-300; alpha[0] = a / c[0]
        for t in range(1, L):
            a = (alpha[t - 1] @ A) * B[t]; c[t] = a.sum() + 1e-300; alpha[t] = a / c[t]
        beta = np.ones((L, K))
        for t in range(L - 2, -1, -1):
            beta[t] = (A @ (B[t + 1] * beta[t + 1])) / c[t + 1]
        g = alpha * beta
        gamma[start: start + L] = g / g.sum(1, keepdims=True)
        xi += A * (alpha[:-1].T @ (B[1:] * beta[1:] / c[1:, None]))
        if return_switch:
            future = B[1:] * beta[1:]
            norm = ((alpha[:-1] @ A) * future).sum(1)
            same = (alpha[:-1] * np.diag(A) * future).sum(1)
            switches[start + 1:start + L] = np.clip(1 - same / np.maximum(norm, 1e-300), 0, 1)
        ll += np.log(c).sum() + m.sum()
        start += L
    return (gamma, xi, ll, switches) if return_switch else (gamma, xi, ll)


def hmm_regression(X, Y, K, lengths, iters=40, restarts=3, seed=0, init=None, return_post=False, return_switch=False):
    """HMM whose hidden state k emits Y_t ~ N(A_k [X_t, 1], Sigma_k), fitted by Baum-Welch.
    X=None: Gaussian HMM.  X=z_{t-1}, Y=z_t: autoregressive HMM (switching linear dynamics)."""
    rng = np.random.default_rng(seed)
    N, dy = Y.shape
    Xd = np.ones((N, 1)) if X is None else np.c_[X, np.ones(N)]
    best, best_ll = None, -np.inf
    for r in range(restarts):
        R = np.eye(K)[init] * 0.8 + 0.2 / K if (r == 0 and init is not None) else rng.dirichlet(np.ones(K), N)
        A = np.full((K, K), 0.1 / max(K - 1, 1)); np.fill_diagonal(A, 0.9)
        pi = np.full(K, 1 / K)
        for it in range(iters):
            logB = np.zeros((N, K))
            for k in range(K):
                w = R[:, k] + 1e-9
                Ak = np.linalg.solve(Xd.T @ (Xd * w[:, None]) + 1e-6 * np.eye(Xd.shape[1]), Xd.T @ (Y * w[:, None]))
                E = Y - Xd @ Ak
                Sig = (E * w[:, None]).T @ E / w.sum() + 1e-4 * np.eye(dy)
                logB[:, k] = -0.5 * np.linalg.slogdet(Sig)[1] - 0.5 * np.einsum("ni,ij,nj->n", E, np.linalg.inv(Sig), E)
            R, xi, ll, switches = forward_backward(logB, A, pi, lengths, return_switch=True)
            A = (xi + 1e-12) / (xi.sum(1, keepdims=True) + K * 1e-12)
        if ll > best_ll:
            best, best_ll, best_switches = R.copy(), ll, switches.copy()
    if return_switch:
        return best.argmax(1), best, best_switches
    return (best.argmax(1), best) if return_post else best.argmax(1)


def switch_probability(post, lengths):
    """Independent-marginal change score (e.g. GMM), not a calibrated HMM transition probability."""
    out = np.zeros(len(post)); start = 0
    for L in lengths:
        g = post[start: start + L]
        out[start + 1: start + L] = 1 - (g[1:] * g[:-1]).sum(1)
        start += L
    return out
