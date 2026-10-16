"""Change-point detectors from Dynamic Mode Decomposition (DMD) reconstruction error.

Both ask whether a small DMD model, refitted on each short stretch of the feature sequence, reconstructs that stretch;
a change shows up as reconstruction failure.  The observations z_t are the window features (PCA-reduced), so on raw
data use e.g. raw_flat with window=1 to run on the samples themselves.

    dmd_error   the 2019 diagnostic (arxiv 1904.09082): the error of a window centred on t, averaged over ranks
    cpdmd       CPDMD (arxiv 2405.15576): delay-embedded DMD on the trailing window, EWMA of the error increments and
                a sequential control chart with restart after each alarm
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

from .base import register_detector, reduce
from .changepoint import ChangePoint


def dmd_errors(X, d=1, r=4):
    """X (B, w, p) windows -> (B,) mean squared reconstruction error of rank-r exact DMD on the delay embedding with
    d delays (Hankel matrix of d*p rows and w-d+1 columns).  The amplitudes are fitted to the first column and the
    window is recovered from the first row and the last column of each channel's reconstructed block."""
    B, w, p = X.shape
    H = sliding_window_view(X, d, axis=1)                                     # (B, w-d+1, p, d)
    H = H.transpose(0, 2, 3, 1).reshape(B, p * d, w - d + 1)
    Zm, Zp = H[..., :-1], H[..., 1:]
    U, s, Vh = np.linalg.svd(Zm, full_matrices=False)
    r = min(r, s.shape[1])
    U, s, V = U[..., :r], s[:, :r], Vh[:, :r].transpose(0, 2, 1)
    inv = np.where(s > 1e-10 * s[:, :1], 1.0 / np.maximum(s, 1e-300), 0.0)
    ZpVS = Zp @ V * inv[:, None, :]                                           # (B, dp, r)
    lam, Wv = np.linalg.eig(U.transpose(0, 2, 1) @ ZpVS)                      # reduced operator A~ = U* Z+ V S^-1
    Phi = ZpVS @ Wv                                                           # exact DMD modes
    b = np.linalg.pinv(Phi) @ H[..., :1].astype(complex)                      # (B, r, 1)
    with np.errstate(over="ignore", invalid="ignore"):
        powers = lam[..., None] ** np.arange(H.shape[-1])                    # (B, r, m)
        Hh = np.nan_to_num((Phi @ (b * powers)).real, nan=1e12, posinf=1e12, neginf=-1e12)
    Hh = Hh.reshape(B, p, d, -1)
    Xh = np.concatenate([Hh[:, :, 0, :], Hh[:, :, 1:, -1]], -1).transpose(0, 2, 1)   # (B, w, p)
    return np.minimum(((X - Xh) ** 2).mean((1, 2)), 1e12)


def window_errors(P, w, d, r, chunk=4096):
    """P (T, p) -> (T - w + 1,) errors of the windows [t, t + w)."""
    win = sliding_window_view(P, w, axis=0).transpose(0, 2, 1)                # (T-w+1, w, p)
    return np.concatenate([dmd_errors(win[i:i + chunk], d, r) for i in range(0, len(win), chunk)])


@register_detector("dmd_error")
class DMDError(ChangePoint):
    """Reconstruction error of DMD fitted to the m features centred on t (so the window straddles a change between
    t-1 and t), for each rank in `ranks`; each rank's error is divided by its median and the logs are averaged."""
    defaults = dict(m=16, ranks=[2, 4, 8], d=1, k=8, scaling="keep")

    def detect(self, Z, K=None, seed=0):
        R, T, _ = Z.shape; m, h = self.m, self.m // 2
        P = reduce(Z.reshape(-1, Z.shape[-1]), self.k, seed, self.scaling).reshape(R, T, -1)
        out = np.full((R, T), -np.inf)
        if T < m:
            return dict(labels=None, cp_score=out)
        logs = []
        for rank in self.ranks:
            E = np.stack([window_errors(P[i], m, self.d, rank) for i in range(R)])   # window [t-h, t-h+m) at t
            logs.append(np.log(E + 1e-12) - np.log(np.median(E) + 1e-12))
        out[:, h:h + T - m + 1] = np.mean(logs, 0)
        return dict(labels=None, cp_score=out)


@register_detector("cpdmd")
class CPDMD(ChangePoint):
    """CPDMD: rank-r DMD with d delays on the trailing w features; eps_t = reconstruction error, S_t = EWMA (alpha)
    of eps_t - eps_{t-1}.  After `burn` values of S the chart monitors |S_t - mu| / sd (mu, sd: running mean and
    standard deviation of S since the segment start, updated with each non-alarm value); the score is that ratio, an
    alarm is raised above L and the procedure restarts on the data after it.  tune=True picks (w, d, r) from `grid`
    on each segment's burn-in by minimal mean error.  The paper's adaptive (Huber) EWMA is a plain EWMA here."""
    defaults = dict(w=20, d=5, r=4, alpha=0.1, L=3.0, burn=50, k=8, scaling="keep", tune=False,
                    grid=dict(w=[10, 20, 40], d=[2, 5, 10], r=[2, 4, 8]))

    def _params(self, P, s):
        if not self.tune:
            return self.w, self.d, self.r
        best, arg = np.inf, None
        for w in self.grid["w"]:
            seg = P[s:s + self.burn + w]
            for d in self.grid["d"]:
                for r in self.grid["r"]:
                    if d < w and len(seg) >= w:
                        e = window_errors(seg, w, d, r).mean()
                        if e < best:
                            best, arg = e, (w, d, r)
        return arg or (self.w, self.d, self.r)

    def _series(self, P):
        T = len(P); score = np.zeros(T); alarms = []
        s = 0
        while s < T:
            w, d, r = self._params(P, s)
            if T - s < w + 1:
                break
            eps = window_errors(P[s:], w, d, r)                               # eps[i]: window ending at s + w - 1 + i
            S, n, mu, M2, nxt = 0.0, 0, 0.0, 0.0, None
            for i in range(1, len(eps)):
                t = s + w - 1 + i
                S = (1 - self.alpha) * S + self.alpha * (eps[i] - eps[i - 1])
                if n >= self.burn:
                    z = abs(S - mu) / (np.sqrt(M2 / (n - 1)) + 1e-12)
                    score[t] = z
                    if z > self.L:
                        alarms.append(t); nxt = t + 1
                        break
                n += 1; delta = S - mu; mu += delta / n; M2 += delta * (S - mu)
            if nxt is None:
                break
            s = nxt
        return score, alarms

    def detect(self, Z, K=None, seed=0):
        R, T, _ = Z.shape
        P = reduce(Z.reshape(-1, Z.shape[-1]), self.k, seed, self.scaling).reshape(R, T, -1)
        res = [self._series(P[i]) for i in range(R)]
        return dict(labels=None, cp_score=np.stack([sc for sc, _ in res]), alarms=[a for _, a in res])
