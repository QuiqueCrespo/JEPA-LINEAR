"""Change-point detectors (a change score per window, no labels)."""
import numpy as np
from scipy.special import gammaln

from .base import Detector, register_detector, reduce


class ChangePoint(Detector):
    kind = "cp"


@register_detector("contrast")
class Contrast(ChangePoint):
    """Mahalanobis distance between the means of the m windows before and after t (whitened PCA space)."""
    defaults = dict(m=2, k=16)

    def detect(self, Z, K=None, seed=0):
        R, T, _ = Z.shape; m = self.m
        P = reduce(Z.reshape(-1, Z.shape[-1]), self.k, seed).reshape(R, T, -1)
        cs = np.concatenate([np.zeros((R, 1, P.shape[-1])), np.cumsum(P, 1)], 1)
        out = np.full((R, T), -np.inf)
        t = np.arange(m, T - m + 1)
        after = (cs[:, t + m] - cs[:, t]) / m; before = (cs[:, t] - cs[:, t - m]) / m
        out[:, t] = ((after - before) ** 2).sum(-1)
        return dict(labels=None, cp_score=out)


@register_detector("mmd")
class MMD(ChangePoint):
    """Kernel two-sample statistic (MMD^2, RBF with median bandwidth) between the m windows before and after t."""
    defaults = dict(m=4, k=16)

    def detect(self, Z, K=None, seed=0):
        R, T, _ = Z.shape; m = self.m
        P = reduce(Z.reshape(-1, Z.shape[-1]), self.k, seed).reshape(R, T, -1)
        sub = P.reshape(-1, P.shape[-1])[np.random.default_rng(seed).choice(R * T, min(2000, R * T), replace=False)]
        d2 = ((sub[:, None] - sub[None]) ** 2).sum(-1); bw = np.median(d2[d2 > 0])
        out = np.full((R, T), -np.inf)
        for r in range(R):
            G = np.exp(-((P[r][:, None] - P[r][None]) ** 2).sum(-1) / bw)       # (T, T)
            for t in range(m, T - m + 1):
                a, b = slice(t - m, t), slice(t, t + m)
                out[r, t] = G[a, a].mean() + G[b, b].mean() - 2 * G[a, b].mean()
        return dict(labels=None, cp_score=out)


@register_detector("bocpd")
class BOCPD(ChangePoint):
    """Bayesian online change-point detection (Adams & MacKay 2007) with independent Normal-Gamma models on the top
    k whitened PCA dimensions.  After window t, run index j holds the last j windows, so a change at window c is the
    run index delay + 1 at time c + delay: the score for c is P(run started at c | x_1..x_{c+delay})."""
    defaults = dict(k=4, hazard=0.1, max_run=200, delay=2)

    def detect(self, Z, K=None, seed=0):
        R, T, _ = Z.shape; delay = self.delay
        P = reduce(Z.reshape(-1, Z.shape[-1]), self.k, seed).reshape(R, T, -1); k = P.shape[-1]
        out = np.zeros((R, T))
        mu0, kap0, a0, b0 = 0.0, 1.0, 1.0, 1.0
        for r in range(R):
            x = P[r]
            logR = np.array([0.0]); mu = np.full((1, k), mu0); kap = np.full((1, 1), kap0)
            a = np.full((1, 1), a0); b = np.full((1, k), b0)
            for t in range(T):
                scale2 = b * (kap + 1) / (a * kap)                       # Student-t predictive per run length
                lp = (gammaln(a + 0.5) - gammaln(a) - 0.5 * np.log(np.pi * 2 * a * scale2)
                      - (a + 0.5) * np.log1p((x[t] - mu) ** 2 / (2 * a * scale2))).sum(1)
                growth = logR + lp + np.log(1 - self.hazard)
                cp = np.logaddexp.reduce(logR + lp + np.log(self.hazard))
                logR = np.r_[cp, growth]; logR -= np.logaddexp.reduce(logR)
                if t >= delay and len(logR) > delay + 1:
                    out[r, t - delay] = np.exp(logR[delay + 1])
                mu_n = np.vstack([np.full((1, k), mu0), (kap * mu + x[t]) / (kap + 1)])
                b_n = np.vstack([np.full((1, k), b0), b + kap * (x[t] - mu) ** 2 / (2 * (kap + 1))])
                kap = np.vstack([[[kap0]], kap + 1]); a = np.vstack([[[a0]], a + 0.5]); mu, b = mu_n, b_n
                if len(logR) > self.max_run:
                    logR, mu, kap, a, b = logR[:self.max_run], mu[:self.max_run], kap[:self.max_run], a[:self.max_run], b[:self.max_run]
        return dict(labels=None, cp_score=out)
