"""Regime processes: sample(R, n, rng) -> S (R, n) integer regime per sample.  All timescales in samples.

`grid` (samples, optional) restricts switches to multiples of `grid`; a source whose dynamics are defined per step
(tier E) requires it and the Dataset sets it.  The Dataset crops every draw at a random offset per series, so the grid
has a random phase relative to the observed series and to any model window."""
import numpy as np
from .registry import Component, register_regime


def _segment_length(dist, mean, rng):
    if dist == "geometric":
        return int(rng.geometric(1.0 / mean))
    if dist == "fixed":
        return int(round(mean))
    if dist == "pareto":                      # heavy tail, alpha = 1.5, scaled to the requested mean
        a = 1.5
        return max(1, int(round(mean * (a - 1) / a * (1 + rng.pareto(a)))))
    raise ValueError(f"unknown dwell distribution {dist!r}")


def regime_path(R, n, K, dwell, rng, dist="geometric", transition="uniform", unit=1):
    """(R, n) semi-Markov regime path: segments of mean `dwell` samples, whole multiples of `unit` samples; the next
    regime is uniform among the others ("uniform") or k -> k+1 ("cyclic", irreversible)."""
    S = np.empty((R, n), int)
    mean = max(dwell / unit, 1.0)
    for r in range(R):
        k = int(rng.integers(K))
        L = _segment_length(dist, mean, rng)
        if dist != "geometric":                                      # stationary residual of the first segment
            L = int(rng.integers(1, max(L, 1) + 1))
        pos = 0
        while pos < n:
            span = max(1, L) * unit
            S[r, pos:pos + span] = k
            pos += span
            k = (k + 1) % K if transition == "cyclic" else int((k + rng.integers(1, K)) % K)
            L = _segment_length(dist, mean, rng)
    return S


class RegimeProcess(Component):
    """`ramp` (samples): parameters move linearly between regimes over this many samples around each switch
    (gradual change points); 0 = abrupt.  Labels stay hard (the switch is the middle of the ramp)."""
    defaults = dict(K=1, ramp=0)
    grid = None


@register_regime("markov")
class Markov(RegimeProcess):
    """Semi-Markov regimes: segment lengths (mean `dwell` samples) from `dist` in {geometric, fixed, pareto};
    `transition` uniform or cyclic (irreversible)."""
    defaults = dict(K=4, dwell=320, dist="geometric", transition="uniform", grid=None)

    def sample(self, R, n, rng):
        return regime_path(R, n, self.K, self.dwell, rng, self.dist, self.transition, unit=int(self.grid or 1))


@register_regime("scripted")
class Scripted(RegimeProcess):
    """Fixed sequence of [regime, length] segments, repeated to fill n samples (debugging, demos)."""
    defaults = dict(segments=[[0, 300], [1, 300]], K=None)

    def init(self):
        self.segments = [list(s) for s in self.segments]
        self.K = self.params["K"] = self.K or max(k for k, _ in self.segments) + 1

    def sample(self, R, n, rng):
        row = np.concatenate([np.full(L, k) for k, L in self.segments])
        return np.tile(np.tile(row, -(-n // len(row)))[:n], (R, 1))


@register_regime("single_cp")
class SingleChange(RegimeProcess):
    """One change per series at a uniform position in [margin, 1 - margin) of the series, between two regimes."""
    defaults = dict(K=2, margin=0.2)

    def sample(self, R, n, rng):
        S = np.empty((R, n), int)
        for r in range(R):
            a = int(rng.integers(self.K)); b = int((a + rng.integers(1, self.K)) % self.K)
            c = int(rng.integers(int(self.margin * n), int((1 - self.margin) * n)))
            S[r, :c], S[r, c:] = a, b
        return S


@register_regime("none")
class NoRegime(RegimeProcess):
    defaults = dict(K=1)

    def sample(self, R, n, rng):
        return np.zeros((R, n), int)
