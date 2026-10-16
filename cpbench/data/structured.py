"""Structured sources: the two constructions that are not "linear source + transform".

phase        oscillator triplets with f_c = f_a + f_b; the regime sets the phase offset theta_c - theta_a - theta_b.
             Spectra, means and covariances are regime-independent; only third-order (bispectral) structure changes.
transition   a hidden state h is rotated by a regime-dependent angle once every `step` samples,
             h_w = rho Rot(alpha_{S_w}) h_{w-1} + sqrt(1 - rho^2) eps, so h ~ N(0, I) in every regime and a single step
             window carries no regime information; the regime lives only in transitions.  coding="linear" shows h
             directly, coding="phase" hides it in cross-frequency phase coupling (offset 2 arctan h_i).  `direction`:
             "asym" (one-directional angles, irreversible dynamics) or "sym" (closed under reversal).  Switches fall on
             the step grid (with a random phase in the observed series); the hidden state is exposed as oracle "hidden".
reversal     the regime sets the direction in which each channel's phase runs: x = s(phi), phi_t = phi_{t-1} +- 2 pi f +
             phase diffusion, with an odd-harmonic, time-asymmetric waveform s.  Reversing time leaves the marginal and
             every second-order statistic unchanged and odd harmonics make all odd moments vanish, so the regimes differ
             only in fourth- and higher-order temporal structure (invisible to mean / variance / spectral / covariance /
             bispectral detectors on the raw series).
gauss_switch a latent that is exactly N(0, I) at every time but jumps between regimes (z_t and z_{t+tau} are not jointly
             Gaussian, so increments have excess kurtosis); see the class docstring for the two codings.
"""
import numpy as np
from scipy.stats import norm

from .params import spread, per_sample, resolve
from .registry import register_source
from .sources import Source, slow_ar1

TRIPLETS = ((0.050, 0.080), (0.060, 0.110), (0.070, 0.150), (0.090, 0.190), (0.040, 0.230), (0.100, 0.170))
FREE = (0.035, 0.100, 0.240, 0.310, 0.370, 0.420)


def oscillator_bank(R, n, triplets, free, psi, rng, diff=0.06):
    """psi: (R, n, len(triplets)) phase offsets.  Returns (R, n, 3*len(triplets) + len(free)) unit-variance signals.
    Phases follow 2 pi f t + random walk (variance `diff` per sample), so they decorrelate within ~1/diff samples."""
    tau = np.arange(n)
    def walk():
        return np.cumsum(np.sqrt(diff) * rng.standard_normal((R, n)), 1) + rng.uniform(0, 2 * np.pi, (R, 1))
    sig = []
    for i, (fa, fb) in enumerate(triplets):
        ta = 2 * np.pi * fa * tau + walk(); tb = 2 * np.pi * fb * tau + walk()
        sig += [np.cos(ta), np.cos(tb), np.cos(ta + tb + psi[..., i])]
    for f in free:
        sig.append(np.cos(2 * np.pi * f * tau + walk()))
    return np.sqrt(2) * np.stack(sig, -1)




@register_source("phase")
class PhaseCoupling(Source):
    defaults = dict(effect=1.0, triplets=4, free=4, diff=0.06)

    @property
    def d(self):
        return 3 * self.triplets + self.free

    def setup(self):
        sp = spread(self.K) * (self.K - 1) / self.K                        # distinct on the circle
        self.psi = self.effect * np.pi * np.stack([self.prng.permutation(sp) for _ in range(self.triplets)], 1)

    def extra_driven(self):
        return self.effect != 0 and self.K > 1

    def simulate(self, ctx):
        R, n = ctx.S.shape
        return oscillator_bank(R, n, TRIPLETS[: self.triplets], FREE[: self.free], per_sample(self.psi, ctx.S, ctx.W), ctx.rng, self.diff)


def transition_angles(K, direction):
    j = np.arange(K)
    if direction == "asym":
        return (2 * j + 1) * np.pi / (2 * K)                              # all positive: one-directional
    half = (2 * np.arange((K + 1) // 2) + 1) * np.pi / (2 * K)
    return np.concatenate([half, -half])[:K]                               # closed under reversal



@register_source("transition")
class Transition(Source):
    defaults = dict(effect=1.0, step=32, planes=2, rho=0.95, coding="phase", direction="asym", free=4, diff=0.06)

    def init(self):
        super().init()
        self.grid = self.step

    @property
    def d(self):
        return 2 * self.planes if self.coding == "linear" else 3 * 2 * self.planes + self.free

    def setup(self):
        self.alpha = self.effect * transition_angles(self.K, self.direction)

    def extra_driven(self):
        return self.effect != 0 and self.K > 1

    def hidden_state(self, S_w, rng):
        R, T = S_w.shape
        h = np.empty((R, T, 2 * self.planes)); cur = rng.standard_normal((R, 2 * self.planes))
        for t in range(T):
            c, s = np.cos(self.alpha[S_w[:, t]]), np.sin(self.alpha[S_w[:, t]])
            new = np.empty_like(cur)
            for p in range(self.planes):
                u, v = cur[:, 2 * p], cur[:, 2 * p + 1]
                new[:, 2 * p], new[:, 2 * p + 1] = c * u - s * v, s * u + c * v
            cur = self.rho * new + np.sqrt(1 - self.rho ** 2) * rng.standard_normal(cur.shape)
            h[:, t] = cur
        return h

    def simulate(self, ctx):
        R, n = ctx.S.shape
        hs = np.repeat(self.hidden_state(ctx.S[:, :: self.step], ctx.rng), self.step, axis=1)[:, :n]
        if self.coding == "linear":
            x = (hs + 0.5 * slow_ar1(R, n, 2 * self.planes, 0.8, ctx.rng)) / np.sqrt(1.25)
        else:
            x = oscillator_bank(R, n, TRIPLETS[: 2 * self.planes], FREE[: self.free], 2 * np.arctan(hs), ctx.rng, self.diff)
        self.oracle = {"hidden": hs}
        return x


@register_source("reversal")
class Reversal(Source):
    """d independent channels x_c = s(phi_c) / sd, s(phi) = sin(phi) + b cos(3 phi); phi_c advances by
    direction[k, c] * 2 pi freq_c per sample (direction +-1 per regime and channel) plus a random walk of variance
    `diff`.  The time reversal of a channel is the same process with the direction flipped, so every statistic that is
    invariant under reversal (marginal law, autocovariance at every lag) is the same in all regimes; channels are
    independent, so cross-covariances are zero; s(phi + pi) = -s(phi) makes the law symmetric, so all odd moments
    (bispectrum, coskewness) vanish.  What changes is the waveform's direction: rising edges become falling ones.
    direction=None: regime k uses the binary digits of k (needs 2^d >= K)."""
    defaults = dict(d=2, freq=[0.031, 0.047], b=0.5, diff=0.002, direction=None)

    def setup(self):
        if self.direction is None:
            if 2 ** self.d < self.K:
                raise ValueError(f"reversal: {self.d} channels give only {2 ** self.d} direction patterns for K={self.K}")
            bits = (np.arange(self.K)[:, None] >> np.arange(self.d)) & 1
            self.dir = 1.0 - 2.0 * bits
        else:
            self.dir = np.sign(resolve(self.direction, self.K, self.d, self.prng))
        self.f = np.broadcast_to(np.asarray(self.freq, float), (self.d,))
        self.sd = np.sqrt(0.5 * (1 + self.b ** 2))

    def extra_driven(self):
        return self.K > 1 and bool(np.ptp(self.dir, axis=0).max() > 0)

    def simulate(self, ctx):
        R, n = ctx.S.shape
        step = per_sample(self.dir, ctx.S, ctx.W) * 2 * np.pi * self.f + np.sqrt(self.diff) * ctx.rng.standard_normal((R, n, self.d))
        phi = np.cumsum(step, 1) + ctx.rng.uniform(0, 2 * np.pi, (R, 1, self.d))
        return (np.sin(phi) + self.b * np.cos(3 * phi)) / self.sd


@register_source("gauss_switch")
class GaussSwitch(Source):
    """A d-dimensional latent with z_t ~ N(0, I) exactly at every t, but jump-like dynamics at regime switches.
    e_t: independent unit-variance AR(1) coordinates with correlation time `timescale` (0: white).
    coding="fresh":  z_t = sqrt(c) m_j + sqrt(1 - c) e_t, with m_j ~ N(0, I) drawn afresh for each regime segment j.
                     Every direction jumps at every switch; the regime label itself is not encoded (change points
                     only).  Increment excess kurtosis along any unit u (geometric dwell, switch hazard h = 1/dwell):
                         kappa_tau = 3 p (1 - p) a^2 / (p a + g)^2,   p = 1 - (1 - h)^tau,  a = 2c,
                         g = 2 (1 - c)(1 - rho^tau),  rho = exp(-1 / timescale)
                     and for white e (g = 2(1 - c)) it peaks at tau* = ln(2 - c) / -ln(1 - h) ~ dwell ln(2 - c) with
                     height 3 c^2 / (4 (1 - c)).
    coding="bands":  along one random unit direction u, z.u = Phi^-1((pi(S_t) + Phi(e_t)) / K): regime k occupies the
                     k-th quantile band of N(0, 1) (pi a random order of the bands), the other d - 1 directions are e.
                     With uniform stationary regimes z.u ~ N(0, 1) exactly and independent of the rest, so z ~ N(0, I);
                     the K regimes are identifiable along u = Q[:, 0].
    Use with standardize: false to keep the marginal exact."""
    defaults = dict(d=8, coding="bands", c=0.8, timescale=0.0)

    def init(self):
        super().init()
        if self.coding not in ("fresh", "bands"):
            raise ValueError(f"gauss_switch: coding must be 'fresh' or 'bands', got {self.coding!r}")

    def setup(self):
        Q, _ = np.linalg.qr(self.prng.standard_normal((self.d, self.d)))
        self.Q = Q                                                       # u = Q[:, 0]
        self.order = self.prng.permutation(self.K)

    def extra_driven(self):
        return self.K > 1

    def noise(self, R, n, ctx):
        if not self.timescale:
            return ctx.rng.standard_normal((R, n, self.d))
        return slow_ar1(R, n, self.d, np.exp(-1.0 / self.timescale), ctx.rng)

    def simulate(self, ctx):
        R, n = ctx.S.shape
        e = self.noise(R, n, ctx)
        if self.coding == "fresh":
            seg = np.cumsum(np.c_[np.zeros((R, 1), int), np.diff(ctx.S, axis=1) != 0], 1)
            m = ctx.rng.standard_normal((R, seg.max() + 1, self.d))
            z = np.sqrt(self.c) * np.take_along_axis(m, seg[..., None], 1) + np.sqrt(1 - self.c) * e
        else:
            u = (self.order[ctx.S] + norm.cdf(e[..., 0])) / self.K
            z = np.concatenate([norm.ppf(np.clip(u, 1e-12, 1 - 1e-12))[..., None], e[..., 1:]], -1) @ self.Q.T
        return z
