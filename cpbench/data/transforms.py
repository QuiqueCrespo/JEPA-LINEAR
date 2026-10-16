"""Transforms act on a Batch in place (Component subclasses; parameters in `defaults`).  Parameters listed in PARAMS
may depend on the regime (data/params.py), so a transform can itself carry a change: per-regime mixing matrices
(correlation changes), per-regime delays (lead-lag), a mirrored nonlinearity (higher-order changes only), ...
    bind(d_in, seed, K) -> d_out   draw fixed random parameters and resolve regime-dependent ones
    apply(batch, rng)              per-call randomness from rng; regime context in batch.ctx (S, W)
    time_axis                      True if the transform moves samples in time (labels and oracles move with it)
regime: the regime process the transform's regime-dependent parameters follow (default: the dataset's target regime).
Naming another process, e.g. a "sensor" regime, makes observation conditions switch independently of the dynamics.
All timescales in samples."""
import numpy as np
from scipy.signal import lfilter
from scipy.stats import norm

from .params import resolve, varies, per_sample
from .pipeline import Context
from .registry import Component, register_transform


def slow_process(R, n, d, timescale, rng):
    """Unit-variance AR(1) with correlation time `timescale` samples, started stationary."""
    phi = np.exp(-1.0 / timescale)
    burn = int(min(20000, 5 * timescale))
    e = rng.standard_normal((R, n + burn, d)) * np.sqrt(1 - phi ** 2)
    return lfilter([1.0], [1.0, -phi], e, axis=1)[:, burn:]


class Transform(Component):
    defaults = dict(regime=None)
    PARAMS = ()                          # regime-dependent numeric parameters
    WIDTH = {}                           # their table widths (default 1)
    time_axis = False

    def spec(self):
        """{type, **params}; regime only when set (keeps the identity of datasets built before it existed)."""
        out = super().spec()
        if out.get("regime") is None:
            out.pop("regime", None)
        return out

    def context(self, b):
        """The regime context this transform follows."""
        return b.ctx if self.regime is None else b.contexts[self.regime]

    def bind(self, d_in, seed, K=1):
        self.d_in, self.K = d_in, K
        self.prng = np.random.default_rng([seed, 4441])
        self.tables = {p: resolve(getattr(self, p), K, self.WIDTH.get(p, 1), self.prng) for p in self.PARAMS}
        return self.setup(d_in)

    def setup(self, d_in):
        return d_in

    @property
    def driven(self):
        return any(varies(t) for t in self.tables.values())

    def value(self, name, b):
        c = self.context(b)
        return per_sample(self.tables[name], c.S, c.W)

    def apply(self, b, rng):
        raise NotImplementedError


def _regime_weights(c, K):
    """(R, n, K) regime weights of context c: soft weights if the regime ramps, else one-hot."""
    return c.W if c.W is not None else np.eye(K)[c.S]


# ----------------------------------------------------------------------------- mixing
@register_transform("project")
class Project(Transform):
    """Random linear map to D channels (injective when D >= input channels), ~unit variance per channel."""
    defaults = dict(D=64)

    def setup(self, d_in):
        if self.D >= d_in:                                               # orthonormal columns: injective
            Q, _ = np.linalg.qr(self.prng.standard_normal((self.D, d_in))); self.P = Q * np.sqrt(self.D / d_in)
        else:                                                            # orthonormal rows: a random compression
            Q, _ = np.linalg.qr(self.prng.standard_normal((d_in, self.D))); self.P = Q.T
        return self.D

    def apply(self, b, rng):
        b.x = b.x @ self.P.T


@register_transform("render")
class Render(Transform):
    """A fixed nonlinear rendering of d latent channels into D observation channels (a redundant, high-dimensional view
    of a low-dimensional state).
    kind="bumps":   D pixels split into d strips; latent channel c moves a Gaussian bump (width `width` in strip
                    units) along its strip, position 2 Phi(x_c / scale) - 1 in [-1, 1]: a 1-D "video" of d moving blobs.
    kind="fourier": random Fourier features sqrt(2) cos(a_j . x / scale + b_j), a_j ~ N(0, I), b_j ~ U(0, 2 pi): a
                    smooth injective embedding (for D >> d) that no linear map inverts.
    Regime-independent; add pixel noise with a `noise` transform after it."""
    defaults = dict(D=64, kind="bumps", width=0.08, scale=1.0)

    def init(self):
        if self.kind not in ("bumps", "fourier"):
            raise ValueError(f"render: kind must be 'bumps' or 'fourier', got {self.kind!r}")

    def setup(self, d_in):
        if self.kind == "bumps":
            if self.D < 2 * d_in:
                raise ValueError(f"render: bumps need at least 2 pixels per latent channel ({d_in} channels, D={self.D})")
            self.strip = np.minimum(np.arange(self.D) * d_in // self.D, d_in - 1)        # which latent each pixel shows
            counts = np.bincount(self.strip, minlength=d_in)
            self.pos = np.concatenate([np.linspace(-1, 1, c) for c in counts])          # pixel position in its strip
        else:
            self.A = self.prng.standard_normal((self.D, d_in)); self.b = self.prng.uniform(0, 2 * np.pi, self.D)
        return self.D

    def apply(self, b, rng):
        x = b.x / self.scale
        if self.kind == "bumps":
            centre = 2 * norm.cdf(x[..., self.strip]) - 1                                  # (R, n, D)
            b.x = np.exp(-0.5 * ((self.pos - centre) / self.width) ** 2)
        else:
            b.x = np.sqrt(2) * np.cos(x @ self.A.T + self.b)


@register_transform("linear")
class Linear(Transform):
    """Linear map y = M x with explicit or random matrices, optionally one per regime.
    weights: a matrix (rows = output channels, one weight per input channel in source order), a list of K matrices
    (one per regime), or {random: true, d_out: None, strength: 1.0, unit_rows: true} for one random matrix per regime
    M_k = I + strength G_k / sqrt(d) with rows normalised to unit norm: every channel keeps its variance and
    autocorrelation while the cross-channel correlation changes with the regime."""
    defaults = dict(weights=[[1.0]])

    def setup(self, d_in):
        w = self.weights
        if isinstance(w, dict):
            d_out = w.get("d_out") or d_in
            Ms = []
            for _ in range(self.K):
                M = np.eye(d_out, d_in) + w.get("strength", 1.0) * self.prng.standard_normal((d_out, d_in)) / np.sqrt(d_in)
                Ms.append(M / np.linalg.norm(M, axis=1, keepdims=True) if w.get("unit_rows", True) else M)
            self.M = np.stack(Ms)
        else:
            M = np.asarray(w, float)
            self.M = np.broadcast_to(M, (self.K,) + M.shape).copy() if M.ndim == 2 else M
        if self.M.ndim != 3 or self.M.shape[0] != self.K or self.M.shape[2] != d_in:
            raise ValueError(f"linear: weights must be a matrix with {d_in} columns or {self.K} such matrices; got {self.M.shape}")
        return self.M.shape[1]

    @property
    def driven(self):
        return bool(np.ptp(self.M, axis=0).max() > 0) if self.K > 1 else False

    def apply(self, b, rng):
        if not self.driven:
            b.x = b.x @ self.M[0].T
            return
        w = _regime_weights(self.context(b), self.K)
        b.x = sum(w[..., k:k + 1] * (b.x @ self.M[k].T) for k in range(self.K))


@register_transform("fold")
class Fold(Transform):
    """Sum of `blocks` equal consecutive channel blocks, y = x[:, :, :D] + x[:, :, D:2D] + ...: separately rendered
    sources (e.g. a signal and a nuisance, each a composite ending in its own render) added into one observation."""
    defaults = dict(blocks=2)

    def setup(self, d_in):
        if d_in % self.blocks:
            raise ValueError(f"fold: {d_in} channels do not split into {self.blocks} equal blocks")
        return d_in // self.blocks

    def apply(self, b, rng):
        b.x = b.x.reshape(*b.x.shape[:-1], self.blocks, -1).sum(-2)


@register_transform("couplings")
class Couplings(Transform):
    """Invertible affine coupling layers (nonlinear mixing across channels)."""
    defaults = dict(n=4, strength=1.0)

    def setup(self, d_in):
        h = d_in // 2
        self.perms = [self.prng.permutation(d_in) for _ in range(self.n)]
        self.A = [self.prng.standard_normal((h, d_in - h)) * 1.5 / np.sqrt(h) for _ in range(self.n)]
        self.C = [self.prng.standard_normal((h, d_in - h)) * 1.5 / np.sqrt(h) for _ in range(self.n)]
        return d_in

    def apply(self, b, rng):
        h, o = self.d_in // 2, b.x
        for perm, A, C in zip(self.perms, self.A, self.C):
            o = o[..., perm]; a, c = o[..., :h], o[..., h:]
            c = c * np.exp(0.5 * self.strength * np.tanh(a @ A)) + self.strength * np.tanh(a @ C)
            o = np.concatenate([a, c], -1)
        b.x = o


# ----------------------------------------------------------------------------- channel-wise
class ChannelTransform(Transform):
    """Applied to a subset of channels (`channels`: indices in source order; None = all)."""
    defaults = dict(channels=None)

    def apply(self, b, rng):
        idx = slice(None) if self.channels is None else list(self.channels)
        x = b.x.copy()
        x[..., idx] = self.f(b.x[..., idx], b, rng)
        b.x = x


@register_transform("pointwise")
class Pointwise(ChannelTransform):
    """A pointwise function, optionally mirrored per regime: y = m f(m x) with m = -1 where mirror is on.
    triangle: a sine of amplitude a becomes a triangle wave ((2a/pi) arcsin(x/a)); sign: square wave; abs;
    tanh: a tanh(x/a); cube: x^3; skew: x + strength (x^2 - 1).  For a symmetric input (e.g. Gaussian) mirroring keeps
    every even-order statistic (variance, autocorrelation, cross-correlation of independent channels) and flips the
    odd-order ones: fn=skew with mirror=[0, 1] is a change in higher-order moments only.  mirror may be given per
    channel, e.g. [[0, 0], [1, 0], [0, 1], [1, 1]] distinguishes 4 regimes on 2 channels."""
    defaults = dict(fn="triangle", amplitude=1.0, strength=0.3, mirror=0.0)
    FNS = ("triangle", "sign", "abs", "tanh", "cube", "skew")

    def init(self):
        if self.fn not in self.FNS:
            raise ValueError(f"pointwise: fn must be one of {self.FNS}, got {self.fn!r}")

    def setup(self, d_in):
        width = d_in if self.channels is None else len(self.channels)
        self.tables["mirror"] = resolve(self.mirror, self.K, width, self.prng)
        return d_in

    def g(self, x):
        a = self.amplitude
        if self.fn == "triangle":
            return a * 2 / np.pi * np.arcsin(np.clip(x / a, -1, 1))
        if self.fn == "sign":
            return a * np.sign(x)
        if self.fn == "abs":
            return np.abs(x)
        if self.fn == "tanh":
            return a * np.tanh(x / a)
        if self.fn == "cube":
            return x ** 3
        return x + self.strength * (x ** 2 - 1)

    def f(self, x, b, rng):
        if not self.driven and self.tables["mirror"][0, 0] == 0:
            return self.g(x)
        m = 1 - 2 * self.value("mirror", b)                             # +1, or -1 where mirrored
        return m * self.g(m * x)


@register_transform("delay")
class Delay(ChannelTransform):
    """Per-channel delays in samples, optionally per regime: y_c(t) = x_c(t - lag_c(t)).  Two copies of one signal
    (linear weights [[1], [1]]) with lags [[0, 4], [4, 0]] swap which channel leads: a lead-lag change."""
    defaults = dict(lags=0)

    def setup(self, d_in):
        width = d_in if self.channels is None else len(self.channels)
        self.tables["lags"] = resolve(self.lags, self.K, width, self.prng)
        return d_in

    def f(self, x, b, rng):
        n = x.shape[1]
        lag = np.rint(self.value("lags", b)).astype(int)
        idx = np.clip(np.arange(n)[None, :, None] - lag, 0, n - 1)
        return np.take_along_axis(x, idx, axis=1)


@register_transform("bandpass")
class Bandpass(ChannelTransform):
    """Second-order resonator (poles r e^{+-2 pi i freq}, r = exp(-pi bandwidth)): turns broadband input such as white
    noise into a narrowband oscillation at `freq` cycles/sample, scaled so that unit white noise gives unit variance."""
    defaults = dict(freq=0.05, bandwidth=0.01)

    def setup(self, d_in):
        r, w = np.exp(-np.pi * self.bandwidth), 2 * np.pi * self.freq
        self.a = [1.0, -2 * r * np.cos(w), r * r]
        h = lfilter([1.0], self.a, np.r_[1.0, np.zeros(int(50 / max(1 - r, 1e-4)))])
        self.b = [1.0 / np.sqrt((h ** 2).sum())]
        return d_in

    def f(self, x, b, rng):
        return lfilter(self.b, self.a, x, axis=1)


@register_transform("affine")
class Affine(Transform):
    """Per-channel calibration y_c = gain_c x_c + bias_c; gain and bias may depend on the regime (sensor conditions).
    Values: scalars, per-regime lists, {channels: [...]} or {base, spread} (independent per channel)."""
    defaults = dict(gain=1.0, bias=0.0)
    PARAMS = ("gain", "bias")

    def setup(self, d_in):
        self.tables = {p: resolve(getattr(self, p), self.K, d_in, self.prng) for p in self.PARAMS}
        return d_in

    def apply(self, b, rng):
        b.x = self.value("gain", b) * b.x + self.value("bias", b)


@register_transform("saturate")
class Saturate(Transform):
    defaults = dict(level=3.0)

    def apply(self, b, rng):
        b.x = self.level * np.tanh(b.x / self.level)


@register_transform("lowpass")
class LowPass(Transform):
    """Per-channel first-order sensor dynamics y_t = a y_{t-1} + (1 - a) x_t, a ~ U(0, max_alpha) fixed per channel."""
    defaults = dict(max_alpha=0.6)

    def setup(self, d_in):
        self.alpha = self.prng.uniform(0, self.max_alpha, d_in)
        return d_in

    def apply(self, b, rng):
        out = np.empty_like(b.x)
        for c, a in enumerate(self.alpha):
            out[..., c] = lfilter([1 - a], [1.0, -a], b.x[..., c], axis=1)
        b.x = out


# ----------------------------------------------------------------------------- sensor corruption
@register_transform("gain_drift")
class GainDrift(Transform):
    """Multiplicative slow gain shared by all channels: x * exp(scale * g_t)."""
    defaults = dict(scale=0.25, timescale=1600)

    def apply(self, b, rng):
        R, n, _ = b.x.shape
        b.x = b.x * np.exp(self.scale * slow_process(R, n, 1, self.timescale, rng))


@register_transform("volatility")
class Volatility(Transform):
    """Stochastic volatility: x * exp(scale * h_t), h a slow AR(1)."""
    defaults = dict(scale=0.3, timescale=640)

    def apply(self, b, rng):
        R, n, _ = b.x.shape
        b.x = b.x * np.exp(self.scale * slow_process(R, n, 1, self.timescale, rng))


@register_transform("noise")
class Noise(Transform):
    """Additive noise: white (timescale None) or coloured (AR(1) with the given correlation time).  sigma may depend on
    the regime (e.g. of a "sensor" regime process)."""
    defaults = dict(sigma=0.5, timescale=None)
    PARAMS = ("sigma",)

    def apply(self, b, rng):
        R, n, d = b.x.shape
        e = rng.standard_normal(b.x.shape) if not self.timescale else slow_process(R, n, d, self.timescale, rng)
        b.x = b.x + self.value("sigma", b) * e


@register_transform("wander")
class Wander(Transform):
    """Per-channel baseline wander (very slow AR(1))."""
    defaults = dict(scale=0.5, timescale=6400)

    def apply(self, b, rng):
        R, n, d = b.x.shape
        b.x = b.x + self.scale * slow_process(R, n, d, self.timescale, rng)


@register_transform("outliers")
class Outliers(Transform):
    defaults = dict(rate=0.01, size=5.0)

    def apply(self, b, rng):
        hit = rng.random(b.x.shape) < self.rate
        b.x = b.x + hit * self.size * rng.standard_normal(b.x.shape)


@register_transform("dropout")
class Dropout(Transform):
    """Whole channels missing in blocks of `block` samples: zero-filled, and recorded in the batch mask.  rate may depend
    on the regime (taken at each block's start)."""
    defaults = dict(rate=0.05, block=32)
    PARAMS = ("rate",)

    def apply(self, b, rng):
        R, n, d = b.x.shape; nb = -(-n // self.block)
        rate = self.value("rate", b)[:, ::self.block]                                # (R, nb, 1)
        keep = np.repeat(rng.random((R, nb, d)) > rate, self.block, axis=1)[:, :n]
        b.x = b.x * keep
        b.mask = keep if b.mask is None else (b.mask & keep)


@register_transform("time_warp")
class TimeWarp(Transform):
    """Smooth random time warp (rate exp(strength * slow AR)); labels, regime context and oracles move with it."""
    defaults = dict(strength=0.3, timescale=128)
    time_axis = True

    def apply(self, b, rng):
        R, n, _ = b.x.shape
        rate = np.exp(self.strength * slow_process(R, n, 1, self.timescale, rng)[..., 0])
        pos = np.cumsum(rate, 1); pos = (pos - pos[:, :1]) / (pos[:, -1:] - pos[:, :1]) * (n - 1)
        grid = np.arange(n); idx = np.clip(np.round(pos).astype(int), 0, n - 1)
        x = np.empty_like(b.x)
        for r in range(R):
            for j in range(b.x.shape[-1]):
                x[r, :, j] = np.interp(pos[r], grid, b.x[r, :, j])
        b.x = x
        take = lambda a: np.stack([a[r, idx[r]] for r in range(R)])
        b.S = take(b.S); b.regimes = {k: take(v) for k, v in b.regimes.items()}
        b.oracle = {k: take(v) for k, v in b.oracle.items()}
        warp = lambda c: Context(take(c.S), None if c.W is None else take(c.W), c.K, c.rng)
        b.ctx = warp(b.ctx); b.contexts = {k: warp(c) for k, c in b.contexts.items()}
        if b.mask is not None:
            b.mask = take(b.mask)


class Standardize(Transform):
    """Per-channel standardisation with statistics calibrated once by the Dataset (not per batch); applied
    automatically at the end of every pipeline unless the config sets standardize: false."""
    def setup(self, d_in):
        self.mu, self.sd = np.zeros(d_in), np.ones(d_in)
        return d_in

    def calibrate(self, x):
        flat = x.reshape(-1, x.shape[-1]); self.mu, self.sd = flat.mean(0), flat.std(0) + 1e-6

    def apply(self, b, rng):
        b.x = (b.x - self.mu) / self.sd
