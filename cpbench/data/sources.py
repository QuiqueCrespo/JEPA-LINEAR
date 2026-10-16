"""Sources: switching LINEAR signals.  Every numeric parameter may depend on the regime (data/params.py), and a
change type is simply a parameter that does: a mean change is `level` with mu=[-1, 1], an autocorrelation change is
`ar` with phi=[0.2, 0.9], a frequency change is `oscillator` with freq=[0.05, 0.08], and so on.  Nonlinearities belong
to transforms (data/transforms.py); `composite` bundles sources with their own transforms into one signal.

Interface (Component subclasses, parameters in `defaults`):
    bind(K, seed)          resolve regime-dependent parameters (seeded); sets `driven` (does anything depend on the regime?)
    d                      number of output channels
    simulate(ctx) -> x     (R, n, d); ctx carries the regime path S (R, n), soft weights W (or None) and rng
    oracle                 dict of extra (R, n, ...) arrays from the last simulate (e.g. a hidden state)
expose: a name under which the Dataset (or an enclosing composite) also returns this source's own output, before
`scale` and any transform, as an oracle array (e.g. the latent of a nuisance that is rendered afterwards).
The Dataset multiplies the output by `scale`.  All timescales in samples."""
import numpy as np
from scipy.signal import lfilter

from .params import resolve, varies, per_sample
from .pipeline import bind_chain, run_chain
from .registry import Component, register_source, build


class Source(Component):
    defaults = dict(scale=1.0, regime="main", expose=None)
    PARAMS = ()                         # names of regime-dependent numeric parameters (resolved to (K, d) tables)
    WIDTH = {}                          # parameters whose table width is not d (e.g. {"corr": 1})
    grid = None

    def init(self):
        self.oracle = {}

    def spec(self):
        """{type, **params}; expose only when set (keeps the identity of datasets built before it existed)."""
        out = super().spec()
        if out.get("expose") is None:
            out.pop("expose", None)
        return out

    def bind(self, K, seed):
        self.K, self.seed = K, seed
        self.prng = np.random.default_rng([seed, 7919])
        self.tables = {p: resolve(getattr(self, p), K, self.WIDTH.get(p, getattr(self, "d", 1)), self.prng) for p in self.PARAMS}
        self.setup()
        self.driven = any(varies(t) for t in self.tables.values()) or self.extra_driven()
        return self

    def setup(self):
        pass

    def extra_driven(self):
        return False

    def value(self, name, ctx):
        """Per-sample values (R, n, width) of a regime-dependent parameter."""
        return per_sample(self.tables[name], ctx.S, ctx.W)

    @property
    def out_dim(self):
        """Number of output channels (d unless the source emits more, e.g. quadrature oscillators)."""
        return self.n_out if hasattr(self, "n_out") else self.d


def slow_ar1(R, n, d, phi, rng):
    """Unit-variance AR(1) with coefficient phi, started stationary (vectorised filter)."""
    burn = int(min(5000, 10 / (1 - phi)))
    e = rng.standard_normal((R, n + burn, d)) * np.sqrt(1 - phi ** 2)
    return lfilter([1.0], [1.0, -phi], e, axis=1)[:, burn:]


class Channels(Source):
    """A linear source with `d` channels."""
    defaults = dict(d=1)


def equicorrelated(R, n, d, rho, rng):
    """Unit-variance noise with pairwise correlation rho (per sample, (R, n, 1)): sqrt(1-rho) e_c + sqrt(rho) e_common."""
    return np.sqrt(1 - rho) * rng.standard_normal((R, n, d)) + np.sqrt(rho) * rng.standard_normal((R, n, 1))


@register_source("level")
class Level(Channels):
    """A constant level per regime: x = mu (a pure step signal when mu depends on the regime)."""
    defaults = dict(mu=0.0)
    PARAMS = ("mu",)

    def simulate(self, ctx):
        return self.value("mu", ctx)


@register_source("white")
class White(Channels):
    """White Gaussian noise with standard deviation sigma and equicorrelation corr between channels."""
    defaults = dict(sigma=1.0, corr=0.0)
    PARAMS = ("sigma", "corr")
    WIDTH = {"corr": 1}

    def simulate(self, ctx):
        R, n = ctx.S.shape
        return self.value("sigma", ctx) * equicorrelated(R, n, self.d, self.value("corr", ctx), ctx.rng)


@register_source("ar")
class AR(Channels):
    """AR(2) per channel: x_t = phi x_{t-1} + phi2 x_{t-2} + g_t e_t, innovations equicorrelated (corr) across channels.
    unit_variance=True scales the innovations so that the stationary standard deviation is sigma in every regime:
    a change of phi is then an autocorrelation change and nothing else."""
    defaults = dict(phi=0.8, phi2=0.0, sigma=1.0, corr=0.0, unit_variance=True)
    PARAMS = ("phi", "phi2", "sigma", "corr")
    WIDTH = {"corr": 1}

    def simulate(self, ctx):
        R, n = ctx.S.shape
        p1, p2, sg = self.value("phi", ctx), self.value("phi2", ctx), self.value("sigma", ctx)
        if self.unit_variance:                                       # stationary variance of AR(2) with unit innovations
            v = (1 - p2) / ((1 + p2) * ((1 - p2) ** 2 - p1 ** 2))
            g = sg / np.sqrt(np.maximum(v, 1e-12))
        else:
            g = sg
        e = g * equicorrelated(R, n, self.d, self.value("corr", ctx), ctx.rng)
        x = np.empty((R, n, self.d))
        a = sg[:, 0] * ctx.rng.standard_normal((R, self.d)); b = a.copy()        # approximately stationary start
        for t in range(n):
            a, b = p1[:, t] * a + p2[:, t] * b + e[:, t], a
            x[:, t] = a
        return x


@register_source("oscillator")
class Oscillator(Channels):
    """d damped oscillators z_t = damp e^{2 pi i freq} z_{t-1} + innovation, output Re z (quadrature=True: Re and Im,
    2d channels).  amplitude: deterministic part of the initial state (random phase per series); sigma: stationary
    standard deviation of the noise part (unit_variance=True keeps it fixed when damp changes).  A pure sine:
    damp=1, sigma=0, amplitude=1."""
    defaults = dict(freq=0.05, damp=0.97, sigma=1.0, amplitude=0.0, quadrature=False, unit_variance=True)
    PARAMS = ("freq", "damp", "sigma")

    @property
    def n_out(self):
        return 2 * self.d if self.quadrature else self.d

    def simulate(self, ctx):
        R, n = ctx.S.shape
        f, r, sg = self.value("freq", ctx), self.value("damp", ctx), self.value("sigma", ctx)
        rot = r * np.exp(2j * np.pi * f)
        g = sg * np.sqrt(np.maximum(1 - r ** 2, 0)) if self.unit_variance else sg
        e = g * (ctx.rng.standard_normal((R, n, self.d)) + 1j * ctx.rng.standard_normal((R, n, self.d)))
        z = self.amplitude * np.exp(2j * np.pi * ctx.rng.random((R, self.d)))
        z = z + sg[:, 0] * (ctx.rng.standard_normal((R, self.d)) + 1j * ctx.rng.standard_normal((R, self.d)))
        out = np.empty((R, n, self.d), complex)
        for t in range(n):
            z = rot[:, t] * z + e[:, t]
            out[:, t] = z
        return np.concatenate([out.real, out.imag], -1) if self.quadrature else out.real


@register_source("trend")
class Trend(Channels):
    """Integrated drift: x_t = (1 - kappa) x_{t-1} + drift_t + sigma_t e_t (kappa > 0: slowly mean-reverting)."""
    defaults = dict(drift=0.0, sigma=0.0, kappa=0.0)
    PARAMS = ("drift", "sigma")

    def simulate(self, ctx):
        R, n = ctx.S.shape
        inc = self.value("drift", ctx) + self.value("sigma", ctx) * ctx.rng.standard_normal((R, n, self.d))
        x = np.empty((R, n, self.d)); cur = np.zeros((R, self.d))
        for t in range(n):
            cur = (1 - self.kappa) * cur + inc[:, t]
            x[:, t] = cur
        return x


@register_source("impulses")
class Impulses(Channels):
    """Sparse heavy-tailed events: Bernoulli(rate) x size x t_3 (non-Gaussian noise; rate may depend on the regime)."""
    defaults = dict(rate=0.02, size=3.0)
    PARAMS = ("rate",)

    def simulate(self, ctx):
        R, n = ctx.S.shape
        hit = ctx.rng.random((R, n, self.d)) < self.value("rate", ctx)
        return hit * self.size * ctx.rng.standard_t(3, (R, n, self.d))


@register_source("composite")
class Composite(Source):
    """Sources plus their own transforms, acting as one source: e.g. an AR process mixed by regime-dependent
    matrices (a correlation change), duplicated and delayed (lead-lag), or passed through a mirrored nonlinearity
    (a change in higher-order moments only).  Inner sources follow the composite's regime.  combine="concat" stacks the
    inner sources' channels; combine="sum" adds them channel by channel (all inner sources need the same width),
    e.g. a mean change = level + AR carrier."""
    defaults = dict(sources=[], transforms=[], combine="concat")

    def bind(self, K, seed):
        self.K, self.seed = K, seed
        self.inner = [build("source", s) for s in self.sources]
        for i, s in enumerate(self.inner):
            s.bind(K, seed * 100 + i)
        widths = [s.out_dim for s in self.inner]
        if self.combine == "sum" and len(set(widths)) > 1:
            raise ValueError(f"composite combine=sum needs inner sources of equal width, got {widths}")
        self.chain = [build("transform", t) for t in self.transforms]
        self.n_out = bind_chain(self.chain, widths[0] if self.combine == "sum" else sum(widths), seed * 100 + 50, K)
        self.grid = next((s.grid for s in self.inner if s.grid), None)
        self.driven = any(s.driven for s in self.inner) or any(t.driven for t in self.chain)
        return self

    def simulate(self, ctx):
        raw = [s.simulate(ctx) for s in self.inner]
        xs = [s.scale * x for s, x in zip(self.inner, raw)]
        self.oracle = {k: v for s in self.inner for k, v in s.oracle.items()}
        self.oracle.update({s.expose: x for s, x in zip(self.inner, raw) if s.expose})
        return run_chain(self.chain, sum(xs) if self.combine == "sum" else np.concatenate(xs, -1), ctx)
