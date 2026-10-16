"""Running chains of transforms (shared by the Dataset and by composite sources).

Context   what a source or transform sees of the regimes: S (R, n) the regime path it follows, W (R, n, K) soft weights
          when the regime process has a ramp (else None), K, and the rng.
Batch     the working container passed through transforms: x (R, n, d), the regime context, the target regime labels
          (moved by time-warping transforms), other regime paths, oracle arrays and the missing-data mask."""
from types import SimpleNamespace


def Context(S, W, K, rng):
    return SimpleNamespace(S=S, W=W, K=K, rng=rng)


class Batch:
    def __init__(self, x, ctx, regimes=None, oracle=None, contexts=None):
        self.x, self.ctx, self.S = x, ctx, ctx.S
        self.regimes, self.oracle, self.mask = regimes or {}, oracle or {}, None
        self.contexts = contexts or {}                                   # every regime process's context, by name


def bind_chain(chain, d, seed, K):
    """Bind transforms in order; returns the output channel count."""
    for i, t in enumerate(chain):
        d = t.bind(d, seed + i, K)
    return d


def run_chain(chain, x, ctx):
    """Apply a chain inside a composite source (no time-warping there: labels are not carried)."""
    b = Batch(x, ctx)
    for t in chain:
        if t.regime is not None:
            raise ValueError(f"{t.name} inside a composite follows the composite's regime; regime= is for dataset transforms")
        if t.time_axis:
            raise ValueError(f"{t.name} moves samples in time: use it in the dataset's transforms, not in a composite")
        t.apply(b, ctx.rng)
    return b.x
