"""Registries of dataset components (regime processes, sources, transforms), built on cpbench.core.

    @register_transform("noise")
    class Noise(Transform):
        defaults = dict(sigma=0.5)              # every parameter declared here; unknown keys are rejected
        def apply(self, b, rng): b.x = b.x + self.sigma * rng.standard_normal(b.x.shape)

    build("transform", {"type": "noise", "sigma": 0.3})   ->  instance
"""
from cpbench.core import Component, Registry  # noqa: F401  (Component is the base class of every data component)

REGISTRIES = dict(regime=Registry("regime"), source=Registry("source"), transform=Registry("transform"))
register_regime = REGISTRIES["regime"].register
register_source = REGISTRIES["source"].register
register_transform = REGISTRIES["transform"].register


def build(kind, spec):
    return REGISTRIES[kind].make(spec)


def available(kind):
    """Registered component names of one kind (regime, source, transform)."""
    return REGISTRIES[kind].available()


def short(kind, spec):
    """Compact readable form of a component: its type plus the parameters that differ from the defaults,
    e.g. autocorr(effect=0.5), noise(sigma=0.3)."""
    spec = dict(spec); t = spec.pop("type")
    if t == "composite":                                                # [inner sources | inner transforms]
        inner = "+".join(short("source", s) for s in spec.get("sources", []))
        chain = ">".join(short("transform", x) for x in spec.get("transforms", []))
        return f"[{inner}{'|' + chain if chain else ''}]"
    defaults = REGISTRIES[kind].defaults(t)
    same = lambda a, b: (list(a) == list(b)) if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)) else a == b
    diff = {k: v for k, v in spec.items() if k not in defaults or not same(defaults[k], v)}
    fmt = lambda v: f"{v:g}" if isinstance(v, float) else str(v).replace(" ", "").replace("'", "")
    return t + (f"({','.join(f'{k}={fmt(v)}' for k, v in sorted(diff.items()))})" if diff else "")
