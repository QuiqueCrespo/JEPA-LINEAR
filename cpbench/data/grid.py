"""Grids of datasets from component slots (YAML or dict).  Each slot lists alternatives; the datasets are the
product of the slots.  Datasets are identified by their construction, not by names.

    regime:     [{type: markov, K: 4, dwell: 320}]                 # alternatives: a regime spec, or {name: spec, ...}
    signal:     [{type: autocorr, effect: 1.0}, {type: phase}]      # alternatives: a source, or a list of sources
    nuisance:   [[], [{type: slow_ar, n: 3}, {type: bursts}]]       # alternatives: lists of nuisance sources
    corruption: [[], [{type: project, D: 64}, {type: noise}]]       # alternatives: lists of transforms
    seed: 0                                                         # anything else is copied into every dataset
    splits: {test: [16, 64000], val: [4, 64000]}

YAML anchors (&name / *name) keep repeated chains short."""
import itertools
from omegaconf import OmegaConf

SLOTS = ("regime", "signal", "nuisance", "corruption")


def load_grid(obj):
    if isinstance(obj, str):
        return OmegaConf.to_container(OmegaConf.load(obj), resolve=True)
    return dict(obj)


def is_grid(cfg):
    return any(k in cfg for k in ("signal", "nuisance", "corruption"))


def expand(obj):
    """Grid (path or dict) -> list of Dataset configs."""
    g = load_grid(obj)
    as_list = lambda v: v if isinstance(v, list) else [v]               # a single alternative may be written bare
    alts = {k: as_list(g.get(k, None)) for k in SLOTS}
    fixed = {k: v for k, v in g.items() if k not in SLOTS}
    out = []
    for reg, sig, nui, cor in itertools.product(*(alts[k] for k in SLOTS)):
        cfg = dict(fixed)
        if reg is not None:
            cfg["regime" if "type" in reg else "regimes"] = reg
        sig = [] if sig is None else (sig if isinstance(sig, list) else [sig])
        cfg["sources"] = list(sig) + list(nui or [])
        cfg["transforms"] = list(cor or [])
        out.append(cfg)
    return out
