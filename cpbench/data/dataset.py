"""Dataset = regime process(es) + sources + transforms + seed, fully described by a plain dict (YAML-able).

    cfg = {
      "regime":     {"type": "markov", "K": 2, "dwell": 320},       # or "regimes": {"main": {...}, "aux": {...}}
      "target":     "main",                                          # the regime path that is scored
      "sources":    [{"type": "ar", "d": 4, "phi": [0.2, 0.9]},     # switching linear signals (data/sources.py);
                     {"type": "ar", "d": 3, "phi": 0.999}],          # any parameter may depend on the regime
      "transforms": [{"type": "project", "D": 64}, {"type": "noise", "sigma": 0.5}],   # nonlinearities, corruption
      "standardize": True,                                           # per-channel standardisation at the end
      "seed": 0, "splits": {"test": [16, 64000], "val": [4, 64000]}, "margin": 32,
    }
    ds = Dataset(cfg)    # or Dataset("my_dataset.yaml"); grids of datasets: cpbench.data.expand(...)

A source whose parameters depend on the regime is part of the SIGNAL, otherwise it is a NUISANCE; the top-level
transforms are the CORRUPTION.  The dataset is identified by its construction: info.id reads
regime|signal|nuisance|corruption#hash.  Data are continuous series; regimes switch at arbitrary samples (or on a
source-required grid, which gets a random phase per series).  Model windows and gaps never enter the data."""
import copy, hashlib, json, os
from types import SimpleNamespace
import numpy as np
from omegaconf import OmegaConf

from . import regime as _regime, sources as _sources, structured as _structured, transforms as _transforms  # noqa: F401 (registration)
from .grid import is_grid
from .params import soft_weights
from .pipeline import Batch, Context
from .registry import build, short
from .transforms import Standardize

CACHE = os.environ.get("CPBENCH_CACHE", os.path.join(os.path.dirname(__file__), "..", "..", "cpbench_cache"))
DEFAULT_SPLITS = {"test": [16, 64000], "val": [4, 64000]}


def load_config(obj):
    """Dataset config from a dict or a YAML / JSON file."""
    cfg = copy.deepcopy(obj) if isinstance(obj, dict) else OmegaConf.to_container(OmegaConf.load(obj), resolve=True)
    if is_grid(cfg):
        raise ValueError("this is a grid of datasets (signal / nuisance / corruption slots): use cpbench.data.expand")
    return cfg


class Dataset:
    def __init__(self, config):
        cfg = load_config(config.config() if isinstance(config, Dataset) else config)
        regs = cfg.get("regimes") or {"main": cfg.get("regime", {"type": "none"})}
        self.regimes = {k: build("regime", v) for k, v in regs.items()}
        self.target = cfg.get("target", "main")
        self.sources = [build("source", s) for s in cfg["sources"]]
        self.transforms = [build("transform", t) for t in cfg.get("transforms", [])]
        self.standardize = cfg.get("standardize", True)
        self.seed = cfg.get("seed", 0)
        self.splits = {**DEFAULT_SPLITS, **cfg.get("splits", {})}
        self.margin = cfg.get("margin", 32)
        for i, s in enumerate(self.sources):                              # fix each source's random parameters
            if s.regime not in self.regimes:
                raise ValueError(f"source {s.name} follows regime {s.regime!r}, which is not defined")
            s.bind(self.regimes[s.regime].K, self.seed * 1000 + i)
            if s.grid:
                r = self.regimes[s.regime]
                if r.grid not in (None, s.grid):
                    raise ValueError(f"source {s.name} needs regime grid {s.grid}, regime has {r.grid}")
                r.grid = s.grid; r.params["grid"] = s.grid
        d = sum(s.out_dim for s in self.sources)
        for i, t in enumerate(self.transforms):                          # each transform follows its own regime process
            reg = t.regime or self.target
            if reg not in self.regimes:
                raise ValueError(f"transform {t.name} follows regime {reg!r}, which is not defined")
            d = t.bind(d, self.seed * 1000 + 500 + i, self.regimes[reg].K)
        self.D = d
        self.d_clean = sum(s.out_dim for s in self.sources if s.driven)
        self._std = Standardize() if self.standardize else None
        if self._std:                                                    # calibrated once, from a fixed draw
            self._std.bind(self.D, 0)
            cal = self.sample(16, 2048, np.random.default_rng([self.seed, 1]))
            self._std.calibrate(cal["obs"].astype(np.float64))

    # ------------------------------------------------------------------ description
    def config(self):
        """The full construction as a plain dict; Dataset(ds.config()) rebuilds an identical dataset."""
        return dict(regimes={k: r.spec() for k, r in self.regimes.items()}, target=self.target,
                    sources=[s.spec() for s in self.sources], transforms=[t.spec() for t in self.transforms],
                    standardize=self.standardize, seed=self.seed, splits=self.splits, margin=self.margin)

    def digest(self):
        """Hash of the construction (splits excluded: the same dataset at another test size keeps its identity)."""
        cfg = {k: v for k, v in self.config().items() if k != "splits"}
        return hashlib.sha1(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:8]

    @property
    def K(self):
        return self.regimes[self.target].K

    @property
    def info(self):
        """Construction summary for results tables: one readable field per slot, plus the id."""
        regime = ",".join((f"{k}=" if len(self.regimes) > 1 else "") + short("regime", r.spec()) for k, r in self.regimes.items())
        signal = "+".join(short("source", s.spec()) for s in self.sources if s.driven) or "-"
        nuisance = "+".join(short("source", s.spec()) for s in self.sources if not s.driven) or "-"
        corruption = ">".join(short("transform", t.spec()) for t in self.transforms) or "-"
        changing = [t.name for t in self.transforms if t.driven]
        return SimpleNamespace(id=f"{regime}|{signal}|{nuisance}|{corruption}#{self.digest()}", digest=self.digest(),
                               regime=regime, signal=signal, nuisance=nuisance, corruption=corruption,
                               regime_dependent_transforms=changing, K=self.K, margin=self.margin, seed=self.seed)

    def with_splits(self, **splits):
        cfg = self.config(); cfg["splits"] = {**cfg["splits"], **{k: list(v) for k, v in splits.items()}}
        return Dataset(cfg)

    def __repr__(self):
        return f"Dataset({self.info.id}, D={self.D})"

    # ------------------------------------------------------------------ sampling
    def sample(self, R, n, rng, burn=256, fix=None):
        """R series of n samples.  Returns obs (R,n,D) float32, S_s (R,n) target regimes, cp_s (R,n) bool,
        clean (R,n,d_clean) float32 regime-driven source channels before transforms, mask (R,n,D) bool if a transform
        drops data, oracle arrays (e.g. hidden), and S_<name> for non-target regimes.
        fix={name: k} holds regime process `name` at regime k throughout.  Regime paths are drawn first, then sources,
        then transforms, all from rng, so two calls with equal rng seeds and different `fix` are counterfactual pairs:
        the same innovations and noise draws under different regimes (e.g. one hidden trajectory seen through two
        sensor conditions)."""
        grid = max([r.grid or 1 for r in self.regimes.values()])
        lead = burn + rng.integers(grid, size=R)                         # per-series random phase of internal grids
        N = n + burn + grid
        paths = {k: r.sample(R, N, rng) for k, r in self.regimes.items()}
        for k, v in (fix or {}).items():
            if k not in paths or not 0 <= v < self.regimes[k].K:
                raise ValueError(f"fix: no regime {v} in process {k!r}")
            paths[k] = np.full_like(paths[k], v)
        ctxs = {k: Context(p, soft_weights(p, self.regimes[k].K, self.regimes[k].ramp), self.regimes[k].K, rng)
                for k, p in paths.items()}
        xs, clean, oracle = [], [], {}
        for s in self.sources:
            raw = s.simulate(ctxs[s.regime]); x = s.scale * raw; xs.append(x)
            if s.driven:
                clean.append(x)
            oracle.update(s.oracle)
            if s.expose:
                oracle[s.expose] = raw
        b = Batch(np.concatenate(xs, -1), ctxs[self.target], {k: v for k, v in paths.items() if k != self.target}, oracle, ctxs)
        b.oracle["__clean"] = np.concatenate(clean, -1) if clean else np.zeros((R, N, 0))   # moves with time warps
        for t in self.transforms:
            t.apply(b, rng)
        if self._std is not None:
            self._std.apply(b, rng)
        idx = lead[:, None] + np.arange(n)[None]
        crop = lambda a: np.take_along_axis(a, idx.reshape(R, n, *([1] * (a.ndim - 2))), axis=1)
        S = crop(b.S)
        cp = np.zeros_like(S, dtype=bool); cp[:, 1:] = S[:, 1:] != S[:, :-1]
        out = dict(obs=crop(b.x).astype(np.float32), S_s=S, cp_s=cp, clean=crop(b.oracle.pop("__clean")).astype(np.float32))
        if b.mask is not None:
            out["mask"] = crop(b.mask)
        out.update({k: crop(v).astype(np.float32) for k, v in b.oracle.items()})
        out.update({f"S_{k}": crop(v) for k, v in b.regimes.items()})
        return out

    def _cached(self, split, seed_tag, cache=True):
        R, n = self.splits[split]
        if not cache:
            return self.sample(R, n, np.random.default_rng([self.seed, seed_tag]))
        os.makedirs(CACHE, exist_ok=True)
        path = os.path.join(CACHE, f"{self.digest()}_{split}_{R}x{n}.npz")
        if os.path.exists(path):
            with np.load(path) as z:
                return {k: z[k] for k in z.files}
        d = self.sample(R, n, np.random.default_rng([self.seed, seed_tag]))
        np.savez(path, **d)
        return d

    def test_set(self, cache=True):
        """Fixed test split (cached on disk under the dataset digest unless cache=False)."""
        return self._cached("test", 2, cache)

    def val_set(self, cache=True):
        return self._cached("val", 3, cache)

    def train_batch(self, B, W, gap, rng):
        """Fresh (context, target) pairs: context = samples [0, W), target = [W + gap, 2W + gap) of new series.
        W and gap are model hyperparameters in samples."""
        d = self.sample(B, 2 * W + gap, rng)
        return d["obs"][:, :W], d["obs"][:, W + gap:], d
