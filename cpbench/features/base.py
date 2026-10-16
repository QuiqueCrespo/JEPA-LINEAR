"""Feature learners: everything that turns observation windows into feature vectors.  They make no decisions;
change-point detection and regime clustering are done by detectors (cpbench/detectors) on top of their output.

FeatureLearner   parameters in `defaults` (cpbench.core.Component), including the window length in samples
                   fit(dataset, run_dir)  learn from the dataset's fresh-sample stream (no-op for fixed features)
                   embed(windows)         (N, W, D) -> (N, d)
                   views(windows)         optional extra feature views, e.g. a JEPA target encoder {name: (N, d)}
                   represent(data)        a split's series -> features (R, T, d) of consecutive windows
                   input_key              which array of the data the learner reads ("obs"; oracles: clean/hidden)
                   save(path) / load(path, D)
Fixed features (raw window statistics, oracles) are in raw.py; learnt ones (TorchFeatureLearner) in torch.py / ssl.py.
"""
import hashlib, json
import numpy as np

from cpbench.core import Component, Registry
from cpbench.data.windows import to_windows

LEARNERS = Registry("feature learner")
register_learner = LEARNERS.register
learner_preset = LEARNERS.preset


def make_learner(spec, **params):
    return LEARNERS.make(spec, **params)


class FeatureLearner(Component):
    defaults = dict(window=32)
    RUNTIME = ()                      # parameters that do not change the result (excluded from the run identity)
    trainable = False
    input_key = "obs"

    def fit(self, dataset, run_dir=None, log=print):
        return self

    def embed(self, windows):
        raise NotImplementedError

    def views(self, windows):
        return {}

    def save(self, path):
        pass

    def load(self, path, D=None):
        return self

    def label(self):
        """Readable name without runtime settings."""
        keep = {k: v for k, v in self.params.items() if k not in self.RUNTIME}
        base = getattr(self, "_defaults", {})
        diff = {k: v for k, v in keep.items() if base.get(k, object()) != v}
        fmt = lambda v: f"{v:g}" if isinstance(v, float) else str(v)
        return self.name + (f"({', '.join(f'{k}={fmt(v)}' for k, v in diff.items())})" if diff else "")

    def identity(self):
        """Parameters that define the features (runtime settings such as workers or device excluded)."""
        return {k: v for k, v in self.params.items() if k not in self.RUNTIME}

    def tag(self):
        """Run-directory tag: w<window>[_g<gap>]_<hash of the identity>."""
        h = hashlib.sha1(json.dumps(self.identity(), sort_keys=True, default=str).encode()).hexdigest()[:6]
        g = f"_g{self.gap}" if "gap" in self.params else ""
        return f"w{self.window}{g}_{h}"

    def represent(self, data, chunk=4000):
        """data[self.input_key] (R, n, D) -> {"features": (R, T, d), **views}, window starts (T,)."""
        if self.input_key not in data:
            raise KeyError(f"{self.name} reads data[{self.input_key!r}], which this dataset does not provide")
        win, starts = to_windows(data[self.input_key], self.window, self.window)
        R, T = win.shape[:2]
        flat = win.reshape(R * T, self.window, -1)
        out = {"features": np.concatenate([self.embed(flat[i:i + chunk]) for i in range(0, len(flat), chunk)])}
        if out["features"].shape[-1] == 0:
            raise ValueError(f"{self.label()} gives no features on {flat.shape[-1]}-channel data")
        for i in range(0, len(flat), chunk):
            for k, v in self.views(flat[i:i + chunk]).items():
                out.setdefault(k, []).append(v)
        out = {k: (np.concatenate(v) if isinstance(v, list) else v).reshape(R, T, -1) for k, v in out.items()}
        return out, starts
