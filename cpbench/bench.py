"""The benchmark composes feature learners and detectors on a dataset.

    from cpbench import Benchmark, make_learner
    bench = Benchmark(dataset_config, detectors=["hmm", "bocpd"], quick=True)   # dataset: config dict, YAML or Dataset
    df = bench.run(make_learner("jepa_shared", steps=2000))     # fit (resumable) -> features -> every detector
    bench.compare(["raw_stats", "jepa_shared"], detector="hmm", metric="acc")

Protocol: the learner is fitted on the dataset's fresh-sample stream; its features of the fixed test split are
computed once and cached; every detector then runs on them (transductively, given K) and is scored at sample level;
supervised probes (diagnostics only) appear as detector "probe".  Results are cached per detector, so adding a
detector or a learner later only computes what is new.

<out>/<signal>_<dataset digest>/<learner>/<learner tag>/s<seed>/
    learner.json, model.pt, log.jsonl                    the fitted learner
    <protocol>/features.npz                              features of the test split (protocol = test split size)
    <protocol>/<detector>_<hash>.csv, probe.csv           results (CSV: small, and independent of pyarrow versions)"""
import hashlib, json, os, time
import numpy as np
import pandas as pd

from cpbench.data.dataset import Dataset
from cpbench.data.windows import majority_labels
from cpbench.detectors import STANDARD, make_detector
from cpbench.eval.probes import probes as run_probes
from cpbench.eval.scoring import score
from cpbench.features import make_learner
from cpbench.results.aggregate import table, read_result


def _hash(obj, n=6):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def dataset_dir(ds):
    i = ds.info
    return f"{''.join(c if c.isalnum() or c in '-_.' else '-' for c in i.signal)[:60]}_{i.digest}"


class Benchmark:
    def __init__(self, dataset, detectors=STANDARD, probes=True, out="cpbench_runs", quick=False, cache_data=True, log=print):
        ds = dataset if isinstance(dataset, Dataset) else Dataset(dataset)
        if quick:                                                         # small fixed sets for fast iteration
            ds = ds.with_splits(test=[4, 16000], val=[2, 9600])
        self.ds, self.out, self.probes, self.cache_data, self.log = ds, out, probes, cache_data, log
        self.detectors = [make_detector(d) if isinstance(d, (str, dict)) else d for d in detectors]
        self._test = None

    def with_test(self, series, samples):
        """Same benchmark on a test split of another size (a different evaluation protocol)."""
        self.ds, self._test = self.ds.with_splits(test=[series, samples]), None
        return self

    @property
    def test(self):
        if self._test is None:
            self._test = self.ds.test_set(cache=self.cache_data)
        return self._test

    def run_dir(self, f):
        return os.path.join(self.out, dataset_dir(self.ds), f.name, f.tag(), f"s{f.params.get('seed', 0)}")

    def protocol(self):
        return "test_{}x{}".format(*self.ds.splits["test"])

    # ------------------------------------------------------------------ steps
    def fit(self, f, refit=False):
        """Fit (or load) a feature learner in its run directory."""
        f = make_learner(f) if isinstance(f, (str, dict)) else f
        rd = self.run_dir(f); os.makedirs(rd, exist_ok=True)
        with open(os.path.join(rd, "learner.json"), "w") as fh:
            json.dump(dict(name=f.name, params=f.params, dataset=self.ds.config()), fh, indent=1, default=str)
        mp = os.path.join(rd, "model.pt")
        if f.trainable and os.path.exists(mp) and not refit:
            f.load(mp, D=self.ds.D)
        else:
            f.fit(self.ds, rd, log=self.log)
            if f.trainable:
                f.save(mp)
        return f

    def features(self, f, refit=False):
        """Features of the test split ({view: (R, T, d)}, window starts), cached per learner and test split."""
        path = os.path.join(self.run_dir(f), self.protocol(), "features.npz")
        if os.path.exists(path) and not refit:
            with np.load(path) as z:
                return {k[2:]: z[k] for k in z.files if k.startswith("v_")}, z["starts"]
        f = self.fit(f, refit)
        views, starts = f.represent(self.test)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, starts=starts, **{f"v_{k}": v.astype(np.float32) for k, v in views.items()})
        return views, starts

    def run(self, f, refit=False):
        """Feature learner (instance, name or spec) -> results of every detector (and the probes) on its features."""
        f = make_learner(f) if isinstance(f, (str, dict)) else f
        views, starts = self.features(f, refit)
        i, data, base = self.ds.info, self.test, os.path.join(self.run_dir(f), self.protocol())
        meta = dict(dataset=i.id, regime=i.regime, signal=i.signal, nuisance=i.nuisance, corruption=i.corruption,
                    features=f.name, features_label=f.label(), W=f.window, gap=f.params.get("gap", 0), seed=f.params.get("seed", 0),
                    learner_params=json.dumps(f.identity(), sort_keys=True, default=str))
        dfs = []
        for det in self.detectors + (["probe"] if self.probes else []):
            stem = os.path.join(base, "probe" if det == "probe" else f"{det.name}_{_hash(det.params)}")
            cached = read_result(stem) if not refit else None
            if cached is not None:
                dfs.append(cached); continue
            t0, rows = time.time(), []
            for view, Z in views.items():
                Z = Z.astype(np.float64)
                if det == "probe":
                    res = run_probes(Z, majority_labels(data["S_s"], starts, f.window, i.K), seed=0)
                    name, label, params = "probe", "probe", {}
                else:
                    res = score(det.detect(Z, i.K, seed=0), starts, f.window, data, i.K, i.margin)
                    name, label, params = det.name, det.label(), det.params
                rows += [dict(meta, view=view, detector=name, detector_label=label, detector_params=json.dumps(params, sort_keys=True),
                              metric=k, value=float(v)) for k, v in res.items()]
            df = pd.DataFrame(rows); df.to_csv(stem + ".csv", index=False); dfs.append(df)
            self.log(f"[{f.name} x {getattr(det, 'name', det)}] {time.time() - t0:.0f}s")
        return pd.concat(dfs, ignore_index=True)

    def decisions(self, f, detectors=None, view="features"):
        """The detectors' actual outputs on a learner's test features, for inspection and plots (not cached):
        {detector label: dict(labels, cp_score)}, plus the window starts."""
        f = make_learner(f) if isinstance(f, (str, dict)) else f
        views, starts = self.features(f)
        dets = [make_detector(d) if isinstance(d, (str, dict)) else d for d in (detectors or self.detectors)]
        return {d.label(): d.detect(views[view].astype(np.float64), self.ds.K, seed=0) for d in dets}, starts

    def compare(self, learners, detector="hmm", metric="acc"):
        """Run several feature learners; one table under one detector, rows = learner / window / gap / seed."""
        df = pd.concat([self.run(f) for f in learners], ignore_index=True)
        return table(df, metric, detector, by=("features", "W", "gap", "seed"), columns=None)[0]
