"""Recoverability audit: how hard is each dataset for cheap features?

For every dataset: the fixed feature learners raw_<feature> (at several windows), oracle_clean and, where the data
expose one, oracle_hidden, each followed by every standard detector.  Results use the common schema and land in --out
like any other run (resumable); the combined table is written to --table.  A dataset is "raw-irrecoverable" when
every raw feature x detector pair is near chance while an oracle is high.

  python -m cpbench.eval.audit --grid cpbench/configs/grid_synthetic.yaml --jobs 2
  python -m cpbench.eval.audit --datasets my_dataset.yaml --windows 16 32 64
"""
import argparse, os, time, traceback
import multiprocessing as mp
import pandas as pd

from cpbench.bench import Benchmark
from cpbench.data import Dataset, expand
from cpbench.features import make_learner
from cpbench.features.raw_functions import RAW
from cpbench.results.aggregate import collect


def audit_dataset(dataset, windows=(32,), raw=None, out="cpbench_audit", series=8, samples=16000):
    t0 = time.time()
    bench = Benchmark(Dataset(dataset).with_splits(test=[series, samples]), out=out, cache_data=False, log=lambda *a: None)
    learners = [make_learner(f"raw_{f}", window=W) for W in windows for f in (raw or RAW)]
    learners += [make_learner("oracle_clean", window=W) for W in windows]
    if "hidden" in bench.test:
        learners += [make_learner("oracle_hidden", window=W) for W in windows]
    df = pd.concat([bench.run(f) for f in learners], ignore_index=True)
    print(f"audited {bench.ds.info.id} in {time.time() - t0:.0f}s", flush=True)
    return df


def _worker(args):
    try:
        audit_dataset(*args)
    except Exception:
        traceback.print_exc()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="*", default=[], help="dataset YAML files")
    ap.add_argument("--grid", default=None, help="grid YAML of component slots (e.g. cpbench/configs/grid_synthetic.yaml)")
    ap.add_argument("--windows", nargs="*", type=int, default=[32])
    ap.add_argument("--raw", nargs="*", default=None, help="raw features to audit (default: all)")
    ap.add_argument("--series", type=int, default=8)
    ap.add_argument("--samples", type=int, default=16000)
    ap.add_argument("--out", default="cpbench_audit", help="run directory (results are resumable)")
    ap.add_argument("--table", default="cpbench_results/audit.csv")
    ap.add_argument("--jobs", type=int, default=1)
    a = ap.parse_args(argv)
    datasets = list(a.datasets) + (expand(a.grid) if a.grid else [])
    jobs = [(d, a.windows, a.raw, a.out, a.series, a.samples) for d in datasets]
    if a.jobs > 1:
        with mp.get_context("spawn").Pool(a.jobs) as pool:
            list(pool.imap_unordered(_worker, jobs))
    else:
        for j in jobs:
            _worker(j)
    df = collect(a.out); os.makedirs(os.path.dirname(a.table) or ".", exist_ok=True); df.to_csv(a.table, index=False)
    print(f"wrote {len(df)} rows to {a.table}")


if __name__ == "__main__":
    main()
