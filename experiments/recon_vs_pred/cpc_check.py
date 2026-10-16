"""Label-free CPC trainability check on F2 (pilot-informed amendment, before HPC training).

    python experiments/recon_vs_pred/cpc_check.py <out dir>     # on a GPU node

Grid: temperature {0.1, 0.5} x lr {3e-4, 1e-4}; CPC, F2, seed 0, 3k steps, otherwise the common settings.
Selection reads ONLY the training-loss ratio (loss at 3k / loss at 1k, as in the collapse gate) and the effective
rank of the features of the pilot validation observations (pilot evaluation seed, not the fresh HPC series).  Periodic validation is replaced by a label-free
eff_rank so no probe or detector is ever fitted: regime labels are never read.
A setting "trains" if loss ratio < 0.9 and eff_rank > 2."""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(HERE))]
import cpbench  # noqa: F401  (before torch: OpenMP)
import numpy as np
import pandas as pd

from cpbench import Benchmark, make_learner
from cpbench.data import Dataset
from design import COMMON, PILOT_EVAL_SEED, dataset
from experiment import _eff_rank, _loss_drop

STEPS, GRID = 3000, [(t, lr) for t in (0.1, 0.5) for lr in (3e-4, 1e-4)]


def label_free_validate(f):
    def validate(val, K, margin=32):
        f.net.eval()
        Z = f.represent(dict(obs=val["obs"]))[0]["features"].astype(np.float64)
        f.net.train()
        return dict(eff_rank=_eff_rank(Z))
    return validate


if __name__ == "__main__":
    out = sys.argv[1]
    ds = Dataset(dataset("F2", eval_seed=PILOT_EVAL_SEED))   # never touches the fresh HPC evaluation series
    bench = Benchmark(ds, detectors=[], probes=False, out=out, log=print)
    rows = []
    for temperature, lr in GRID:
        p = {**COMMON, "lr": lr}
        f = make_learner("cpc", **p, temperature=temperature, steps=STEPS, warmup=1000, seed=0, workers=6,
                         eval_every=1000, ckpt_every=1000, eval_detectors=[])
        f.validate = label_free_validate(f)
        f = bench.fit(f)
        val = ds.val_set()
        Z = f.represent(dict(obs=val["obs"]))[0]["features"].astype(np.float64)
        ratio, rank = _loss_drop(bench.run_dir(f), STEPS), _eff_rank(Z)
        rows.append(dict(temperature=temperature, lr=lr, loss_ratio=ratio, eff_rank=rank,
                         trains=bool(ratio < 0.9 and rank > 2), run_dir=bench.run_dir(f)))
        print(rows[-1], flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out, "cpc_check.csv"), index=False)
    print(df.to_string(index=False))
