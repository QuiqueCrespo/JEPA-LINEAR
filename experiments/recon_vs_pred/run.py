"""Audit (controls only), local pilot or a full local evaluation of the experiment.

    python experiments/recon_vs_pred/run.py audit            # oracle / raw / untrained controls on all 5 conditions
    python experiments/recon_vs_pred/run.py pilot            # N0, F2, S2 x 6 learners x seed 0 x 3k steps (exploratory)
    python experiments/recon_vs_pred/run.py hpc <runs dir>   # score synced HPC runs (20k steps, 3 seeds): the confirmatory test
Audit and pilot use the pilot evaluation series (PILOT_EVAL_SEED); hpc uses the fresh series (EVAL_SEED).
Rows go to cpbench_results/recon_vs_pred_<mode>.csv; verdicts are printed."""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [HERE, os.path.dirname(os.path.dirname(HERE))]  # this folder and the repo root (cpbench)
import cpbench  # noqa: F401  (before torch: OpenMP)
import pandas as pd

from design import CONDITIONS, PILOT, PILOT_EVAL_SEED
from experiment import evaluate, verdicts

if __name__ == "__main__":                       # DataLoader workers re-import this file on macOS
    mode = sys.argv[1]
    if mode == "audit":
        df = evaluate(CONDITIONS, 0, [0], "cpbench_runs_recon_vs_pred_pilot", learners=(), eval_seed=PILOT_EVAL_SEED)
    elif mode == "pilot":
        df = evaluate(PILOT, 3000, [0], "cpbench_runs_recon_vs_pred_pilot", eval_seed=PILOT_EVAL_SEED)
    elif mode == "hpc":
        df = evaluate(CONDITIONS, 20000, [0, 1, 2], sys.argv[2], load_only=True, diagnostics="--diagnostics" in sys.argv)
    else:
        raise SystemExit(__doc__)
    os.makedirs("cpbench_results", exist_ok=True)
    df.to_csv(f"cpbench_results/recon_vs_pred_{mode}.csv", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
    print(verdicts(df, confirmatory=mode == "hpc").to_string(index=False))
