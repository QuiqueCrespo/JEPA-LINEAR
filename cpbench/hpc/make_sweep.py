"""Expand a sweep YAML into a SLURM job array (one array element = one dataset x feature learner x seed: fit it, then
run every detector on its features).

  python -m cpbench.hpc.make_sweep cpbench/configs/sweep_first.yaml          # writes sweeps/<name>/
  sbatch sweeps/<name>/job.sbatch

Sweep YAML keys: name, out, datasets (a grid YAML path, an inline grid of component slots (data/grid.py), or a list
of grids and / or plain dataset configs),
learners + seeds + hparams {hp.<name>: [values]} (trainable feature learners; full product; window and gap are
hyperparameters, swept, never set from knowledge of the regimes) + overrides (dotlist for every learnt job),
baselines + baseline_hparams (fixed feature learners, e.g. raw_stats over hp.window), detectors (run on every
learner's features; default: the standard set), plugins (modules registering custom learners / detectors),
slurm {partition, gres, time, cpus, mem, max_parallel, setup, workers, signal_lead}.
Every array element runs  python -m cpbench.run <line>  (fit + evaluate through the common method interface).

Long training: signal_lead seconds (default 600) before the time limit SLURM sends SIGUSR1; the job script forwards it
to Python, which writes its checkpoint and exits with code 3, and the script requeues the array task.  The requeued
task resumes from the checkpoint (and finished tasks only reload their model and cached results).  Progress:
  python -m cpbench.hpc.status sweeps/<name>
"""
import os, sys
from omegaconf import OmegaConf
import itertools

from cpbench.data import Dataset, expand as expand_grid
from cpbench.data.grid import is_grid

SBATCH = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --array=1-{n}%{max_parallel}
#SBATCH --partition={partition}
#SBATCH --gres={gres}
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
#SBATCH --time={time}
#SBATCH --output={logdir}/%A_%a.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --signal=B:USR1@{signal_lead}
set -uo pipefail
set -f                      # never glob-expand job arguments
{setup}
cd {repo}
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export CPBENCH_CACHE={cache}
LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {manifest})
echo "== $(date) task ${{SLURM_ARRAY_TASK_ID}} (restart ${{SLURM_RESTART_COUNT:-0}}): $LINE"
python -m cpbench.run $LINE &
PID=$!
trap 'echo "== $(date) time limit near: checkpointing"; kill -USR1 $PID' USR1
wait $PID; CODE=$?
while kill -0 $PID 2>/dev/null; do wait $PID; CODE=$?; done   # wait again after the trap interrupted it
if [ "$CODE" -eq 3 ]; then
    echo "== requeue"; scontrol requeue "${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}"
fi
exit $CODE
"""


def _product(hp):
    hp = hp or {}
    return [dict(zip(hp.keys(), v)) for v in itertools.product(*hp.values())] if hp else [{}]


def write_datasets(sw, d):
    """Expand the dataset grid and write one YAML per dataset (named by its construction digest)."""
    os.makedirs(os.path.join(d, "datasets"), exist_ok=True)
    paths = []
    spec = sw.datasets if isinstance(sw.datasets, str) else OmegaConf.to_container(sw.datasets)
    items = spec if isinstance(spec, list) else [spec]
    cfgs = [c for it in items for c in (expand_grid(it) if isinstance(it, str) or is_grid(it) else [it])]
    for cfg in cfgs:
        ds = Dataset(cfg)
        p = os.path.abspath(os.path.join(d, "datasets", f"{ds.info.digest}.yaml"))
        OmegaConf.save(OmegaConf.create(ds.config()), p)
        paths.append(p)
    return paths


def expand(sw, dataset_paths):
    plug = [f"plugins=[{','.join(sw.plugins)}]"] if sw.get("plugins") else []
    plug += [f"detectors=[{','.join(sw.detectors)}]"] if sw.get("detectors") else []
    workers = [f"hp.workers={sw.slurm.get('workers', 6)}"]
    lines = []
    for t in dataset_paths:
        for b in sw.get("baselines", []):
            for c in _product(sw.get("baseline_hparams")):
                lines.append(" ".join([f"dataset={t}", f"features={b}", f"out={sw.out}", *[f"{k}={v}" for k, v in c.items()], *plug]))
        for m in sw.get("learners", []):
            for c in _product(sw.get("hparams")):
                for s in sw.seeds:
                    lines.append(" ".join([f"dataset={t}", f"features={m}", f"seed={s}", f"out={sw.out}",
                                           *[f"{k}={v}" for k, v in c.items()], *sw.get("overrides", []), *workers, *plug]))
    return lines


def main(path, dry_run=False):
    sw = OmegaConf.load(path)
    if dry_run:
        spec = sw.datasets if isinstance(sw.datasets, str) else OmegaConf.to_container(sw.datasets)
        items = spec if isinstance(spec, list) else [spec]
        cfgs = [c for it in items for c in (expand_grid(it) if isinstance(it, str) or is_grid(it) else [it])]
        import hashlib, json
        paths = ["dataset_" + hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest() for c in cfgs]
        lines = expand(sw, paths)
        from cpbench.features import make_learner
        identities = []
        for line in lines:
            cfg = OmegaConf.from_dotlist(line.split())
            hp = OmegaConf.to_container(cfg.hp) if "hp" in cfg else {}
            if cfg.features in sw.get("learners", []):
                hp["seed"] = cfg.seed
            f = make_learner(cfg.features, **hp)
            identities.append((cfg.dataset, f.name, f.tag(), f.params.get("seed", 0)))
        if len(identities) != len(set(identities)):
            raise ValueError("Duplicate resolved learner identities in sweep")
        trained = len(cfgs) * len(sw.get("learners", [])) * len(_product(sw.get("hparams"))) * len(sw.seeds)
        print(f"{len(lines)} distinct job identities (dataset specifications): {trained} trained + {len(lines) - trained} baseline; no files written")
        return lines
    d = os.path.join("sweeps", sw.name); os.makedirs(os.path.join(d, "logs"), exist_ok=True)
    lines = expand(sw, write_datasets(sw, d))
    manifest = os.path.abspath(os.path.join(d, "manifest.txt"))
    open(manifest, "w").write("\n".join(lines) + "\n")
    sl = sw.slurm
    txt = SBATCH.format(name=sw.name, n=len(lines), max_parallel=sl.get("max_parallel", 50), partition=sl.partition,
                        gres=sl.get("gres", "gpu:1"), cpus=sl.get("cpus", 8), mem=sl.get("mem", "32G"),
                        time=sl.get("time", "08:00:00"), signal_lead=sl.get("signal_lead", 600),
                        logdir=os.path.abspath(os.path.join(d, "logs")),
                        setup=sl.get("setup", ""), repo=os.path.abspath("."), manifest=manifest,
                        cache=os.path.abspath(sl.get("cache", "cpbench_cache")))
    open(os.path.join(d, "job.sbatch"), "w").write(txt)
    print(f"{len(lines)} jobs -> {d}/job.sbatch  (manifest {manifest})")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    main(args.path, dry_run=args.dry_run)
