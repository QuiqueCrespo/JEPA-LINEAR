"""Progress of a sweep written by make_sweep:  python -m cpbench.hpc.status sweeps/<name>

For every manifest line: done (model and test results present), training (latest logged step / total, last
validation scores), or not started.  Prints the array indices still to run, ready for
  sbatch --array=<indices> sweeps/<name>/job.sbatch
(resubmitting finished tasks is harmless: they reload their model and cached results)."""
import json, os, sys
from omegaconf import OmegaConf

from cpbench.bench import Benchmark
from cpbench.features import LEARNERS, make_learner


def task_state(line, benches):
    cfg = OmegaConf.from_dotlist(line.split())
    for p in cfg.get("plugins") or []:
        __import__(p)
    if cfg.dataset not in benches:
        benches[cfg.dataset] = Benchmark(cfg.dataset, out=cfg.get("out", "cpbench_runs"), log=lambda *a: None)
    bench = benches[cfg.dataset]
    hp = OmegaConf.to_container(cfg.get("hp", OmegaConf.create({})))
    trainable = LEARNERS.entries[cfg.features][0].trainable
    f = make_learner(cfg.features, **hp, **({"seed": cfg.get("seed", 0)} if trainable else {}))
    rd = bench.run_dir(f)
    results = os.path.join(rd, bench.protocol(), "features.npz")
    if os.path.exists(results) and (not trainable or os.path.exists(os.path.join(rd, "model.pt"))):
        return "done", f.label(), ""
    log = os.path.join(rd, "log.jsonl")
    if not os.path.exists(log):
        return "not started", f.label(), ""
    recs = [json.loads(l) for l in open(log)]
    step = max((r["step"] for r in recs), default=0)
    ev = [r for r in recs if r["kind"] == "eval"]
    last = ", ".join(f"{k} {v:.2f}" for k, v in ev[-1].items() if k.endswith("acc") or k == "probe") if ev else ""
    return "training", f.label(), f"step {step}/{f.params['steps']}  {last}"


def ranges(ix):
    """[1, 2, 3, 7, 9, 10] -> "1-3,7,9-10" (sbatch --array syntax)."""
    out, start = [], None
    for a, b in zip(ix, ix[1:] + [None]):
        start = a if start is None else start
        if b != a + 1:
            out.append(f"{start}-{a}" if a > start else str(a)); start = None
    return ",".join(out)


def main(sweep_dir):
    lines = [l.strip() for l in open(os.path.join(sweep_dir, "manifest.txt")) if l.strip()]
    benches, todo, counts = {}, [], {}
    for i, line in enumerate(lines, 1):
        state, label, info = task_state(line, benches)
        counts[state] = counts.get(state, 0) + 1
        if state != "done":
            todo.append(i)
            print(f"{i:4d} {state:12s} {label}  {info}")
    print(", ".join(f"{k}: {v}" for k, v in counts.items()), f"(of {len(lines)})")
    if todo:
        print("to run:", f"sbatch --array={ranges(todo)} {os.path.join(sweep_dir, 'job.sbatch')}")


if __name__ == "__main__":
    main(sys.argv[1])
