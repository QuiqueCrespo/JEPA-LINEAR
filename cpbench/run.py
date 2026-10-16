"""Command line for one (dataset, feature learner, seed): fit, then every detector on its features.

  python -m cpbench.run dataset=path/to/dataset.yaml features=jepa_shared seed=0 hp.window=32 hp.gap=32 hp.steps=20000
  python -m cpbench.run dataset=path/to/dataset.yaml features=raw_stats hp.window=16 detectors=[hmm,bocpd]
  python -m cpbench.run --list

Options (dotlist): dataset (YAML), features (a registered feature learner), hp.<name>=<value> (its parameters), seed,
detectors (default: the standard set), out (default cpbench_runs), quick (bool), plugins=[module,...] (modules that
register custom feature learners or detectors; imported first).
"""
import importlib, os, sys
from omegaconf import OmegaConf

from cpbench.bench import Benchmark
from cpbench.detectors import DETECTORS, STANDARD
from cpbench.features import LEARNERS, make_learner


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--list" in argv:
        print("trainable feature learners:", ", ".join(LEARNERS.available(lambda c: c.trainable)))
        print("fixed feature learners:", ", ".join(LEARNERS.available(lambda c: not c.trainable)))
        print("detectors:", ", ".join(DETECTORS.available())); return
    cfg = OmegaConf.merge(OmegaConf.create(dict(dataset=None, features=None, seed=0, out="cpbench_runs", quick=False,
                                                hp={}, detectors=list(STANDARD), plugins=[])), OmegaConf.from_dotlist(argv))
    for p in cfg.plugins:
        importlib.import_module(p)
    hp = OmegaConf.to_container(cfg.hp)
    trainable = LEARNERS.entries[cfg.features][0].trainable
    f = make_learner(cfg.features, **hp, **({"seed": cfg.seed} if trainable else {}))
    bench = Benchmark(cfg.dataset, detectors=list(cfg.detectors), out=cfg.out, quick=cfg.quick)
    df = bench.run(f)
    head = df[(df.view == "features") & df.metric.isin(["acc", "cp_ap"])]
    print(head.pivot_table(index="detector", columns="metric", values="value").round(3).to_string())
    print("results:", bench.run_dir(f))


if __name__ == "__main__":
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    main()
