"""Modular synthetic datasets: regime process(es) + switching linear sources + transforms (see dataset.py); grids of
datasets from component slots (grid.py); change-type recipes in cpbench/configs/grid_synthetic.yaml.

    from cpbench.data import Dataset, expand, available, register_source, register_transform, register_regime
    ds = Dataset({"regime": {...}, "sources": [...], "transforms": [...]})
"""
from .registry import register_regime, register_source, register_transform, available, build, short
from .dataset import Dataset, load_config
from .grid import expand
