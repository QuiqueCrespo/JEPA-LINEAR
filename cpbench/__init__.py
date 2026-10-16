"""cpbench: synthetic benchmark for latent change-point detection and regime clustering.

Datasets (cpbench.data) -> feature learners (cpbench.features) -> detectors (cpbench.detectors), composed and scored by
Benchmark.

    from cpbench import Benchmark, make_learner, make_detector
"""
from .bench import Benchmark
from .features import make_learner, register_learner, FeatureLearner, TorchFeatureLearner
from .detectors import make_detector, register_detector, Detector
