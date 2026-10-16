"""Feature learners: windows -> feature vectors (fixed raw statistics, oracles, self-supervised encoders).

    from cpbench.features import make_learner, LEARNERS, register_learner, FeatureLearner, TorchFeatureLearner
    f = make_learner("jepa_shared", window=32, gap=32, steps=2000)
"""
from .base import FeatureLearner, LEARNERS, register_learner, learner_preset, make_learner
from .torch import TorchFeatureLearner
from . import raw, ssl  # noqa: F401  (registration)
