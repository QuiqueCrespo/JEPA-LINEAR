"""Fixed (untrained) feature learners.

  raw_<feature>    a raw-window statistic of the observations (flat, mean, stats, spectra, cov, lagcov, coskew,
                   bispec, asym): the cheap baselines.  Any detector on top gives a classical method, e.g. raw_stats x hmm.
  oracle_clean     the same statistic of the uncorrupted regime-driven signal: an upper bound
  oracle_hidden    window mean of a source's hidden state (tier E)"""
import numpy as np

from .base import FeatureLearner, register_learner, learner_preset
from .raw_functions import RAW


@register_learner("raw")
class RawFeatures(FeatureLearner):
    defaults = dict(feature="stats")

    def embed(self, windows):
        return RAW[self.feature](np.asarray(windows, np.float64))


@register_learner("oracle")
class Oracle(RawFeatures):
    """Reads ground-truth arrays of the data instead of the observations (input = clean | hidden)."""
    defaults = dict(input="clean")

    @property
    def input_key(self):
        return self.input


for _f in RAW:
    learner_preset(f"raw_{_f}", "raw", feature=_f)
learner_preset("oracle_clean", "oracle", input="clean", feature="stats")
learner_preset("oracle_hidden", "oracle", input="hidden", feature="mean")
