"""Change-point and regime-clustering detectors, applied on top of any feature learner's output.

    from cpbench.detectors import make_detector, DETECTORS, STANDARD
    det = make_detector("hmm", k=8);  det.detect(Z, K)  ->  {"labels": ..., "cp_score": ...}
"""
from .base import Detector, DETECTORS, register_detector, make_detector, reduce
from . import clustering, changepoint, dmd  # noqa: F401  (registration)

STANDARD = ("gmm", "hmm", "arhmm", "temporal_excess", "contrast", "mmd", "bocpd")   # the default detector set
