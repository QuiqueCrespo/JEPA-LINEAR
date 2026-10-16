"""Scoring at SAMPLE resolution, so the window length is a fair hyperparameter: window decisions are mapped back to
samples (data/windows.py) and windows that straddle a switch cost accuracy.

    score(decisions, starts, W, data, K, margin)  -> {metric: value}   for one detector's dict(labels, cp_score)
"""
from cpbench.data.windows import labels_to_samples, score_positions
from .metrics import clustering, cp_metrics


def score(decisions, starts, W, data, K, margin):
    S_s, cp_s, n = data["S_s"], data["cp_s"], data["S_s"].shape[1]
    out = {}
    if decisions.get("labels") is not None:
        out.update(clustering(S_s, labels_to_samples(decisions["labels"], starts, W, n), K))
    if decisions.get("cp_score") is not None:
        cm = cp_metrics(decisions["cp_score"], score_positions(starts, W), cp_s, margin, decisions.get("alarms"))
        out.update({f"cp_{m}": cm[m] for m in ("ap", "f1", "f1_best", "covering")})
    return out
