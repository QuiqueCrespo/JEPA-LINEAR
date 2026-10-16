"""Supervised diagnostics: is the regime in a representation at all?  Never headline numbers."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score


def _flat(Z):
    return Z.reshape(-1, Z.shape[-1])


def probes(Z, S, seed=0):
    """Supervised diagnostics on held-out series (fit on the first half of the series, balanced accuracy on the rest):
    probe_linear      logistic regression on single windows
    probe_mlp_single  fixed MLP (64 hidden, alpha 1e-3, early stopping, 300 iterations, random_state 0) on single windows
    probe_mlp_pair    MLP on pairs of consecutive windows"""
    R = Z.shape[0]
    if R < 2:
        raise ValueError("Probes require at least two independent series")
    h = R // 2
    out = {}
    sc = StandardScaler().fit(_flat(Z[:h]))
    lin = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(_flat(Z[:h])), S[:h].ravel())
    out["probe_linear"] = balanced_accuracy_score(S[h:].ravel(), lin.predict(sc.transform(_flat(Z[h:]))))
    mlp1 = MLPClassifier(hidden_layer_sizes=(64,), alpha=1e-3, early_stopping=True, max_iter=300, random_state=0)
    mlp1.fit(sc.transform(_flat(Z[:h])), S[:h].ravel())
    out["probe_mlp_single"] = balanced_accuracy_score(S[h:].ravel(), mlp1.predict(sc.transform(_flat(Z[h:]))))
    P = np.concatenate([Z[:, :-1], Z[:, 1:]], -1); y = S[:, 1:]
    sc2 = StandardScaler().fit(_flat(P[:h]))
    mlp = MLPClassifier((128, 128), max_iter=200, early_stopping=True, random_state=seed).fit(sc2.transform(_flat(P[:h])), y[:h].ravel())
    out["probe_mlp_pair"] = balanced_accuracy_score(y[h:].ravel(), mlp.predict(sc2.transform(_flat(P[h:]))))
    return out
