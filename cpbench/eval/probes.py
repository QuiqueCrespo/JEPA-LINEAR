"""Supervised diagnostics: is the regime in a representation at all?  Never headline numbers."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score


def _flat(Z):
    return Z.reshape(-1, Z.shape[-1])


def probes(Z, S, seed=0):
    """Supervised diagnostics on held-out series: linear probe on single windows, MLP on window pairs."""
    R = Z.shape[0]; h = R // 2
    out = {}
    sc = StandardScaler().fit(_flat(Z[:h]))
    lin = LogisticRegression(max_iter=3000, C=0.1).fit(sc.transform(_flat(Z[:h])), S[:h].ravel())
    out["probe_linear"] = balanced_accuracy_score(S[h:].ravel(), lin.predict(sc.transform(_flat(Z[h:]))))
    P = np.concatenate([Z[:, :-1], Z[:, 1:]], -1); y = S[:, 1:]
    sc2 = StandardScaler().fit(_flat(P[:h]))
    mlp = MLPClassifier((128, 128), max_iter=200, early_stopping=True, random_state=seed).fit(sc2.transform(_flat(P[:h])), y[:h].ravel())
    out["probe_mlp_pair"] = balanced_accuracy_score(y[h:].ravel(), mlp.predict(sc2.transform(_flat(P[h:]))))
    return out
