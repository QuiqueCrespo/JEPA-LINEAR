"""Clustering and change-point metrics.

Everything is scored at SAMPLE resolution, so the model window W is a fair hyperparameter: window outputs are
mapped back to samples (data/windows.py) and windows straddling a switch cost accuracy.
Clustering: best-permutation balanced accuracy, ARI, NMI against sample regime labels.
Change points (TCPD conventions, van den Burg & Williams 2020):
  ap          average precision of the ranked peak list; a detection counts if within `margin` samples of an
              unmatched true change point (threshold-free)
  f1_best     best F1 over thresholds (oracle threshold: an upper bound, reported as such)
  f1          F1 at the label-free default threshold median + 3 MAD of the score
  covering    segment covering of the default-threshold segmentation
"""
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, balanced_accuracy_score


def best_perm_acc(true, pred, K):
    M = np.zeros((K, max(K, pred.max() + 1)))
    np.add.at(M, (true, pred), 1)
    weights = M / np.maximum(M.sum(1, keepdims=True), 1)
    r, c = linear_sum_assignment(-weights)
    mapping = dict(zip(c, r))
    return balanced_accuracy_score(true, [mapping.get(p, -1) for p in pred])


def clustering(true, pred, K):
    true, pred = np.asarray(true).ravel(), np.asarray(pred).ravel()
    return dict(acc=best_perm_acc(true, pred, K), ari=adjusted_rand_score(true, pred),
                nmi=normalized_mutual_info_score(true, pred))


def _peaks(score, radius=1):
    """Local maxima of each series' window score within +-radius windows.  Returns list of (series, t, value)."""
    out = []
    for r, s in enumerate(score):
        for t in range(1, len(s)):
            lo, hi = max(1, t - radius), min(len(s), t + radius + 1)
            seg = s[lo:hi]
            if np.isfinite(s[t]) and s[t] >= seg.max() and s[t] > seg.min():   # flat stretches are not peaks
                out.append((r, t, s[t]))
    return out


def cp_metrics(score, positions, cp_s, margin, alarms=None):
    """Change-point metrics at sample resolution.
    score (R, T) window-level change scores; positions (T,) their sample locations (data/windows.score_positions);
    cp_s (R, n) bool true change samples; margin in samples (independent of the model window).  f1 and covering use
    the detector's own alarms (per series, window indices) when given, else peaks above a robust default threshold
    (median + 3 MAD of the score)."""
    score = np.asarray(score, float); R, n = cp_s.shape
    true = [np.flatnonzero(cp_s[r]) for r in range(R)]
    n_true = sum(len(t) for t in true)
    peaks = sorted(_peaks(score), key=lambda p: -p[2])
    used = [np.zeros(len(t), bool) for t in true]
    tp = np.zeros(len(peaks)); thr = np.zeros(len(peaks))
    for i, (r, t, v) in enumerate(peaks):
        thr[i] = v
        if len(true[r]):
            dist = np.abs(true[r] - positions[t]).astype(float); dist[used[r]] = np.inf
            j = int(np.argmin(dist))
            if dist[j] <= margin:
                used[r][j] = True; tp[i] = 1
    ctp = np.cumsum(tp); k = np.arange(1, len(peaks) + 1)
    prec, rec = ctp / np.maximum(k, 1), ctp / max(n_true, 1)
    ap = float(np.sum(np.diff(np.r_[0, rec]) * prec)) if len(peaks) else 0.0
    f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
    flat = score[np.isfinite(score)]
    t_def = np.median(flat) + 3 * np.median(np.abs(flat - np.median(flat))) * 1.4826
    n_def = int((thr > t_def).sum())
    if alarms is not None:
        det = [sorted(int(round(positions[t])) for t in a) for a in alarms]
        n_def = sum(len(d) for d in det)
        f1 = 2 * _matched(true, det, margin) / max(n_def + n_true, 1)
    else:
        det = [sorted(int(round(positions[t])) for (r2, t, v) in peaks[:n_def] if r2 == r) for r in range(R)]
        f1 = float(f1s[n_def - 1]) if n_def else 0.0
    return dict(ap=ap, f1_best=float(f1s.max()) if len(f1s) else 0.0, f1=f1,
                covering=covering(true, det, n), n_true=n_true, n_det=n_def)


def _matched(true, det, margin):
    """Number of detections matched one-to-one to a true change within margin samples (greedy, in time order)."""
    tp = 0
    for t_list, d_list in zip(true, det):
        used = np.zeros(len(t_list), bool)
        for d in d_list:
            dist = np.abs(np.asarray(t_list) - d).astype(float); dist[used] = np.inf
            if len(dist) and dist.min() <= margin:
                used[int(np.argmin(dist))] = True; tp += 1
    return tp


def _segments(bounds, n):
    b = np.unique(np.clip(np.r_[0, np.asarray(bounds, int), n], 0, n))
    return [(b[i], b[i + 1]) for i in range(len(b) - 1) if b[i + 1] > b[i]]


def covering(true_cps, det_cps, n):
    """Segment covering (TCPD) averaged over series; change-point lists in samples."""
    tot = 0.0
    for t_list, p_list in zip(true_cps, det_cps):
        St, Sp = _segments(t_list, n), _segments(p_list, n)
        c = 0.0
        for a0, a1 in St:
            best = 0.0
            for b0, b1 in Sp:
                inter = max(0, min(a1, b1) - max(a0, b0))
                if inter:
                    best = max(best, inter / (max(a1, b1) - min(a0, b0)))
            c += (a1 - a0) * best
        tot += c / n
    return tot / len(true_cps)
