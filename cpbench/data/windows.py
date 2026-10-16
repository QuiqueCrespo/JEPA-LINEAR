"""Model-side windowing of sample-level series, and mapping window outputs back to samples.

Window t covers samples [starts[t], starts[t] + W).  A window-level label is spread back to the samples nearest to
that window's centre; a window-level change score for t (a change between windows t-1 and t) is located at the
midpoint of the two window centres."""
import numpy as np


def window_starts(n, W, stride):
    return np.arange(0, n - W + 1, stride)


def to_windows(obs, W, stride):
    """obs (R, n, D) -> windows (R, T, W, D) (a strided view copied to contiguous) and starts (T,)."""
    starts = window_starts(obs.shape[1], W, stride)
    idx = starts[:, None] + np.arange(W)[None]
    return obs[:, idx], starts


def centres(starts, W):
    return starts + (W - 1) / 2.0


def labels_to_samples(lab, starts, W, n):
    """Window labels (R, T) -> sample labels (R, n): each sample takes the label of the window with the nearest centre."""
    c = centres(starts, W)
    nearest = np.clip(np.searchsorted((c[1:] + c[:-1]) / 2, np.arange(n)), 0, len(c) - 1)
    return lab[:, nearest]


def score_positions(starts, W):
    """Sample position of a window-level change score for t (between windows t-1 and t)."""
    c = centres(starts, W)
    return np.r_[c[0], (c[1:] + c[:-1]) / 2]


def majority_labels(S_s, starts, W, K):
    """Majority regime inside each window (diagnostics only, e.g. supervised probes)."""
    idx = starts[:, None] + np.arange(W)[None]
    win = S_s[:, idx]                                                    # (R, T, W)
    return np.stack([(win == k).sum(-1) for k in range(K)], -1).argmax(-1)
