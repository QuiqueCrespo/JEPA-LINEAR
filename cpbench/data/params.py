"""Regime-dependent parameters.  Any numeric parameter of a source or transform can be written as

    0.8                          the same value in every regime (and every channel)
    [0.2, 0.9]                   one value per regime (each may itself be a per-channel list)
    {base: 0.4, spread: 0.5}     K values evenly spaced in [base - spread, base + spread], independently permuted per
                                 channel (seeded by the dataset) -- a quick way to make every regime different
    {values: [a, b, c], permute: true}   the given per-regime values, permuted per channel
    {channels: [a, b, ...]}      per-channel values, the same in every regime

resolve() turns any of these into a (K, d) table; per_sample() turns the table into per-sample values (R, n, d) from
the hard regime path S, or from soft weights W (R, n, K) when the regime process has a ramp (gradual changes)."""
import numpy as np


def spread(K):
    """K evenly spaced values in [-1, 1] (0 for a single regime)."""
    return np.linspace(-1, 1, K) if K > 1 else np.zeros(1)


def resolve(value, K, d, prng):
    """Parameter spec -> (K, d) float array."""
    if isinstance(value, dict):
        if "channels" in value:
            ch = np.asarray(value["channels"], float)
            if len(ch) != d:
                raise ValueError(f"{len(ch)} per-channel values given for {d} channels")
            return np.tile(ch, (K, 1))
        if "base" in value:
            sp = value.get("spread", 0.0) * spread(K)
            return value["base"] + np.stack([prng.permutation(sp) for _ in range(d)], 1)
        vals = np.asarray(value["values"], float)
        if len(vals) != K:
            raise ValueError(f"{len(vals)} per-regime values given for K={K} regimes")
        if value.get("permute", False):
            return np.stack([prng.permutation(vals) for _ in range(d)], 1)
        return np.broadcast_to(vals.reshape(K, -1), (K, d)).astype(float).copy()
    arr = np.asarray(value, float)
    if arr.ndim == 0:
        return np.full((K, d), float(arr))
    if arr.shape[0] != K:
        raise ValueError(f"a list parameter gives one value per regime: expected {K} entries, got {arr.shape[0]}")
    return np.broadcast_to(arr.reshape(K, -1), (K, d)).astype(float).copy()


def varies(table):
    """True if a resolved (K, d) table differs between regimes."""
    return bool(np.ptp(table, axis=0).max() > 0) if len(table) > 1 else False


def per_sample(table, S, W=None):
    """(K, d) table -> (R, n, d) per-sample values (hard switches from S, or linear blends from soft weights W)."""
    return table[S] if W is None else W @ table


def soft_weights(S, K, ramp):
    """Soft regime weights (R, n, K): one-hot S smoothed by a centred moving average of `ramp` samples, so parameters
    move linearly between regimes over `ramp` samples around each switch.  None when ramp <= 1 (abrupt changes)."""
    if not ramp or ramp <= 1:
        return None
    onehot = np.eye(K)[S]
    c = np.cumsum(np.pad(onehot, ((0, 0), (ramp // 2 + 1, ramp - ramp // 2), (0, 0)), mode="edge"), axis=1)
    return (c[:, ramp:] - c[:, :-ramp])[:, : S.shape[1]] / ramp
