"""
Hidden-transition regime benchmark (cross-frequency phase coupling).

Construction
  S_w      sticky K-state Markov chain over windows (p_stay 0.9).
  x_w      hidden state in R^{2P} (P rotation planes):  x_w = rho Rot(alpha_{S_w}) x_{w-1} + sqrt(1-rho^2) eps.
           x_w ~ N(0, I) in every regime and is independent of S_w, so ONE window carries no regime information.
           The regime lives only in the transition x_{w-1} -> x_w.  alpha in {0, pi/4, pi/2, -pi/4}: the doubled
           angles {0, pi/2, pi, -pi/2} stay distinct, so a training gap of one window keeps regimes distinguishable.
  coding   each coordinate x_i sets the phase offset of an oscillator triplet with f_c = f_a + f_b:
           theta_c = theta_a + theta_b + psi_i,  psi_i = 2 arctan(x_i).  theta_a, theta_b diffuse randomly, so power
           spectra, means and all (cross-)covariances are independent of x; only third-order (bispectral) structure
           carries it.
  nuisance free oscillators, slow AR decoys visible in channel means, log-volatility, gain, white noise,
           outliers, channel dropout; random mixing into D channels.  No nonlinearity after mixing: a pointwise
           nonlinearity would leak the phase coupling into second-order statistics.
"""
import numpy as np
from dataclasses import dataclass, field


@dataclass
class HiddenSpec:
    K: int = 4
    p_stay: float = 0.9
    W: int = 64                       # samples per window
    D: int = 64                       # channels
    P: int = 2                        # rotation planes -> x in R^4
    rho: float = 0.95
    alphas: tuple = (0.0, np.pi / 4, np.pi / 2, -np.pi / 4)
    # triplets (f_a, f_b) in cycles/sample; f_c = f_a + f_b
    triplets: tuple = ((0.050, 0.080), (0.060, 0.110), (0.070, 0.150), (0.090, 0.190))
    free_freqs: tuple = (0.035, 0.100, 0.240, 0.310, 0.370, 0.420)
    phase_diff: float = 0.06          # phase random-walk variance per sample (coherence ~ 1/phase_diff samples)
    n_slow: int = 3                   # slow decoys (window-level AR, visible in channel means)
    slow_lam: float = 0.97
    slow_scale: float = 1.0
    vol_lam: float = 0.95
    vol_scale: float = 0.5
    obs_noise: float = 0.3
    outliers: float = 0.01
    dropout: float = 0.05
    seed: int = 0


def make_mixing(spec):
    r = np.random.default_rng([spec.seed, 99])
    n_osc = 3 * len(spec.triplets) + len(spec.free_freqs)
    return dict(M=r.standard_normal((spec.D, n_osc)) / np.sqrt(n_osc),
                L=r.standard_normal((spec.D, spec.n_slow)) / np.sqrt(spec.n_slow),
                amp=r.uniform(0.7, 1.3, n_osc))


def simulate_hidden(spec, R, T, rng, mix=None, return_obs=True):
    """R series x T windows.  Returns dict with S (R,T), x (R,T,2P), obs (R,T,W,D) float32."""
    mix = mix or make_mixing(spec)
    K, W, D = spec.K, spec.W, spec.D
    S = np.empty((R, T), int); s = rng.integers(0, K, R)
    for t in range(T):
        sw = rng.random(R) > spec.p_stay
        s = np.where(sw, (s + rng.integers(1, K, R)) % K, s); S[:, t] = s
    al = np.array(spec.alphas)
    x = np.empty((R, T, 2 * spec.P)); cur = rng.standard_normal((R, 2 * spec.P))
    for t in range(T):
        c, sn = np.cos(al[S[:, t]]), np.sin(al[S[:, t]])
        new = np.empty_like(cur)
        for p in range(spec.P):
            u, v = cur[:, 2 * p], cur[:, 2 * p + 1]
            new[:, 2 * p], new[:, 2 * p + 1] = c * u - sn * v, sn * u + c * v
        cur = spec.rho * new + np.sqrt(1 - spec.rho ** 2) * rng.standard_normal(cur.shape)
        x[:, t] = cur
    slow = np.empty((R, T, spec.n_slow)); z = rng.standard_normal((R, spec.n_slow))
    h = np.empty((R, T)); hv = rng.standard_normal(R)
    for t in range(T):
        z = spec.slow_lam * z + np.sqrt(1 - spec.slow_lam ** 2) * rng.standard_normal(z.shape); slow[:, t] = z
        hv = spec.vol_lam * hv + np.sqrt(1 - spec.vol_lam ** 2) * rng.standard_normal(R); h[:, t] = hv
    out = dict(S=S, x=x, slow=slow, logvol=h)
    if not return_obs:
        return out
    # oscillators at sample level, continuous across windows (phase random walk), coupling offset per window
    n_samp = T * W; tau = np.arange(n_samp)
    psi = 2 * np.arctan(x)                                                    # (R, T, 2P)
    psi_s = np.repeat(psi, W, axis=1)                                         # (R, n_samp, 2P)
    obs = np.zeros((R, n_samp, D), np.float32)
    k = 0
    def walk():
        return np.cumsum(np.sqrt(spec.phase_diff) * rng.standard_normal((R, n_samp)), 1) + rng.uniform(0, 2 * np.pi, (R, 1))
    for i, (fa, fb) in enumerate(spec.triplets):
        ta = 2 * np.pi * fa * tau + walk(); tb = 2 * np.pi * fb * tau + walk(); tc = ta + tb + psi_s[..., i]
        for th in (ta, tb, tc):
            obs += (mix["amp"][k] * np.cos(th))[..., None].astype(np.float32) * mix["M"][:, k].astype(np.float32); k += 1
    for f in spec.free_freqs:
        th = 2 * np.pi * f * tau + walk()
        obs += (mix["amp"][k] * np.cos(th))[..., None].astype(np.float32) * mix["M"][:, k].astype(np.float32); k += 1
    obs = obs.reshape(R, T, W, D)
    obs *= np.exp(spec.vol_scale * h)[..., None, None].astype(np.float32)
    obs += (spec.slow_scale * slow @ mix["L"].T)[:, :, None, :].astype(np.float32)
    obs += spec.obs_noise * rng.standard_normal(obs.shape).astype(np.float32)
    hit = rng.random(obs.shape) < spec.outliers
    obs[hit] += 5 * rng.standard_normal(hit.sum()).astype(np.float32)
    obs *= (rng.random((R, T, 1, D)) > spec.dropout).astype(np.float32)
    out["obs"] = obs
    return out
