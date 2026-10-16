"""Is the hidden state recoverable at all?  Known-source decoder: unmix with the true mixing matrix, complex-demodulate
each oscillator at its frequency, average the triple product z_c conj(z_a z_b) over the window -> estimate of e^{i psi}."""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from hidden_bench import HiddenSpec, simulate_hidden, make_mixing
from dataclasses import replace
for pdiff in [float(a) for a in sys.argv[1:]] or [0.06, 0.02, 0.005]:
    spec = replace(HiddenSpec(), phase_diff=pdiff); mix = make_mixing(spec)
    d = simulate_hidden(spec, 4, 1000, np.random.default_rng(1), mix=mix)
    o = d["obs"].astype(np.float64); N = o.shape[0] * o.shape[1]; o = o.reshape(N, spec.W, spec.D)
    src = o @ np.linalg.pinv(mix["M"]).T                                       # (N, W, n_osc)
    tau = np.arange(spec.W); win = np.hanning(spec.W)
    est = []
    for i, (fa, fb) in enumerate(spec.triplets):
        z = [src[:, :, 3 * i + j] * np.exp(-2j * np.pi * f * tau) for j, f in enumerate((fa, fb, fa + fb))]
        # short smoothing (8 samples) = narrowband filter around each frequency
        z = [np.apply_along_axis(lambda v: np.convolve(v, np.ones(8) / 8, "same"), 1, zz) for zz in z]
        tp = (z[2] * np.conj(z[0] * z[1]) * win).sum(1)
        est.append(np.angle(tp))
    est = np.stack(est, 1); psi = 2 * np.arctan(d["x"].reshape(N, -1))
    circ = np.abs(np.mean(np.exp(1j * (est - psi)), 0))                       # 1 = perfect, 0 = no information
    print(f"phase_diff {pdiff}: circular agreement of decoded vs true psi per coordinate {np.round(circ, 3)}")

