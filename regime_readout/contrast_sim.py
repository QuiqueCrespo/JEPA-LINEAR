"""
Regime identification from windowed temporal contrasts: a small simulation.

Experiment 1 (1-D benchmark)
    Y_t = B_t |G_t|  (sticky sign chain times half-normal filler; exact N(0,1) marginal)
    versus the Gaussian process with identical marginal and autocovariance.
    Excess kurtosis of the window contrast D^(m) as a function of m, against a closed form.

Experiment 2 (12-D, K = 3)
    2 regime dimensions : a 2-D standard Gaussian partitioned into three 120-degree sectors;
                          S_t picks the sector, position inside it is i.i.d. filler.
                          Marginal is exactly N(0, I_2); S_t is exactly linearly decodable.
    9 Gaussian AR(1)    : persistence from 0.5 to 0.999 (several slower than S).
    1 confound          : continuous diffusion with N(0,1) stationary law and
                          state-dependent speed (non-Gaussian transitions, no regimes).
    Mixed by a random rotation. Then:
      (a) per-source contrast kurtosis vs m
      (b) fourth-cumulant subspace vs the true regime subspace, with a Gaussian surrogate null
      (c) label recovery in the estimated subspace vs clustering in the full space
      (d) lag-consistency (Chapman-Kolmogorov) of candidate partitions
"""
import itertools
import numpy as np

rng = np.random.default_rng(7)


# ----------------------------------------------------------------------------- helpers
def contrasts(Z, m, step=None):
    """Z: (R, T, d). D_t = mean(Z[t+1..t+m]) - mean(Z[t-m+1..t]), within each block."""
    R, T, d = Z.shape
    step = step or max(1, m // 2)
    cs = np.concatenate([np.zeros((R, 1, d)), np.cumsum(Z, axis=1)], axis=1)
    t = np.arange(m, T - m + 1, step)
    D = (cs[:, t + m] - 2 * cs[:, t] + cs[:, t - m]) / m
    return D.reshape(-1, d)


def excess_kurtosis(x):
    x = x - x.mean(0)
    return (x**4).mean(0) / (x**2).mean(0) ** 2 - 3


def cum4_unfolding(D):
    """Mode-1 unfolding (d x d^3) of the fourth-cumulant tensor of D."""
    D = D - D.mean(0)
    N, d = D.shape
    W = (D[:, :, None] * D[:, None, :]).reshape(N, d * d)
    M4 = (W.T @ W / N).reshape(d, d, d, d)
    S = D.T @ D / N
    C4 = (M4 - np.einsum("ij,kl->ijkl", S, S)
             - np.einsum("ik,jl->ijkl", S, S)
             - np.einsum("il,jk->ijkl", S, S))
    return C4.reshape(d, d**3)


def cum4_subspace(D, r):
    T = cum4_unfolding(D)
    w, V = np.linalg.eigh(T @ T.T)
    order = np.argsort(w)[::-1]
    return np.sqrt(np.maximum(w[order], 0)), V[:, order[:r]], V[:, order]


def max_principal_angle_deg(U, V):
    Qu, _ = np.linalg.qr(U)
    Qv, _ = np.linalg.qr(V)
    s = np.linalg.svd(Qu.T @ Qv, compute_uv=False)
    return np.degrees(np.arccos(np.clip(s.min(), -1, 1)))


def phase_surrogate(Z, rng):
    """Gaussian-like surrogate preserving all auto- and cross-covariances (common random phases)."""
    R, T, d = Z.shape
    F = np.fft.rfft(Z, axis=1)
    ph = np.exp(2j * np.pi * rng.random((R, F.shape[1], 1)))
    ph[:, 0] = 1
    return np.fft.irfft(F * ph, n=T, axis=1)


def kmeans(X, K, rng, restarts=10, iters=100):
    best = (np.inf, None, None)
    for _ in range(restarts):
        c = X[rng.choice(len(X), K, replace=False)]
        for _ in range(iters):
            lab = ((X[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
            new = np.array([X[lab == k].mean(0) if (lab == k).any() else c[k] for k in range(K)])
            if np.allclose(new, c):
                break
            c = new
        cost = ((X - c[lab]) ** 2).sum()
        if cost < best[0]:
            best = (cost, lab, c)
    return best[1], best[2]


def best_perm_accuracy(lab, truth, K):
    return max(np.mean(np.array(p)[lab] == truth) for p in itertools.permutations(range(K)))


def ck_discrepancy(lab, K, lag):
    """max |P_hat(lag) - P_hat(1)^lag| for a label array (R, T)."""
    def P(l):
        a, b = lab[:, :-l].ravel(), lab[:, l:].ravel()
        M = np.zeros((K, K))
        np.add.at(M, (a, b), 1)
        return M / M.sum(1, keepdims=True)
    return np.abs(P(lag) - np.linalg.matrix_power(P(1), lag)).max()


# ----------------------------------------------------------------------------- simulators
def sticky_chain(R, T, K, q, rng):
    S = np.empty((R, T), dtype=int)
    s = rng.integers(0, K, R)
    for t in range(T):
        sw = rng.random(R) < q
        s = np.where(sw, (s + rng.integers(1, K, R)) % K, s)
        S[:, t] = s
    return S


def ar1(R, T, rho, rng):
    rho = np.atleast_1d(rho)
    X = np.empty((R, T, len(rho)))
    x = rng.standard_normal((R, len(rho)))
    sd = np.sqrt(1 - rho**2)
    for t in range(T):
        x = rho * x + sd * rng.standard_normal(x.shape)
        X[:, t] = x
    return X


def speed_diffusion(R, T, rng, c=0.02, a=0.1, sub=20):
    """dX = b(X) dt + sigma(X) dW with sigma^2 = c (a + x^2) and b chosen so the
    stationary law is N(0,1):  b = (sigma^2)'/2 - x sigma^2 / 2."""
    X = np.empty((R, T))
    x = rng.standard_normal(R)
    dt = 1.0 / sub
    for t in range(T):
        for _ in range(sub):
            s2 = c * (a + x**2)
            b = c * x - 0.5 * x * s2
            x = x + b * dt + np.sqrt(s2 * dt) * rng.standard_normal(R)
        X[:, t] = x
    return X


# ============================================================================= Experiment 1
print("=" * 78)
print("EXPERIMENT 1: Y = B|G| vs second-order-matched Gaussian process")
R, T, q = 400, 5000, 0.01
lam = 1 - 2 * q
B = 2 * sticky_chain(R, T, 2, q, rng) - 1
Y = (B * np.abs(rng.standard_normal((R, T))))[:, :, None]
U = ar1(R, T, lam, rng)
Cg = np.sqrt(2 / np.pi) * U + np.sqrt(1 - 2 / np.pi) * rng.standard_normal((R, T, 1))

print(f"marginal excess kurtosis   Y: {excess_kurtosis(Y.reshape(-1,1))[0]:+.3f}   "
      f"Gaussian twin: {excess_kurtosis(Cg.reshape(-1,1))[0]:+.3f}")
for tau in (1, 10, 50):
    cy = np.mean(Y[:, :-tau] * Y[:, tau:]); cc = np.mean(Cg[:, :-tau] * Cg[:, tau:])
    print(f"lag-{tau:<3d} covariance  Y: {cy:.4f}   twin: {cc:.4f}   theory: {2/np.pi*lam**tau:.4f}")

sig2, Delta = 1 - 2 / np.pi, 2 * np.sqrt(2 / np.pi)


def approx_kurt(m):
    """Single-switch approximation (valid only while 2 m q << 1)."""
    u = np.concatenate([np.arange(1, m) / m, [1.0], np.arange(1, m) / m])
    Es2, Es4 = q * Delta**2 * (u**2).sum(), q * Delta**4 * (u**4).sum()
    return (Es4 - 3 * Es2**2) / (Es2 + 2 * sig2 / m) ** 2


def exact_kurt(m):
    """Exact excess kurtosis of D^(m) for Y = B|G|.
    D = (mu/m) W + n,  W = sum_j c_j B_j (c = -1 on the back window, +1 on the front),
    n = (1/m) sum_j c_j B_j xi_j with xi = |G| - mu.  W's law by dynamic programming."""
    mu = np.sqrt(2 / np.pi)
    m3 = mu * (4 / np.pi - 1)                       # third central moment of |G|
    k4 = (3 - 2 * mu**2 - 3 * mu**4) - 3 * sig2**2  # fourth cumulant of |G|
    n = 2 * m
    dist = np.zeros((2, 2 * n + 1))                 # dist[b, w + n]
    c = np.r_[-np.ones(m, int), np.ones(m, int)]
    for b, sgn in ((0, -1), (1, 1)):
        dist[b, n + c[0] * sgn] = 0.5
    for j in range(1, n):
        new = np.zeros_like(dist)
        for b, sgn in ((0, -1), (1, 1)):
            inc = c[j] * sgn
            src = (1 - q) * dist[b] + q * dist[1 - b]
            new[b] = np.roll(src, inc)
        dist = new
    w = np.arange(-n, n + 1)
    pw = dist.sum(0)
    EW2, EW4 = (pw * w**2).sum(), (pw * w**4).sum()
    Es2, Es4 = mu**2 * EW2 / m**2, mu**4 * EW4 / m**4
    vn = 2 * sig2 / m
    En4 = 3 * vn**2 + 2 * m * k4 / m**4
    Esn3 = mu * m3 * EW2 / m**4
    return (Es4 + 6 * Es2 * vn + 4 * Esn3 + En4) / (Es2 + vn) ** 2 - 3


print(f"\npredicted peak window m* = 3 sigma / (Delta sqrt(q)) = {3*np.sqrt(sig2)/(Delta*np.sqrt(q)):.1f}")
print(f"{'m':>4} {'Y sim':>8} {'Y exact':>8} {'1-switch':>9} {'twin sim':>9}")
for m in (1, 2, 4, 6, 8, 11, 16, 24, 32, 48, 64):
    print(f"{m:4d} {excess_kurtosis(contrasts(Y, m, 1))[0]:8.3f} {exact_kurt(m):8.3f} "
          f"{approx_kurt(m):9.3f} {excess_kurtosis(contrasts(Cg, m, 1))[0]:9.3f}")
grid = np.arange(1, 80)
print("exact-kurtosis peak at m =", grid[np.argmax([exact_kurt(int(g)) for g in grid])])

# ============================================================================= Experiment 2
print("\n" + "=" * 78)
print("EXPERIMENT 2: d = 12, K = 3, slow Gaussian factors + diffusion confound")
R, T, K, q, d = 400, 5000, 3, 0.01, 12
S = sticky_chain(R, T, K, q, rng)
phi = 2 * np.pi * S / K + rng.uniform(-np.pi / K, np.pi / K, (R, T))
rad = np.sqrt(-2 * np.log(rng.random((R, T))))
reg = np.stack([rad * np.cos(phi), rad * np.sin(phi)], -1)
rhos = np.array([0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.998, 0.999])
gauss = ar1(R, T, rhos, rng)
diff = speed_diffusion(R, T, rng)[:, :, None]
src = np.concatenate([reg, gauss, diff], -1)
names = ["S-1", "S-2"] + [f"AR {r}" for r in rhos] + ["diffusion"]
Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
Z = src @ Q.T
V_true, v_diff = Q[:, :2], Q[:, [11]]

print(f"S eigenvalue of P: {1 - 1.5*q:.3f}  (AR factors at 0.99 and above are more persistent)")
print("marginal excess kurtosis per source:", np.round(excess_kurtosis(src.reshape(-1, d)), 3))
print("max |Cov(Z) - I|:", np.abs(np.cov(Z.reshape(-1, d).T) - np.eye(d)).max().round(3))
acc = np.mean(np.argmax(reg @ np.array([[np.cos(2*np.pi*k/K), np.sin(2*np.pi*k/K)] for k in range(K)]).T, -1) == S)
print(f"linear (argmax) decoding accuracy of S from Z: {acc:.4f}")

ms = (1, 2, 4, 8, 12, 16, 24, 32, 64)
print("\n(a) contrast excess kurtosis by source (source coordinates, for diagnosis only)")
print(f"{'m':>4} " + " ".join(f"{n:>9}" for n in ["S-1", "S-2", "AR 0.5", "AR 0.99", "AR 0.999", "diffusion"])
      + f" {'var D: S-1':>11} {'var D: diff':>12}")
for m in ms:
    Dm = contrasts(src, m)
    k, v = excess_kurtosis(Dm), Dm.var(0)
    print(f"{m:4d} " + " ".join(f"{k[i]:9.3f}" for i in (0, 1, 2, 7, 10, 11)) + f" {v[0]:11.4f} {v[11]:12.5f}")


def inv_sqrt_and_sqrt(Sig):
    w, V = np.linalg.eigh(Sig)
    return V @ np.diag(w**-0.5) @ V.T, V @ np.diag(w**0.5) @ V.T


def shuffle_time(Z, rng):
    R_, T_, d_ = Z.shape
    return Z.reshape(-1, d_)[rng.permutation(R_ * T_)].reshape(R_, T_, d_)


Zs = phase_surrogate(Z, rng)
Zsh = shuffle_time(Z, rng)
fit, rep = Z[0::2], Z[1::2]

print("\n(b1) fourth-cumulant subspace, RAW contrasts (odd blocks fit, even blocks replicate)")
print(f"{'m':>4} {'sv1':>7} {'sv2':>7} {'sv3':>7} {'sv4':>7} {'surr sv1':>9} {'shuf sv1':>9} "
      f"{'angle to S':>11} {'replicate':>10} {'|cos| v3~diff':>14}")
est = {}
for m in ms:
    sv, V2, Vall = cum4_subspace(contrasts(fit, m), 2)
    _, V2r, _ = cum4_subspace(contrasts(rep, m), 2)
    svn, _, _ = cum4_subspace(contrasts(Zs[0::2], m), 2)
    svs, _, _ = cum4_subspace(contrasts(Zsh[0::2], m), 2)
    est[m] = V2
    print(f"{m:4d} {sv[0]:7.4f} {sv[1]:7.4f} {sv[2]:7.4f} {sv[3]:7.4f} {svn[0]:9.4f} {svs[0]:9.4f} "
          f"{max_principal_angle_deg(V2, V_true):11.2f} {max_principal_angle_deg(V2, V2r):10.2f} "
          f"{abs((Vall[:, 2] @ v_diff).item()):14.3f}")

print("\n(b2) same, STANDARDISED contrasts (D whitened by its own covariance at each m)")
print("     share of each leading direction lying in the true regime plane / along the diffusion")
print(f"{'m':>4} {'sv1':>7} {'sv2':>7} {'sv3':>7} {'sv4':>7} {'surr sv1':>9} "
      f"{'v1 S|diff':>11} {'v2 S|diff':>11} {'v3 S|diff':>11} {'top-2 angle to S':>17}")
for m in ms:
    D = contrasts(fit, m)
    Wm, Wp = inv_sqrt_and_sqrt(np.cov(D.T))
    sv, V2w, Vall = cum4_subspace(D @ Wm, 2)
    Ds = contrasts(Zs[0::2], m)
    svn, _, _ = cum4_subspace(Ds @ inv_sqrt_and_sqrt(np.cov(Ds.T))[0], 2)
    back = Wp @ Vall[:, :3]
    back /= np.linalg.norm(back, axis=0)
    shS = ((V_true.T @ back) ** 2).sum(0)
    shD = ((v_diff.T @ back) ** 2).sum(0)
    print(f"{m:4d} {sv[0]:7.3f} {sv[1]:7.3f} {sv[2]:7.3f} {sv[3]:7.3f} {svn[0]:9.3f} "
          + " ".join(f"{shS[i]:5.2f}|{shD[i]:4.2f} " for i in range(3))
          + f"{max_principal_angle_deg(back[:, :2], V_true):17.2f}")

print("\n(c) label recovery (non-overlapping window means, k-means with K = 3)")
m_star, w = 12, 8
V2 = est[m_star]


def window_means(X, w):
    R_, T_, d_ = X.shape
    return X[:, : T_ // w * w].reshape(R_, T_ // w, w, d_).mean(2)


Sw = S[:, : T // w * w].reshape(R, T // w, w)
truth_w = np.array([[np.bincount(x, minlength=K).argmax() for x in row] for row in Sw]).ravel()
sub = rng.choice(len(truth_w), 40000, replace=False)
WM = window_means(Z @ V2, w)
lab_sub, cen = kmeans(WM.reshape(-1, 2)[sub], K, rng)
print(f"  estimated 2-D subspace, window means : {best_perm_accuracy(lab_sub, truth_w[sub], K):.4f}")
lab_full, _ = kmeans(window_means(Z, w).reshape(-1, d)[sub], K, rng)
print(f"  full 12-D space, window means        : {best_perm_accuracy(lab_full, truth_w[sub], K):.4f}")
lab_one, _ = kmeans((Z @ V2).reshape(-1, 2)[rng.choice(R * T, 40000, replace=False)], K, rng)
per_sample = np.argmax((Z @ V2) @ cen.T, -1)
print(f"  per-sample argmax on window centroids: {best_perm_accuracy(per_sample.ravel(), S.ravel(), K):.4f}")
for name, arr in (("single samples", (Z @ V2).reshape(-1, 2)), ("window means  ", WM.reshape(-1, 2))):
    bc = np.histogram(arr @ (cen[0] - cen[1]) / np.linalg.norm(cen[0] - cen[1]), bins=30)[0]
    print(f"  histogram along centroid 0-1 axis, {name} (30 bins, per mille):")
    print("   ", " ".join(f"{int(1000*x/bc.sum()):d}" for x in bc))

# ----------------------------------------------------------------------------- (d)
print("\n(d) does a candidate partition behave like a regime process?")
ter = lambda x: np.digitize(x, np.quantile(x, [1/3, 2/3]))


def majority_filter(lab, K, width=9):
    h = width // 2
    onehot = np.stack([(lab == k) for k in range(K)], -1).astype(float)
    cs = np.concatenate([np.zeros((lab.shape[0], 1, K)), np.cumsum(onehot, 1)], 1)
    T_ = lab.shape[1]
    lo = np.clip(np.arange(T_) - h, 0, T_)
    hi = np.clip(np.arange(T_) + h + 1, 0, T_)
    return (cs[:, hi] - cs[:, lo]).argmax(-1)


def trans(lab, K, l):
    a, b = lab[:, :-l].ravel(), lab[:, l:].ravel()
    M = np.zeros((K, K))
    np.add.at(M, (a, b), 1)
    return M / M.sum(1, keepdims=True)


def ck(lab, K, base, L):
    return np.abs(trans(lab, K, L) - np.linalg.matrix_power(trans(lab, K, base), L // base)).max()


def disagree(lab, l):
    return np.mean(lab[:, :-l] != lab[:, l:])


def scaling_exponent(lab):
    """Local exponent of r(L) = P(label_t != label_{t+L}), offset-free:
    log2[(r(8)-r(4)) / (r(4)-r(2))].  Jumps -> 1, threshold crossings of a diffusion -> 1/2."""
    return np.log2((disagree(lab, 8) - disagree(lab, 4)) / (disagree(lab, 4) - disagree(lab, 2)))


noisy_S = np.where(rng.random(S.shape) < 0.008, (S + rng.integers(1, K, S.shape)) % K, S)
cands = {"true S (oracle)": S,
         "true S + 0.8% label noise": noisy_S,
         "recovered regime labels": per_sample,
         "diffusion, terciles": ter(diff[..., 0]),
         "AR 0.999, terciles": ter(gauss[..., 8]),
         "AR 0.99, terciles": ter(gauss[..., 5])}

print("  d1. Chapman-Kolmogorov, raw labels, base lag 1: max |P(L) - P(1)^L|")
print(f"{'partition':>28} {'L=10':>8} {'L=50':>8} {'L=200':>8}")
for n_, l_ in cands.items():
    print(f"{n_:>28} " + " ".join(f"{ck(l_, K, 1, L):8.4f}" for L in (10, 50, 200)))

print("  d2. same after a 9-sample majority filter, base lag 10: max |P(L) - P(10)^(L/10)|")
print(f"{'partition':>28} {'L=20':>8} {'L=50':>8} {'L=100':>8} {'L=200':>8}")
filt = {n_: majority_filter(l_, K) for n_, l_ in cands.items()}
for n_, l_ in filt.items():
    print(f"{n_:>28} " + " ".join(f"{ck(l_, K, 10, L):8.4f}" for L in (20, 50, 100, 200)))

print("  d3. short-lag scaling exponent of the label-disagreement rate (jump = 1, diffusive = 0.5)")
print(f"{'partition':>28} {'r(2)':>8} {'r(4)':>8} {'r(8)':>8} {'exponent':>9}")
for n_, l_ in cands.items():
    print(f"{n_:>28} {disagree(l_,2):8.4f} {disagree(l_,4):8.4f} {disagree(l_,8):8.4f} {scaling_exponent(l_):9.2f}")

print("  d4. autonomy: P(label differs 40 steps later) by within-state margin (window means, w = 8)")


def autonomy(lab_w, margin_w, lagw=5):
    a, b, mg = lab_w[:, :-lagw].ravel(), lab_w[:, lagw:].ravel(), margin_w[:, :-lagw].ravel()
    cut = np.quantile(mg, [1/3, 2/3])
    g = np.digitize(mg, cut)
    return [np.mean(a[g == i] != b[g == i]) for i in range(3)]


sc = WM @ cen.T
srt = np.sort(sc, -1)
rows = {"recovered regime plane": (sc.argmax(-1), srt[..., -1] - srt[..., -2])}
for n_, x in (("diffusion, terciles", diff), ("AR 0.999, terciles", gauss[..., [8]]), ("AR 0.99, terciles", gauss[..., [5]])):
    xm = window_means(x, w)[..., 0]
    qs = np.quantile(xm, [1/3, 2/3])
    rows[n_] = (np.digitize(xm, qs), np.minimum(np.abs(xm - qs[0]), np.abs(xm - qs[1])))
print(f"{'partition':>28} {'low':>8} {'middle':>8} {'high':>8} {'low/high':>9}")
for n_, (lw, mw) in rows.items():
    p = autonomy(lw, mw)
    print(f"{n_:>28} {p[0]:8.4f} {p[1]:8.4f} {p[2]:8.4f} {p[0]/p[2]:9.2f}")
