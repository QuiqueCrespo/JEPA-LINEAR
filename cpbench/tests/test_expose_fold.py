"""Exposed source latents, the fold transform and the nuisance R^2 metric."""
import numpy as np

from cpbench.data import Dataset
from cpbench.eval.invariance import nuisance_r2

BASE = dict(regime=dict(type="markov", K=2, dwell=100),
            sources=[dict(type="ar", d=2, phi=[0.2, 0.9])], transforms=[dict(type="noise", sigma=0.3)], seed=0)


def two_rendered(phi, scale, expose="nuisance"):
    return dict(regime=dict(type="markov", K=2, dwell=300),
                sources=[dict(type="composite", sources=[dict(type="oscillator", d=1, quadrature=True, freq=[0.02, -0.02], expose="latent")],
                              transforms=[dict(type="render", D=16, kind="fourier")]),
                         dict(type="composite", scale=scale, sources=[dict(type="ar", d=2, phi=phi, expose=expose)],
                              transforms=[dict(type="render", D=16, kind="fourier")])],
                transforms=[dict(type="fold", blocks=2)], seed=0)


def test_expose_absent_from_spec_when_unset():
    ds = Dataset(BASE)
    assert all("expose" not in s for s in ds.config()["sources"])
    assert Dataset(BASE).digest() == ds.digest()


def test_exposed_latents_and_fold():
    ds = Dataset(two_rendered(0.9, 1.0))
    d = ds.sample(2, 500, np.random.default_rng(0))
    assert ds.D == 16 and d["obs"].shape == (2, 500, 16)
    assert d["latent"].shape == (2, 500, 2) and d["nuisance"].shape == (2, 500, 2)
    assert d["clean"].shape == (2, 500, 16)                         # the rendered signal only (the nuisance is not driven)


def test_scale_zero_removes_the_nuisance():
    a = Dataset(two_rendered(0.9, 0.0)).sample(2, 300, np.random.default_rng(1))
    o, c = a["obs"].reshape(-1, 16), a["clean"].reshape(-1, 16)       # equal up to the dataset's standardisation
    assert min(np.corrcoef(o[:, j], c[:, j])[0, 1] for j in range(16)) > 0.9999


def test_nuisance_autocorrelation_matches_phi():
    phi = float(np.exp(-1 / 16))
    u = Dataset(two_rendered(phi, 1.0)).sample(4, 20000, np.random.default_rng(2))["nuisance"]
    ac = np.mean([np.corrcoef(u[r, :-16, c], u[r, 16:, c])[0, 1] for r in range(4) for c in range(2)])
    assert abs(ac - phi ** 16) < 0.05


def test_nuisance_r2_bounds():
    rng = np.random.default_rng(0)
    U = rng.standard_normal((4, 400, 2)); starts = np.arange(0, 368, 8); W = 32
    Z_full = np.concatenate([U[:, starts + o] for o in (0, W // 2, W - 1)], -1)
    assert nuisance_r2(Z_full, U, starts, W) > 0.99
    assert nuisance_r2(rng.standard_normal(Z_full.shape), U, starts, W) < 0.05


def test_legacy_digest_schema_is_preserved():
    import hashlib
    import json
    # Explicit pre-expose schema, independently of current Component defaults/spec().
    legacy = dict(regimes=dict(main=dict(type='markov', K=2, ramp=0, dwell=100,
                                        dist='geometric', transition='uniform', grid=None)),
                  target='main', sources=[dict(type='ar', scale=1.0, regime='main', d=2,
                                              phi=[0.2, 0.9], phi2=0.0, sigma=1.0, corr=0.0, unit_variance=True)],
                  transforms=[dict(type='noise', sigma=0.3, timescale=None)],
                  standardize=True, seed=0, margin=32)
    expected = hashlib.sha1(json.dumps(legacy, sort_keys=True, default=str).encode()).hexdigest()[:8]
    assert Dataset(BASE).digest() == expected


def test_nuisance_lag96_matches_design():
    from cpbench.data.sources import AR
    from cpbench.data.pipeline import Context
    for tau, tolerance in ((16, 0.025), (2000, 0.025)):
        phi = float(np.exp(-1 / tau))
        source = AR(d=2, phi=phi).bind(2, 0)
        S = np.zeros((4, 20000), int)
        u = source.simulate(Context(S, None, 2, np.random.default_rng(42)))
        ac = np.mean([np.corrcoef(u[r, :-96, c], u[r, 96:, c])[0, 1] for r in range(4) for c in range(2)])
        assert abs(ac - phi ** 96) < tolerance


def test_resampling_nuisance_preserves_dynamics_labels_and_noise():
    cfg = two_rendered(0.9, 1.0)
    cfg['standardize'] = False
    ds = Dataset(cfg)
    a = ds.sample(2, 128, np.random.default_rng(0))
    b = ds.sample(2, 128, np.random.default_rng(0), source_seeds={1: 123})
    np.testing.assert_array_equal(a['latent'], b['latent'])
    np.testing.assert_array_equal(a['S_s'], b['S_s'])
    assert not np.array_equal(a['nuisance'], b['nuisance'])
    # With zero nuisance scale, observation noise and all remaining draws are exactly equal.
    cfg['sources'][1]['scale'] = 0
    cfg['transforms'].append(dict(type='noise', sigma=0.3))
    ds = Dataset(cfg)
    a = ds.sample(2, 128, np.random.default_rng(0))
    b = ds.sample(2, 128, np.random.default_rng(0), source_seeds={1: 123})
    np.testing.assert_array_equal(a['obs'], b['obs'])
