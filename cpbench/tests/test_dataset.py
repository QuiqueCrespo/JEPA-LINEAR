"""The dataset layer: regime-dependent parameters, composition, ramps, custom components, identity."""
import numpy as np
import pytest

from cpbench.data import Dataset, expand, register_transform, register_source, available
from cpbench.data.params import resolve, soft_weights
from cpbench.data.sources import Channels
from cpbench.data.transforms import Transform

REG = dict(type="markov", K=2, dwell=320)


def peak_freq(x):
    f = np.fft.rfftfreq(len(x)); return f[(np.abs(np.fft.rfft(x - x.mean())) ** 2).argmax()]


def test_parameter_forms():
    prng = np.random.default_rng(0)
    assert resolve(0.8, 3, 2, prng).shape == (3, 2) and (resolve(0.8, 3, 2, prng) == 0.8).all()
    assert (resolve([0.1, 0.9], 2, 3, prng)[:, 0] == [0.1, 0.9]).all()
    t = resolve(dict(base=0.4, spread=0.5), 3, 4, prng)
    assert all(sorted(t[:, j]) == pytest.approx([-0.1, 0.4, 0.9]) for j in range(4))
    assert (resolve(dict(channels=[0.1, 0.2]), 3, 2, prng) == [[0.1, 0.2]] * 3).all()
    with pytest.raises(ValueError):
        resolve([1, 2, 3], 2, 1, prng)


def test_soft_weights_ramp_linearly_around_the_switch():
    S = np.r_[np.zeros(50, int), np.ones(50, int)][None]
    W = soft_weights(S, 2, 10)
    assert np.allclose(W.sum(-1), 1) and W[0, 50, 1] == pytest.approx(0.5, abs=0.06) and W[0, 40, 1] == 0 and W[0, 60, 1] == 1


def test_spec_roundtrip_is_exact():
    ds = Dataset(dict(regime=REG, sources=[dict(type="oscillator", d=2, freq=[0.05, 0.08]), dict(type="ar", d=2, phi=0.99)],
                      transforms=[dict(type="project", D=16), dict(type="couplings"), dict(type="dropout")]))
    ds2 = Dataset(ds.config())
    assert ds.digest() == ds2.digest()
    assert np.array_equal(ds.sample(2, 500, np.random.default_rng(0))["obs"], ds2.sample(2, 500, np.random.default_rng(0))["obs"])


def test_signal_and_nuisance_are_detected_from_regime_dependence():
    ds = Dataset(dict(regime=REG, sources=[dict(type="ar", d=2, phi=[0.2, 0.9]), dict(type="ar", d=3, phi=0.99)]))
    assert ds.info.signal == "ar(d=2,phi=[0.2,0.9])" and ds.info.nuisance == "ar(d=3,phi=0.99)" and ds.d_clean == 2


def test_composition_factorial_regimes_and_mask():
    spec = dict(regimes={"main": dict(type="markov", K=3, dwell=200), "aux": dict(type="markov", K=2, dwell=1000)},
                sources=[dict(type="ar", d=4, phi={"base": 0.4, "spread": 0.5}), dict(type="ar", d=4, sigma=[0.5, 2], regime="aux"),
                         dict(type="white", d=2, scale=1.5)],
                transforms=[dict(type="project", D=16), dict(type="noise", sigma=0.3), dict(type="dropout", rate=0.1)])
    d = Dataset(spec).sample(3, 4000, np.random.default_rng(1))
    assert d["obs"].shape == (3, 4000, 16) and d["S_s"].max() <= 2 and d["S_aux"].max() <= 1
    assert d["clean"].shape[-1] == 8 and 0.85 < d["mask"].mean() < 0.95


def test_sine_square_triangle_with_frequency_change():
    sine = dict(type="oscillator", freq=[0.01, 0.03], damp=1.0, sigma=0.0, amplitude=1.0)
    for fn in (None, "sign", "triangle"):
        ds = Dataset(dict(regime=dict(REG, dwell=2000), sources=[sine], standardize=False,
                          transforms=[dict(type="pointwise", fn=fn)] if fn else []))
        d = ds.sample(1, 12000, np.random.default_rng(3)); x, S = d["obs"][0, :, 0], d["S_s"][0]
        assert np.abs(x).max() <= 1 + 1e-6
        assert peak_freq(x[S == 0]) == pytest.approx(0.01, abs=0.002) and peak_freq(x[S == 1]) == pytest.approx(0.03, abs=0.002)


def test_unit_variance_isolates_the_autocorrelation_change():
    d = Dataset(dict(regime=REG, sources=[dict(type="ar", d=1, phi=[0.1, 0.95])], standardize=False)).sample(16, 20000, np.random.default_rng(2))
    x, S = d["obs"][..., 0], d["S_s"]
    assert abs(x[S == 0].var() - x[S == 1].var()) < 0.1


def test_bandpass_turns_noise_into_a_spectral_signal():
    d = Dataset(dict(regime=REG, sources=[dict(type="white", d=2)], standardize=False,
                     transforms=[dict(type="bandpass", freq=0.05, channels=[0])])).sample(8, 20000, np.random.default_rng(1))["obs"]
    assert peak_freq(d[0, :, 0]) == pytest.approx(0.05, abs=0.003) and abs(d[..., 0].var() - 1) < 0.1 and abs(d[..., 1].var() - 1) < 0.05


def test_custom_components_plug_in():
    @register_transform("test_square")
    class Square(Transform):
        def apply(self, b, rng):
            b.x = np.sign(b.x) * b.x ** 2

    @register_source("test_step")
    class Step(Channels):
        """Regime k -> constant level k."""
        def simulate(self, ctx):
            return ctx.S[..., None].astype(float).repeat(self.d, -1)

        def extra_driven(self):
            return True

    ds = Dataset(dict(regime=dict(type="scripted", segments=[[0, 100], [1, 100]]), sources=[dict(type="test_step")],
                      transforms=[dict(type="test_square")], standardize=False))
    assert set(np.unique(ds.sample(1, 400, np.random.default_rng(0))["obs"])) <= {0.0, 1.0}
    assert "test_square" in available("transform") and "test_step" in available("source")


def test_time_warp_moves_labels_with_signal():
    ds = Dataset(dict(regime=dict(type="scripted", segments=[[0, 50], [1, 50]]), sources=[dict(type="level", mu=[-5, 5])],
                      transforms=[dict(type="time_warp", strength=0.8, timescale=20)], standardize=False))
    d = ds.sample(4, 1000, np.random.default_rng(3))
    assert (np.sign(d["obs"][..., 0]) == np.where(d["S_s"] == 1, 1, -1)).mean() > 0.97


def test_grid_sources_set_regime_grid_with_random_phase():
    ds = Dataset(dict(regime=dict(type="markov"), sources=[dict(type="transition")], transforms=[dict(type="project", D=16)]))
    assert ds.regimes["main"].grid == 32
    d = ds.sample(32, 3200, np.random.default_rng(4))
    assert len(np.unique(np.flatnonzero(d["cp_s"].ravel()) % 3200 % 32)) > 8


def test_grid_expansion_and_unknown_parameters():
    cfgs = expand(dict(regime=dict(type="markov", K=3), signal=[dict(type="level", mu=[0, 1, 2]), dict(type="phase")],
                       nuisance=[[], [dict(type="white", d=2)]], corruption=[[], [dict(type="project", D=8)]], seed=3))
    assert len(cfgs) == 8 and len({Dataset(c).info.id for c in cfgs}) == 8
    with pytest.raises(TypeError):
        Dataset(dict(regime=REG, sources=[dict(type="ar", phii=0.9)]))
    with pytest.raises(ValueError):
        Dataset(dict(regime=REG, sources=[dict(type="ar", d=2)], transforms=[dict(type="linear", weights=[[1, 1, 1]])]))
