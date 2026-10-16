"""Sensor regimes independent of the dynamics, counterfactual pairs, the render transform, the invariance score and
the reference-faithful TS2Vec loss."""
import numpy as np
import cpbench  # noqa: F401  (import before torch: libomp order)
import torch

from cpbench import make_learner
from cpbench.data import Dataset
from cpbench.eval.invariance import sensitivity
from cpbench.models.nets import MixtureJEPA, TS2Vec, hierarchical_contrast

CFG = dict(regimes=dict(main=dict(type="markov", K=2, dwell=100), sensor=dict(type="markov", K=2, dwell=300)), target="main",
           sources=[dict(type="oscillator", d=1, quadrature=True, freq=[0.019, -0.019], damp=0.98)],
           transforms=[dict(type="render", D=16, kind="bumps"), dict(type="affine", gain=[1.0, 1.5], bias=[0.0, 0.3], regime="sensor"),
                       dict(type="noise", sigma=[0.1, 0.6], regime="sensor")], seed=0)


def test_counterfactual_pairs_share_the_trajectory():
    ds = Dataset(CFG)
    a = ds.sample(2, 400, np.random.default_rng(1), fix={"sensor": 0})
    b = ds.sample(2, 400, np.random.default_rng(1), fix={"sensor": 1})
    assert (a["S_s"] == b["S_s"]).all() and np.allclose(a["clean"], b["clean"])
    assert (a["S_sensor"] == 0).all() and (b["S_sensor"] == 1).all() and not np.allclose(a["obs"], b["obs"])
    free = ds.sample(8, 4000, np.random.default_rng(2))
    assert abs(np.corrcoef(free["S_s"].ravel(), free["S_sensor"].ravel())[0, 1]) < 0.2


def test_render_shapes_and_identity_of_old_datasets():
    for kind in ("bumps", "fourier"):
        ds = Dataset({**CFG, "transforms": [dict(type="render", D=24, kind=kind)]})
        assert ds.D == 24 and np.isfinite(ds.sample(1, 100, np.random.default_rng(0))["obs"]).all()
    plain = dict(regime=dict(type="markov", K=2, dwell=50), sources=[dict(type="ar", d=2)], transforms=[dict(type="noise", sigma=0.3)])
    assert "regime" not in Dataset(plain).config()["transforms"][0]       # unset regime= leaves identities unchanged


def test_invariance_score():
    ds = Dataset(CFG)
    iv = {"none": ({"sensor": 0}, {"sensor": 0}), "sensor": ({"sensor": 0}, {"sensor": 1})}
    s = sensitivity(make_learner("raw_flat", window=16), ds, iv, R=2, n=800)
    assert s["none"] == 0 and s["sensor"] > 0.3


def test_ts2vec_reference_loss():
    torch.manual_seed(0)
    net = TS2Vec(D=4, W=32, d=8, depth=3)
    loss, _ = net.loss(torch.randn(16, 32, 4), None); loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    z = torch.randn(4, 16, 5)
    assert hierarchical_contrast(z, z) < hierarchical_contrast(z, torch.randn(4, 16, 5))
    net.eval(); assert net.encode(torch.randn(3, 32, 4)).shape == (3, 8)


def test_mixture_jepa_weights():
    torch.manual_seed(0)
    net = MixtureJEPA(D=4, W=16, experts=3)
    loss, parts = net.loss(torch.randn(8, 16, 4), torch.randn(8, 16, 4)); loss.backward()
    assert torch.isfinite(loss) and net.A.grad is not None and set(parts) >= {"pred", "usage_entropy"}
    net.eval(); w = net.encode(torch.randn(5, 16, 4))
    assert w.shape == (5, 3) and torch.allclose(w.sum(-1), torch.ones(5))
