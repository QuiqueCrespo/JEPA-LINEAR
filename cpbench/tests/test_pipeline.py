"""Feature learners and detectors compose through the Benchmark; custom ones plug in without touching the package."""
import numpy as np
import pytest
import torch.nn as nn

from cpbench import Benchmark, make_learner, make_detector, register_learner, register_detector, TorchFeatureLearner, Detector
from cpbench.features import LEARNERS, FeatureLearner, learner_preset

DATA = dict(regime=dict(type="markov"), sources=[dict(type="level", d=4, mu=dict(base=0, spread=1.0)), dict(type="ar", d=4)], transforms=[dict(type="project", D=16), dict(type="noise")])
FAST = dict(steps=20, batch=16, workers=0, eval_every=10_000, warmup=5, d=16, H=32,
            device_name=__import__("os").environ.get("CPB_TEST_DEVICE", "cpu"))   # MPS shows rare forward NaNs at this tiny size


@pytest.fixture(scope="module")
def bench(tmp_path_factory):
    return Benchmark(DATA, detectors=["gmm", "hmm", "contrast"], probes=False, out=str(tmp_path_factory.mktemp("runs")),
                     quick=True, log=lambda *a: None)


@pytest.mark.parametrize("name", LEARNERS.available(lambda c: c.trainable))
def test_every_trainable_learner_feeds_every_detector(bench, name):
    df = bench.run(make_learner(name, **FAST))
    assert set(df.detector) == {"gmm", "hmm", "contrast"}
    acc = df[(df.detector == "hmm") & (df.metric == "acc") & (df.view == "features")].value
    assert len(acc) == 1 and 0 <= acc.iloc[0] <= 1


@pytest.mark.parametrize("name", ["raw_stats", "raw_flat", "oracle_clean"])
def test_fixed_learners(bench, name):
    df = bench.run(make_learner(name, window=16))
    assert (df.W == 16).all() and {"acc", "cp_ap"} <= set(df.metric)


def test_detectors_are_parametrised_and_results_cached_per_detector(tmp_path):
    b = Benchmark(DATA, detectors=[make_detector("hmm", k=4), make_detector("hmm", k=8)], probes=False, out=str(tmp_path),
                  quick=True, log=lambda *a: None)
    df = b.run("raw_stats")
    assert df[df.metric == "acc"].detector_params.nunique() == 2
    b.detectors = [make_detector("bocpd")]                                # adding a detector computes only that one
    df2 = b.run("raw_stats")
    assert set(df2.detector) == {"bocpd"}


def test_runtime_settings_do_not_change_the_run():
    a, b = make_learner("jepa_shared", **FAST), make_learner("jepa_shared", **{**FAST, "workers": 3, "log_every": 7})
    assert a.tag() == b.tag() and a.tag() != make_learner("jepa_shared", **{**FAST, "lr": 1e-3}).tag()


def test_unknown_parameters_are_rejected():
    with pytest.raises(TypeError):
        make_learner("jepa_shared", windw=32)
    with pytest.raises(TypeError):
        make_detector("hmm", kk=3)


def test_test_split_changes_the_protocol_not_the_run(tmp_path):
    full = Benchmark(DATA, out=str(tmp_path), log=lambda *a: None).with_test(2, 3200)
    quick = Benchmark(DATA, out=str(tmp_path), quick=True, log=lambda *a: None)
    assert full.protocol() != quick.protocol()
    assert full.run_dir(make_learner("raw_stats")) == quick.run_dir(make_learner("raw_stats"))


def test_custom_learner_and_detector(bench):
    class TinyNet(nn.Module):
        def __init__(self, D, W, d=8):
            super().__init__(); self.f = nn.Sequential(nn.Flatten(), nn.Linear(D * W, d)); self.g = nn.Linear(d, D * W)

        def loss(self, xc, xt):
            l = (self.g(self.f(xc)) - xc.flatten(1)).square().mean(); return l, dict(rec=l.item())

        def encode(self, x):
            return self.f(x)

    @register_learner("test_tiny_ae")
    class TinyAE(TorchFeatureLearner):
        defaults = dict(d=8)
        def build(self, D, W):
            return TinyNet(D, W, self.d)

    @register_learner("test_window_mean")
    class WindowMean(FeatureLearner):
        def embed(self, windows):
            return windows.mean(1)

    @register_detector("test_first_channel_sign")
    class FirstChannelSign(Detector):
        defaults = dict(channel=0)
        def detect(self, Z, K, seed=0):
            lab = (Z[..., self.channel] > np.median(Z[..., self.channel])).astype(int)
            sc = np.zeros(lab.shape); sc[:, 1:] = lab[:, 1:] != lab[:, :-1]
            return dict(labels=lab, cp_score=sc)

    b = Benchmark(DATA, detectors=["test_first_channel_sign", "hmm"], probes=False, out=bench.out, quick=True, log=lambda *a: None)
    for f in (make_learner("test_tiny_ae", steps=20, batch=16, workers=0, eval_every=10_000, device_name="cpu"),
              make_learner("test_window_mean", window=16)):
        df = b.run(f)
        assert {"test_first_channel_sign", "hmm"} == set(df.detector)
    tab = b.compare(["test_window_mean", "raw_stats"], detector="test_first_channel_sign")
    assert len(tab) == 2
