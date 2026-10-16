"""Small regression tests for scientific verdicts, load-only safety, and probability scoring."""
import itertools
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from cpbench import Benchmark, make_learner
from cpbench.data import Dataset
from cpbench.detectors.hmm import forward_backward
from cpbench.eval.metrics import best_perm_acc
from cpbench.features.stream import Stream
from cpbench.hpc.make_sweep import main as sweep_main
from experiments.recon_vs_pred.design import CONDITIONS, CONTROLS, LEARNERS
from experiments.recon_vs_pred.experiment import _boot_diff, _loss_drop, verdicts


def complete_results():
    rows = []
    for c in CONDITIONS:
        for l in (*LEARNERS, *CONTROLS):
            for s in ((0, 1, 2) if l in LEARNERS else (0,)):
                rows.append(dict(cond=c, learner=l, seed=s, probe=0.95 if l != 'untrained' else 0.6,
                                 nuis_r2=0.7, hmm=0.97, gmm=0.9, eff_rank=8.0, loss_drop=0.5,
                                 steps=20000 if l in LEARNERS else 0, protocol='audit_v2'))
    return pd.DataFrame(rows)


@pytest.mark.parametrize('damage', ['missing_oracle', 'missing_seed', 'duplicate', 'nan_loss', 'nan_rank',
                                     'oracle_failure', 'collapse', 'pilot_budget', 'legacy', 'no_trained'])
def test_invalid_confirmatory_results_never_decide_hypotheses(damage):
    df = complete_results()
    if damage == 'missing_oracle':
        df = df[df.learner != 'oracle']
    elif damage == 'missing_seed':
        df = df.drop(df[(df.cond == 'F2') & (df.learner == 'ae') & (df.seed == 2)].index)
    elif damage == 'duplicate':
        df = pd.concat([df, df.iloc[:1]])
    elif damage == 'nan_loss':
        df.loc[0, 'loss_drop'] = np.nan
    elif damage == 'nan_rank':
        df.loc[0, 'eff_rank'] = np.nan
    elif damage == 'oracle_failure':
        df.loc[df.learner == 'oracle', 'hmm'] = 0.8
    elif damage == 'collapse':
        df.loc[0, 'eff_rank'] = 1
    elif damage == 'pilot_budget':
        df.loc[df.learner.isin(LEARNERS), 'steps'] = 3000
    elif damage == 'legacy':
        df = df.drop(columns='protocol')
    else:
        df = df[~df.learner.isin(LEARNERS)]
    out = verdicts(df)
    assert (out[out.test.str.startswith('H')].verdict == 'void').all()
    assert len(out[out.test.str.startswith('H')]) == 4


def test_valid_and_exploratory_verdicts():
    df = complete_results()
    out = verdicts(df)
    assert out[out.test.str.startswith('gate:')].verdict.eq('pass').all()
    assert len(out[out.test.str.startswith('H')]) == 4
    pilot = verdicts(df, confirmatory=False)
    assert pilot[pilot.test.str.startswith('H')].verdict.eq('exploratory').all()
    assert verdicts(pd.DataFrame()).verdict.iloc[-1] == 'void'


def test_bootstrap_preserves_seed_pairing():
    delta, ci = _boot_diff(np.array([0.2, 0.5, 0.8]), np.array([0.1, 0.4, 0.7]))
    assert delta == pytest.approx(0.1)
    np.testing.assert_allclose(ci, [0.1, 0.1])


def test_loss_gate_requires_exact_logged_endpoints(tmp_path):
    p = tmp_path / 'log.jsonl'
    p.write_text('\n'.join(json.dumps(dict(kind='train', step=s, loss=v)) for s, v in [(1000, 2), (19000, 1)]))
    assert np.isnan(_loss_drop(tmp_path, 20000))
    with p.open('a') as f:
        f.write('\n' + json.dumps(dict(kind='train', step=20000, loss=0.5)))
    assert _loss_drop(tmp_path, 20000) == 0.25


def test_load_only_never_calls_training(tmp_path, monkeypatch):
    ds = Dataset(dict(sources=[dict(type='white')], standardize=False))
    b = Benchmark(ds, out=str(tmp_path), load_only=True)
    f = make_learner('ae', steps=20)
    monkeypatch.setattr(f, 'fit', lambda *a, **k: pytest.fail('training was invoked'))
    with pytest.raises(FileNotFoundError, match='Load-only'):
        b.fit(f)
    assert not list(tmp_path.iterdir())


def test_balanced_alignment_handles_imbalanced_classes():
    true = np.repeat([0, 0, 1, 1], [60, 40, 9, 1])
    pred = np.repeat([0, 1, 0, 1], [60, 40, 9, 1])
    assert best_perm_acc(true, pred, 2) == pytest.approx(0.65)


def test_joint_switch_posteriors_match_enumeration():
    A = np.array([[0.9, 0.1], [0.2, 0.8]])
    pi = np.array([0.4, 0.6])
    B = np.array([[0.2, 0.8], [0.6, 0.4], [0.7, 0.3]])
    paths = list(itertools.product(range(2), repeat=3))
    weights = np.array([pi[p[0]] * B[0, p[0]] * A[p[0], p[1]] * B[1, p[1]] * A[p[1], p[2]] * B[2, p[2]] for p in paths])
    weights /= weights.sum()
    expected = [0] + [sum(w for p, w in zip(paths, weights) if p[t] != p[t-1]) for t in (1, 2)]
    _, _, _, switch = forward_backward(np.log(B), A, pi, [3], return_switch=True)
    np.testing.assert_allclose(switch, expected)
    _, _, _, switch = forward_backward(np.log(np.tile(B, (2, 1))), A, pi, [3, 3], return_switch=True)
    np.testing.assert_allclose(switch, expected * 2)


def test_stream_batch_randomness_is_independent_of_workers_and_resume(monkeypatch):
    import cpbench.features.stream as mod
    class DummyDataset:
        def __init__(self, spec):
            pass
        def train_batch(self, batch, window, gap, rng):
            x = rng.normal(size=(batch, window, 1)).astype(np.float32)
            return x, x.copy(), {}
    monkeypatch.setattr(mod, 'Dataset', DummyDataset)
    monkeypatch.setattr(mod.torch.utils.data, 'get_worker_info', lambda: None)
    single = iter(Stream({}, 2, 4, 2, 7, 0))
    expected = [next(single)[0].numpy() for _ in range(6)]
    workers = []
    for wid in range(2):
        monkeypatch.setattr(mod.torch.utils.data, 'get_worker_info', lambda wid=wid: SimpleNamespace(id=wid, num_workers=2))
        workers.append(iter(Stream({}, 2, 4, 2, 7, 0)))
        # __iter__ captures worker info on first next, not construction.
        actual = [next(workers[-1])[0].numpy() for _ in range(3)]
        for a, e in zip(actual, expected[wid::2]):
            np.testing.assert_array_equal(a, e)
    monkeypatch.setattr(mod.torch.utils.data, 'get_worker_info', lambda: None)
    resumed = iter(Stream({}, 2, 4, 2, 7, 3))
    np.testing.assert_array_equal(next(resumed)[0].numpy(), expected[3])


def test_dry_run_does_not_simulate_or_write(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    import cpbench.hpc.make_sweep as mod
    config = dict(name='tiny', out='runs', datasets=[dict(sources=[dict(type='white')])],
                  learners=['ae', 'vae'], seeds=[0, 1, 2], baselines=['raw_flat'], slurm={})
    path = tmp_path / 'sweep.yaml'
    OmegaConf.save(OmegaConf.create(config), path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mod, 'Dataset', lambda *a: pytest.fail('dry-run simulated data'))
    lines = sweep_main(str(path), dry_run=True)
    assert len(lines) == 7
    assert set(p.name for p in tmp_path.iterdir()) == {'sweep.yaml'}


def test_corrected_and_legacy_results_are_never_pooled(tmp_path):
    from cpbench.results.aggregate import collect
    pd.DataFrame([dict(metric='acc', value=0.5)]).to_csv(tmp_path / 'hmm_old.csv', index=False)
    pd.DataFrame([dict(metric='acc', value=0.8, scoring_protocol='audit_v2')]).to_csv(tmp_path / 'hmm_new.csv', index=False)
    assert collect(str(tmp_path)).value.tolist() == [0.8]
    assert collect(str(tmp_path), protocol='legacy').value.tolist() == [0.5]
    with pytest.raises(ValueError, match='Mixed scoring'):
        collect(str(tmp_path), protocol=None)


def test_tiny_detector_pipeline_and_versioned_cache(tmp_path):
    ds = Dataset(dict(regime=dict(type='markov', K=2, dwell=32),
                      sources=[dict(type='ar', d=2, phi=[0.2, 0.8])], standardize=False,
                      splits=dict(test=[2, 128], val=[2, 128])))
    b = Benchmark(ds, detectors=[dict(type='hmm', k=2, select='whiten'),
                                 dict(type='gmm', k=2, select='whiten'),
                                 dict(type='arhmm', k=2, select='whiten')],
                  probes=False, cache_data=False, out=str(tmp_path), log=lambda *a: None)
    f = make_learner('raw_stats', window=8)
    first = b.run(f)
    assert set(first.scoring_protocol) == {'audit_v2'}
    assert {'acc', 'cp_ap', 'cp_f1', 'cp_covering', 'cp_f1_best'} <= set(first.metric)
    assert np.isfinite(first.value).all()
    pd.testing.assert_frame_equal(first, b.run(f), check_dtype=False)
    assert len(list(tmp_path.rglob('*_scoring_v2.csv'))) == 3


def test_nonlinear_nuisance_probe_detects_information_missed_by_ridge():
    from cpbench.eval.invariance import nuisance_mlp_r2, nuisance_r2
    rng = np.random.default_rng(7)
    Z = rng.uniform(-2, 2, (4, 200, 1))
    U = Z ** 2
    starts = np.arange(200)
    assert nuisance_r2(Z, U, starts, W=1) < 0.1
    assert nuisance_mlp_r2(Z, U, starts, W=1) > 0.5


def test_paired_nuisance_distance_is_zero_without_nuisance():
    from cpbench.eval.invariance import resampled_nuisance_distance
    ds = Dataset(dict(sources=[dict(type='ar', d=2, phi=0.8), dict(type='ar', d=2, phi=0.9, scale=0)],
                      transforms=[dict(type='fold', blocks=2)], standardize=False))
    f = make_learner('raw_flat', window=8)
    assert resampled_nuisance_distance(f, ds, source=1, R=2, n=64) == 0
