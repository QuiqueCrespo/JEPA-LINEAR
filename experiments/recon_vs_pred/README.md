# Reconstruction vs contrastive vs prediction (pre-registered, 2026-10-16)

## Question
Does a prediction objective keep a dynamical regime better than a reconstruction objective? If it does, is the
reason that it discards what it cannot predict?

## Design
The observation is the regime signal plus a nuisance plus noise:

    obs = render_s(rotation latent) + sigma_n * render_n(nuisance AR(1) latent) + noise(0.3)

| Component | Definition |
|---|---|
| Regime | Direction of a 2-D damped rotation (omega = ±0.12 rad/sample, period ≈ 52, damp 0.98), Markov, K = 2, dwell 600 |
| Rendering | Each latent goes through its own random Fourier render to D = 64 channels; the two renders are summed (`fold`) |
| Nuisance | 2-D AR(1), independent of the regime |
| Window, gap | W = 64, gap = 32, so the target starts W + gap = 96 samples after the context |

Only the nuisance changes between conditions:

| Condition | Nuisance timescale | Correlation at lag 96 | sigma_n |
|---|---|---|---|
| N0 | — | — | 0 |
| F1, F2 | fast, tau = 16 | ≈ 0.003 | 1, 2 |
| S1, S2 | slow, tau = 2000 | ≈ 0.95 | 1, 2 |

The fast nuisance is smooth within a window, so reconstruction has to spend capacity on it. It is forgotten by the
time of the target, so prediction can drop it. The slow nuisance persists into the target, so prediction should keep
it too.

## Learners
All trained learners share the same TCN backbone and settings: d = 16, batch 256, lr 3e-4 with cosine decay
(min_lr 0.05).

| Family | Learners |
|---|---|
| Reconstruction | `ae`, `vae` |
| Prediction | `jepa_separate`, `jepa_shared` |
| Contrastive, future target | `cpc` |
| Contrastive, same-time target | `ts2vec` |

Controls:
- untrained encoder;
- `raw_flat`;
- `raw_lagcov`, a hand-built ceiling;
- oracle: lagged covariance of the true latent.

## Metrics, fixed before any learner was trained
- **probe:** linear-probe balanced accuracy of the regime, trained on the first half of the series and tested on the second.
- **nuis_r2:** ridge R² of the nuisance latent at window offsets 0, W/2 and W−1.
- **hmm:** sample-level accuracy of an HMM with k = 16, so the features are not truncated. This is the primary clustering metric.
- **gmm:** secondary clustering metric.

No maximum is taken over detectors.

Notation: pred = mean of the two JEPAs, recon = mean of AE and VAE. CIs are bootstrapped over runs.

| Hypothesis | Supported if | Refuted if |
|---|---|---|
| H1: under fast nuisance, prediction keeps the regime better | F2: probe(pred) − probe(recon) ≥ 0.10, CI excludes 0 | difference < 0.05 (0.05–0.10 is inconclusive) |
| H2: mechanism is predictability | nuis_r2 pred: F2 ≤ 0.3, S2 ≥ 0.6; recon ≥ 0.6 on both | nuis_r2 pred F2 ≥ recon F2 − 0.1 |
| H3: clusterability follows nuisance removal | hmm pred: F2 ≥ 0.80 and ≥ S2 + 0.15 | hmm pred F2 < 0.65 |
| H4 (exploratory): target time, not loss type, is the split | F2: nuis_r2(ts2vec) − nuis_r2(cpc) ≥ 0.3 | — |

Validity gates. If any gate fails, the run is void, not evidence:
- the oracle's hmm accuracy is ≥ 0.95 on every condition;
- on N0, at least one trained learner has a probe ≥ 0.9;
- no collapse: effective rank > 2 and the final loss is below the loss at step 1k;
- a family "keeps the regime" only if its probe is ≥ 0.05 above the untrained encoder.

Only the confirmatory run decides the hypotheses: HPC, 20k steps, 3 seeds, all 5 conditions. The local pilot
(N0, F2 and S2, 1 seed, 3k steps) is exploratory.

## Amendment after the control audit, before any learner was trained (2026-10-16)
The audit used only the controls: oracle, raw features and the untrained encoder.

| Condition | `raw_flat` linear probe | `raw_flat` hmm |
|---|---|---|
| N0 | 0.50 | 0.97 |
| F1 | 0.49 | 0.97 |
| F2 | — | 0.51 |
| S1 | — | 0.53 |
| S2 | — | 0.53 |

On N0 and F1 the regime of raw windows sits in the feature *covariance*, which a full-covariance HMM uses and a
linear probe cannot see. A linear probe can therefore under-report retention.

Amendment: the MLP probe on pairs of consecutive windows (`probe_mlp`) is reported as a secondary retention metric.
H1–H4 and their thresholds are unchanged.

Other audit results:
- Oracle hmm: 0.963 on every condition, so the gate passes.
- `raw_lagcov` hmm: 0.97 everywhere, so a hand-built single-window statistic solves every condition.
- The untrained encoder's linear probe: 0.75 (N0), 0.63 (F1), 0.56 (F2), 0.62 (S1), 0.56 (S2).
- nuis_r2 can be negative when ridge regression on high-dimensional raw features overfits. Read it as 0.

Code:
- `design.py`: conditions; writes `cpbench/configs/sweep_recon_vs_pred.yaml`.
- `experiment.py`: metrics and verdicts.
- `run.py`: audit, pilot and HPC scoring.
