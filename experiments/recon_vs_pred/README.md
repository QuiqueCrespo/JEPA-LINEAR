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

The intended hypothesis is that the fast nuisance is less useful across the gap than the slow nuisance, so prediction
may devote less capacity to it. This is not a guarantee of nuisance removal; see the timing and interpretation
corrections in the audit amendment below.

## Learners
The common settings are: d = 16, batch 256, lr 3e-4 with cosine decay
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
- `raw_lagcov`, a mechanism-informed reference;
- oracle: lagged covariance of the true latent.

## Metrics, fixed before any learner was trained
- **probe:** linear-probe balanced accuracy of the regime, trained on the first half of independent series and tested on the remaining series (not a temporal split within each series).
- **nuis_r2:** ridge R² of the nuisance latent at window offsets 0, W/2 and W−1.
- **hmm:** sample-level accuracy of an HMM with k = 16, so the features are not truncated. This is the primary clustering metric.
- **gmm:** secondary clustering metric.

No maximum is taken over detectors.

Notation: pred = mean of the two JEPAs, recon = mean of AE and VAE. CIs bootstrap paired seed-level family differences; each family is averaged within seed first.

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
- nuis_r2 can be negative when the probe generalises worse than the test-mean baseline. Preserve negative values; they do not establish absence of information.

Code:
- `design.py`: conditions; writes `cpbench/configs/sweep_recon_vs_pred.yaml`.
- `experiment.py`: metrics and verdicts.
- `run.py`: audit, pilot and HPC scoring.


## Scientific/code audit amendment (2026-10-16)

This amendment follows inspection of the existing control audit and code. It does not claim a new independent
preregistration. H1–H4 numerical thresholds, datasets and primary persistence-selected HMM/GMM remain unchanged.
Historical results must not be silently combined with corrected results.

- `verdicts` requires every condition, six methods, three specified seeds, four controls, finite primary metrics,
  20k training steps, and the corrected `audit_v2` scoring protocol. A failed gate makes all hypotheses void.
  Missing loss at exactly 1k or the final training step fails the loss gate. Pilots cannot issue confirmatory verdicts.
  Family retention relative to the untrained encoder is explicitly qualified separately from relative H1 superiority.
- The H1 CI resamples three paired seed-level family differences. It is a fragile estimate of training variability,
  conditional on dataset seed 0, fixed renderings and fixed evaluation series. Generalisation across dynamical systems
  requires a separately specified multi-dataset-seed experiment; the present run cannot support that claim.
- AE, VAE, CPC and JEPA use the common TCN encoder. TS2Vec retains its reference width-64 dilated CNN, masking,
  crops, dropout and max pooling. Total parameter counts and backbone names are reported. This compares configured
  methods, not loss functions at equal capacity. An objective-only claim requires a separate matched-architecture
  ablation (including comparable regularisation and data exposure), not relabelling TS2Vec-lite as TS2Vec.
- The nuisance factor varies both amplitude and timescale. Lag 96 is start-to-start. The closest context/target samples
  are 33 samples apart: fast AR correlation is exp(-33/16) ≈ 0.127. A window spans four fast correlation times.
  The manipulation is relative predictability, not exact independence or a nuisance constant within each window.
- Ridge R² measures linear accessibility of the pre-render nuisance coordinates. Fourier rendering is nonlinear;
  low R² does not prove nuisance removal. Negative values remain unchanged in tables, plots and hypothesis scoring.
  `run.py hpc <runs-dir> --diagnostics` adds a fixed nonlinear MLP nuisance probe and paired nuisance-resampling
  distances (same dynamics, observation noise and standardisation). These are secondary, with no new H1–H4 thresholds.
- Primary HMM/GMM explicitly use `select=persistence`: three scalings, multiple mixture initialisations, and HMM
  candidates selected by persistence. This is label-free but can prefer slow nuisance. Optional diagnostics include
  fixed whitened preprocessing fits. H3 concerns the specified detector, not all possible regime inference.
- Report HMM/GMM CP AP, default-threshold F1 and covering separately from regime accuracy. Margin remains 32 samples;
  nonoverlapping windows have stride 64. `cp_f1_best` is an oracle-threshold upper bound, secondary only. HMM change
  scores now use joint adjacent-state posteriors; GMM retains its independent-marginal change score.
- Label permutations now maximise balanced accuracy. Result cache filenames are versioned to avoid reusing old
  scoring/probe CSVs. Result aggregation defaults to `audit_v2`; use `collect(..., protocol="legacy")` explicitly for
  historical results. Mixed protocols are never pooled. Existing training and feature artifacts are retained, but new stream-version identities differ.
- Fresh training batches use global batch-index seeds (`stream_version=2`), independent of worker count and resume
  position. Worker count remains a runtime option. Historical streams must be reproduced with historical code;
  corrected analysis never silently locates legacy training identities.
- `run.py hpc` is load-only: it preflights all trained model paths and fails if any is absent. The steps=0 control may
  be constructed without optimisation. Detector/probe fitting can still be expensive: run full scoring on a compute
  node, even though it cannot launch encoder training.
- `python -m cpbench.hpc.make_sweep cpbench/configs/sweep_recon_vs_pred.yaml --dry-run` expands job specifications
  without dataset simulation, calibration, directory creation or submission. The design has 90 trained jobs plus
  10 raw-baseline jobs = 100 total. Dry-run checks resolved learner identities within dataset specifications, not checkpoint content or semantic
  equivalence of dataset configs with different explicit defaults.

The oracle demonstrates latent recoverability, not recoverability from noisy observations. `raw_lagcov` is a
mechanism-informed reference, not a mathematical upper bound. Loss reduction is a preregistered validity condition,
not evidence of convergence. Neither reconstruction nor same-time contrastive learning is guaranteed to preserve
every high-variance factor, and predictive learning can encode unpredictable factors without its predictor using them.

## Pilot-first status and proposed revision (2026-10-16; not enacted)

After the initial HPC submission, the user supplied an earlier pilot status (8/18 runs) and a plan to finish the pilot
before revising the HPC protocol. The newly submitted preflight 7470408, training array 7470411 and analysis 7470412
were all still pending and have been placed on user hold. No task in this launch had started. The original frozen
snapshot is retained for provenance; it must not be released as though it implements a revised protocol.

The following pilot numbers are user-reported interim results, not independently verified from a local pilot CSV:
N0 AE/VAE linear probe about 0.48–0.49 versus pair-MLP about 0.97; F2 VAE HMM about 0.97 with nuisance ridge R² about
0.70. Only the audit CSV is currently present among the local recon_vs_pred result CSVs. No final pilot results or
HPC hypothesis verdicts are inferred from these interim numbers.

Proposals to freeze after the complete pilot report, before releasing any HPC training:

1. H1: distinguish linear accessibility from broader retention. A fixed nonlinear probe on **single-window** features
   is the preferred retention primary; keep the linear probe secondary. The existing `probe_mlp_pair` reads two windows,
   whereas the linear probe reads one, so their gap alone does not isolate nonlinear accessibility or establish that
   information is specifically in a single-window covariance. The architecture, regularisation, training budget,
   series split and effect threshold must be specified prospectively. Do not silently substitute the pair metric into
   the existing H1 code or select a threshold after inspecting the final pilot contrast.
2. H3: remove any claim that nuisance removal is necessary for successful clustering. Good VAE clustering together
   with high nuisance accessibility challenges that necessity claim, but it does not logically refute the original
   numerical JEPA F2-versus-S2 hypothesis. Preserve that original test as a labelled secondary/legacy hypothesis.
   A revised primary may compare prediction versus reconstruction HMM recovery directly on F2, with a prospectively
   specified effect threshold and paired seed analysis. Association between nuisance accessibility and clustering
   should be reported as an exploratory mechanism diagnostic, not causal evidence of nuisance removal.
3. Describe the revision as **pilot-informed and fixed before HPC training**, rather than fixed before any experiment.
   Reserve fresh evaluation trajectories with a separately frozen evaluation seed; do not claim an independent
   confirmatory test while reusing evaluation trajectories already inspected in the pilot. Keep paired conditions
   and renderings explicit, and rebuild the frozen launch snapshot after the revision is settled.

These are proposals, not amendments to implemented H1–H4 thresholds or a report that the pilot has finished.
No additional experiment is launched by this status update.
## Complete pilot results (exploratory: 18/18 runs, 1 seed, 3k steps, N0 / F2 / S2; 2026-10-16)
Computed locally with the code of commit db6ac6f, i.e. *before* the audit amendment above: legacy scoring protocol
(not `audit_v2`), stream version 1, evaluation trajectories from dataset seed 0. "MLP probe" below is `probe_mlp_pair`
(two consecutive windows), not a single-window nonlinear probe. These numbers supersede the interim figures quoted
in the pilot-first status section; they do not issue verdicts.
`cpbench_results/recon_vs_pred_pilot.csv`, `experiment_recon_vs_pred_pilot_run.ipynb`.

HMM accuracy (unsupervised, chance 0.5):

| | N0 | F2 (fast) | S2 (slow) |
|---|---|---|---|
| jepa_separate / jepa_shared | 0.97 / 0.97 | 0.97 / 0.97 | 0.55 / 0.54 |
| ae / vae | 0.97 / 0.97 | 0.54 / 0.97 | 0.54 / 0.55 |
| cpc / ts2vec | 0.96 / 0.96 | 0.49 (collapsed) / 0.54 | 0.51 / 0.53 |
| raw_flat / raw_lagcov | 0.97 / 0.97 | 0.51 / 0.97 | 0.53 / 0.97 |

Mechanical verdicts on the pilot, informative only:

| Test | Verdict | Note |
|---|---|---|
| Gates: oracle, N0 learnability | pass | |
| Gate: no collapse | fail | CPC on F2 has effective rank 1.1 |
| H1 | "supported" (+0.49) | Driven by the linear-probe bias. On the MLP probe the difference is +0.09 (0.976 vs 0.882), which is inconclusive. |
| H2 | inconclusive | pred nuis_r2: F2 0.15, S2 0.17. JEPAs do not *linearly* keep the slow nuisance, yet it breaks their clustering. |
| H3 | supported | pred HMM: F2 0.97, S2 0.55 |
| H4 | not consistent | TS2Vec drops the fast nuisance like JEPA (R² 0.17) but still does not cluster |

Proposed changes for the confirmatory HPC run, to be fixed before it is submitted:
- H1 uses `probe_mlp`.
- H2 adds a nonlinear (MLP) nuisance probe.
- CPC's failure under nuisance is diagnosed and fixed or documented first. Its loss barely falls (×0.96–0.99), but it trains fine on N0.
