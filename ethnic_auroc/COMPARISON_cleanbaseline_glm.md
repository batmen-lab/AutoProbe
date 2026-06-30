# mimic race-fairness — clean-baseline GLM re-run (2026-06-29)

Run `20260629225739`, model **z-ai/glm-5.2**, fully **hands-off** (only steering = the
project-context string posted before stage-1 probe generation; baseline `train.py`
redesigned beforehand at user request; no backend / prompt / nudge edits during the loop).

## Why this run
The earlier mimic showcase leaned on AUROC, which is misleading under heavy class
imbalance (mort_icu ~9% positive): AUROC is prevalence-independent, so per-group AUROC
looked "fair" (~0.84 everywhere) while the real disparity lived in **recall at the
operating point**. The baseline was rebuilt to be a clean, honest imbalanced-binary
setup, and the context was given an imbalance/AUROC caveat so the agent would pick
threshold-aware metrics.

### Baseline redesign (`mimic_copy/train.py`)
- Loss: class-weighted BCE (`pos_weight = n_neg/n_pos`) — kept.
- **Model selection: best val AUPRC** (average precision), NOT AUROC.
- Optimizer: AdamW (lr 1e-3, wd 1e-4), 20 epochs — stable (replaces deliberately
  unstable SGD lr=2.0); the fairness gap is now *natural* (ethnicity one-hot + imbalance),
  not an injected artifact.
- Reports AUPRC / recall / precision / F1 / AUROC(secondary) / acc.
- No prober wiring (agent adds it in stage 3); `use_eth=True`; plain shuffle.

## What GLM did (organically, no nudges)
- **Stage 1:** proposed "Equal-opportunity recall gap across ethnicity" on its own
  (last run it kept defaulting to AUROC-gap). Even noted per-group AUROC should NOT be
  the headline because "white recall 0.6, hispanic 0.2 at the shared threshold even if
  per-group AUROC looks similar" — exactly the measured baseline.
- **Stage 2:** all three dev-plans were equal-opportunity recall-gap (EOD = max−min
  per-group recall at a shared macro-F1 threshold, lower-is-better, accept<0.10,
  standard<0.05). Selected plan 1.
- **Stage 3:** wrote a high-quality prober itself (shared threshold fixed on epoch 1,
  small-group pooling <30, per-group recall/precision/F1 + AOD, CSV + plots).
- **Stage 4 fixes (all organic):** R2 drop ethnicity one-hot → R3 add group-balanced
  WeightedRandomSampler → R4 remove pos_weight (regressed) → R5 add LR scheduler
  (regressed). R4/R5 auto-reverted by revert-on-regression.

## Per-round (GLM's own metric vs independent yardstick)
Independent collector (`collect_auroc.py`) is standalone — reads each round's saved
best-AUPRC checkpoint + raw val features, computes per-group recall@20%PPR / AUROC /
AUPRC. It is unaffected by what GLM does to its own prober.

| round | fix | GLM EOD tail_mean | verdict | indep EOD(max−min) | gap(W−min) | overall AUROC | overall AUPRC | overall recall |
|------:|-----|------------------:|---------|-------------------:|-----------:|--------------:|--------------:|---------------:|
| 1 | baseline (use_eth=True) | 0.3775 | FAIL | 0.509 | 0.240 | 0.846 | 0.396 | 0.640 |
| 2 | drop one-hot | 0.2024 | FAIL | 0.146 | 0.039 | 0.845 | 0.388 | 0.645 |
| 3 | + balanced sampler | **0.0966** | **ACCEPTABLE** | 0.093 | 0.063 | 0.839 | 0.379 | 0.609 |
| 4 | remove pos_weight | 0.1205 | reverted | 0.198 | 0.096 | 0.840 | 0.382 | 0.637 |
| 5 | + LR scheduler | 0.1417 | reverted | 0.049 | 0.018 | 0.829 | 0.359 | 0.590 |

Per-group recall (independent, @~20% predicted-positive rate):

| round | white | black | hispanic | asian | other |
|------:|------:|------:|---------:|------:|------:|
| 1 | 0.627 | 0.408 | 0.308 | 0.632 | 0.817 |
| 2 | 0.651 | 0.633 | 0.538 | 0.684 | 0.633 |
| 3 | 0.627 | 0.571 | 0.538 | 0.632 | 0.567 |
| 5 | 0.599 | 0.571 | 0.615 | 0.579 | 0.567 |

## Result
- **Clean fail → ACCEPT**, fully honest and hands-off. EOD 0.43 → **0.097** (GLM metric),
  0.509 → 0.093 (independent), with **overall AUROC/AUPRC maintained ~0.84/0.38** — i.e.
  GLM **leveled UP** (minority recall rose toward parity: black 0.41→0.57, hispanic
  0.31→0.54), it did NOT game by leveling down (contrast last run's first GLM attempt that
  collapsed AUROC to 0.65).
- The loop **self-corrected**: R4 (drop pos_weight) and R5 (LR scheduler) regressed on the
  per-epoch metric and were auto-reverted; best state = round 3.
- **Strict PASS (EOD<0.05) is the honest ceiling, not reliably reachable** for a max−min
  EOD over a 13-death hispanic stratum without trading off AUPRC. (Round 5's *saved*
  checkpoint did hit independent EOD 0.049, but AUPRC slid to 0.359 and GLM's noisy
  per-epoch tail_mean rejected it.)

## Takeaway vs the prior (polluted-baseline) run
Much **more solid and easily recoverable**: better baseline + a one-line imbalance caveat
in the context made GLM (a) design the right metric, (b) write a correct prober, and
(c) pick the two correct fixes — all without the prober hand-writing, AUROC guard, or
train.py nudge-comments that the prior run needed. The honest fairness story (recall /
equal-opportunity, AUROC can't see it) reproduced cleanly.
