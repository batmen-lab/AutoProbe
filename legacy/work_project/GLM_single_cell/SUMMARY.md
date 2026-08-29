# scVI batch-integration probe — GLM run (PASS)

- **Run id:** 20260704140836
- **Workspace:** `scVI/` (scvi-tools SCVI on local 2-batch PBMC)
- **Model driving pipeline:** GLM (`openrouter,z-ai/glm-5.2-20260616` via CCR)
- **Probe:** kBET mean acceptance on the scVI latent (via `scib_metrics.kbet`), `higher_is_better`
- **Thresholds:** standard (PASS) = 0.80, acceptable = 0.55
- **Anchor:** scVI ELBO (`elbo_validation`, lower is better)
- **Result:** **PASS — kBET 0.8145** (round 10)

## Trajectory (Fail → Pass)

| round | kBET | status | change |
|------|--------|--------|--------|
| 1 | 0.3415 | FAIL | stage-3 baseline (real values; device-safe) |
| 2 | 0.271 | FAIL | widen latent/hidden, drop dropout — reverted |
| 3 | 0.509 | FAIL | 50 epochs + KL warmup + lr decay — kept |
| 4 | 0.5455 | FAIL | AdversarialTrainingPlan (batch classifier) — kept |
| 5 | 0.6185 | FAIL (acceptable met) | adversarial weight 2 + 100 epochs — kept |
| 6 | 0.7615 | FAIL | deepen encoder n_layers=2 — kept |
| 7 | 0.7385 | FAIL | batch-aware HVG-2000 — reverted (no improvement) |
| 8 | 0.723 | FAIL | full batch balance (1.0) — reverted (no improvement) |
| 9 | 0.799 | FAIL | widen n_hidden=256 + 150 epochs — kept |
| 10 | **0.8145** | **PASS** | extend to 200 epochs — kept |

## Winning config (final train.py)
- n_latent=10, **n_hidden=256, n_layers=2**, dropout 0.1
- **train-epochs=200**, KL warmup
- **AdversarialTrainingPlan**: `adversarial_classifier=True`, `scale_adversarial_loss=2.0`
- scVI run on **CPU** for the probe (per the device-config comment in train.py — the per-epoch
  latent-extraction callback hit a cpu-vs-cuda mismatch on GPU that pinned kBET to NaN).

## Notes
- Rounds 1–4 driven by the automatic fix-loop (4-round cap → best-effort). Rounds 5–10 were manual
  fix-plan rounds, the later ones steered by a hint drawn from the user's proven prior scVI solve
  (deeper encoder was the decisive capacity lever; HVG and full batch-balance regressed here because
  this prober's kBET scale differs from that earlier run's).
- Per-round visualizations are in `result/round_<N>/` (UMAP before/after + scib metrics.csv), written
  by the modified `save_visualizations` (one dir per train.py execution). There are more `round_*`
  dirs than probe rounds because the agent runs train.py additional times during fix exploration.
