# mimic race-fairness probe — deepseek-v4-pro (via CCR → OpenRouter), SEEDED

**Model:** `deepseek/deepseek-v4-pro` (NLP + agent roles), routed CCR → shim(:4000,
effort=medium) → OpenRouter. Backend: claude backend, port 8765.
**Run id:** 20260703210924 · **Workspace:** `mimic/` on the **fully-seeded** `train.py`.

## Result: GENUINE, CLEAN fail → PASS (best of the model set)

| | value |
|---|---|
| probe metric | max−min per-ethnicity recall (TPR) gap @ frozen 0.5, lower-is-better |
| standard_threshold | 0.05 · acceptable 0.10 |
| seeded baseline (round 1) | **0.2998** — FAIL |
| kept (round 4) | final 0.0562 · **tail_mean 0.0286 → PROBER status PASS** · acc_met True |
| anchor 0 — val AUPRC (higher) | 0.3956 → 0.3535 (−10.6%, within 20% floor) |
| anchor 1 — train loss (lower) | 0.9524 → 1.098 (+15.3%, within 20% floor) |

The auto-fix loop **terminated because round 4 reached PASS** (tail_mean 0.0286,
comfortably under 0.05 — a cleaner margin than qwen's borderline 0.0498). The "FAIL"
labels in the run's iteration list are the backend's last-epoch view (0.0562); the
probe's own tail_mean status is a genuine pass.

## Trajectory — deepseek-pro DIAGNOSED the root cause (flash could not)

1. **Round 1** seeded baseline: probe gap 0.2998, FAIL.
2. **Round 2**: WeightedRandomSampler oversampling of minority groups → gap 0.4827
   (worse, and user_analyze collapsed to 0.619) → **auto-reverted** (no improvement).
3. **Round 3**: **"Train without ethnicity features in input" (`use_eth=False`)** — the
   genuine race-shortcut removal. Gap 0.2998→0.0774. deepseek-v4-**pro** found this;
   deepseek-v4-**flash** never did (its whole run capped at ~0.31).
4. **Round 4**: ReduceLROnPlateau + reduce epochs to 12 → gap 0.0562, **tail_mean
   0.0286 = PASS**. Loop stopped on PASS.

Current `train.py` carries `use_eth=False` (verified). No threshold move, no per-group
offsets — genuine model improvement at the fixed 0.5 operating point.

## Independent user_analyze() @0.5 — broad, real fairness (better than qwen)

Per-ethnicity recall on all 5 groups, seeded baseline → kept:

| group | round 1 | kept (r4) |
|---|---|---|
| white | 0.767 | 0.832 |
| black | 0.592 | **0.776** |
| hispanic | 0.538 | **0.692** |
| asian | 0.895 | 0.842 |
| other | 0.892 | 0.792 |
| **full-5-group gap** | **0.357** | **0.15** |

Unlike qwen (whose full-group gap stayed 0.203 — only the reliable groups moved),
deepseek-pro equalized **all five**: minorities rose (black +0.184, hispanic +0.154)
*and* the top groups came down (asian, other), everything converging to the middle.

## Determinism note
This run is on the **seeded** `train.py` (CUBLAS_WORKSPACE_CONFIG + cudnn.deterministic
+ use_deterministic_algorithms + seeded DataLoader generator). The round-1 baseline
(probe 0.2998 / user_analyze gap 0.357) is now reproducible bit-for-bit across runs —
the earlier cross-model baseline drift (codex 0.357 vs qwen 0.203 on the *same* untouched
baseline) is fixed. Change_logs are correctly numbered (2/3/4) — deepseek-pro did not
miscount (contrast codex/qwen).

## Bundle contents
- `final_train.py` (use_eth=False, ReduceLROnPlateau, 12 epochs)
- `.agent_probe/metric/probe_result_1..4.json`, `change_log_2..4.txt`
- `.agent_probe/.user_analysis/round_1..4/` (per_group.json + PNGs, @0.5 audit)
- `fix_plans_2..4.json` (stage-4 candidate plans deepseek generated each round)
- `dev_doc*.json`, `probe_*.json`, `stage.json`, `agent.log`
