# ieee_fraud — auto-research echelon (Opus, 10 rounds)

**Model:** Claude Opus (native subscription), AutoProbe auto-research mode
**train.py model:** XGBoost
**Metric:** AUPRC (higher is better)
**Run:** response_20260702223253  (10 iterate rounds)

## Result
- Baseline (naive): **~0.16**
- Best (kept): **0.4951 (round 7)**
- Trajectory: 0.258(r1) -> 0.436 -> 0.441 -> [rev] -> [rev] -> 0.493 -> 0.495 -> ...
- Take-offs: rounds 2, 3, 6, 7 (4 take-offs)

## Integrity (verified)
- probe_result_*.json: **10**  |  change_log_*.txt: **10**  (indices 1..10, matched)
- keep/revert logic: OK (each round's kept/reverted decision matches last-epoch vs running-best in the metric direction)
- final_train.py == the best round's snapshot (revert-to-best confirmed)
