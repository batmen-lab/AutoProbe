# home_credit — auto-research echelon (Opus, 10 rounds)

**Model:** Claude Opus (native subscription), AutoProbe auto-research mode
**train.py model:** torch MLP
**Metric:** AUPRC (higher is better)
**Run:** response_20260702221627  (10 iterate rounds)

## Result
- Baseline (naive): **0.079**
- Best (kept): **0.2222 (round 9)**
- Trajectory: 0.079(base) -> 0.097 -> 0.171 -> 0.199 -> 0.201 -> [rev] -> 0.208 -> [rev] -> 0.222 -> 0.222 -> [rev]
- Take-offs: rounds 2, 3, 6, 8/9 (4 take-offs)

## Integrity (verified)
- probe_result_*.json: **10**  |  change_log_*.txt: **10**  (indices 1..10, matched)
- keep/revert logic: OK (each round's kept/reverted decision matches last-epoch vs running-best in the metric direction)
- final_train.py == the best round's snapshot (revert-to-best confirmed)
