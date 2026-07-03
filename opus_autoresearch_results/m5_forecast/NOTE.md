# m5_forecast — auto-research echelon (Opus, 10 rounds)

**Model:** Claude Opus (native subscription), AutoProbe auto-research mode
**train.py model:** LightGBM
**Metric:** RMSE (lower is better)
**Run:** response_20260702232247  (10 iterate rounds)

## Result
- Baseline (naive): **3.365**
- Best (kept): **2.9694 (round 3)**
- Trajectory: 3.365(base) -> 3.279 -> 2.972 -> 2.969 -> ... (plateau; near-ceiling)
- Take-offs: rounds 1, 2, 3 (3 take-offs; m5 is a known near-ceiling/flat-response metric)

## Integrity (verified)
- probe_result_*.json: **10**  |  change_log_*.txt: **10**  (indices 1..10, matched)
- keep/revert logic: OK (each round's kept/reverted decision matches last-epoch vs running-best in the metric direction)
- final_train.py == the best round's snapshot (revert-to-best confirmed)
