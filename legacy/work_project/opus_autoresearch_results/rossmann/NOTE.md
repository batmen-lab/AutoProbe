# rossmann — auto-research echelon (Opus, 10 rounds)

**Model:** Claude Opus (native subscription), AutoProbe auto-research mode
**train.py model:** LightGBM
**Metric:** RMSPE (lower is better)
**Run:** response_20260702225919  (10 iterate rounds)

## Result
- Baseline (naive): **0.533**
- Best (kept): **0.3011 (round 10)**
- Trajectory: 0.533(base) -> 0.504 -> 0.452 -> 0.383 -> 0.301 -> ... -> 0.301
- Take-offs: rounds 2, 3, 4, 10 (4 take-offs)

## Integrity (verified)
- probe_result_*.json: **10**  |  change_log_*.txt: **10**  (indices 1..10, matched)
- keep/revert logic: OK (each round's kept/reverted decision matches last-epoch vs running-best in the metric direction)
- final_train.py == the best round's snapshot (revert-to-best confirmed)
