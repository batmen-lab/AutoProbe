# AutoProbe auto-research echelons — Opus, 10 rounds each

Re-run of all four auto-research projects on the native **Claude Opus** subscription,
**10 iterate rounds** each (one `# potential_improvement` marker applied per round,
revert-on-regression keeps only improving rounds). home_credit uses the **MLP** train.py.

| project | model | metric | baseline -> best | take-offs | 10 logs+results |
|---|---|---|---|---|---|
| home_credit | torch MLP | AUPRC (up) | 0.079 -> 0.222 | 4 | yes |
| ieee_fraud  | XGBoost   | AUPRC (up) | 0.258 -> 0.495 | 4 | yes |
| rossmann    | LightGBM  | RMSPE (down) | 0.533 -> 0.301 | 4 | yes |
| m5_forecast | LightGBM  | RMSE (down)  | 3.365 -> 2.969 | 3 | yes |

Every run verified: 10 probe_result_*.json == 10 change_log_*.txt, keep/revert logic
correct, and final_train.py == best-round snapshot. Naive baselines for the
git-recoverable repos (home_credit MLP = nested-git d7277bf, ieee = 019ecdc) are exact;
rossmann/m5 baselines were reconstructed (saved under ../baselines/) since their clean
versions were not git-recoverable.

Each `<project>/response_<id>/.agent_probe/` = run artifacts; `final_train.py` = best; `NOTE.md` = per-project detail.
