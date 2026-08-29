# home_credit auto-research results — model comparison (GLM, AUPRC metric)

Each subfolder is one "result": the run's `response_<id>/` (stage.json, agent.log,
fix logs) with the workspace `.agent_probe/` (per-round metric artifacts + prober)
moved inside it, plus `final_train.py` (the best-round model code).

All runs: deterministic, 10 rounds, metric = validation AUPRC (higher better),
4-digit artifacts. No-skill AUPRC floor ~0.079 (8% positive rate).

| model | baseline AUPRC | final AUPRC | big take-offs (>=+0.01) | total kept steps |
|-------|---------------:|------------:|:-----------------------:|:----------------:|
| RandomForest (sklearn)      | 0.1097 | **0.2364** | 5 | 6 |
| XGBoost (xgboost 3.3)       | 0.1091 | **0.2213** | 3 | 7 |

For reference (earlier runs, not in this folder):
- torch MLP (2 hidden layers): 0.0788 -> 0.2153, 10/10 rounds improved, scaling gate + features.
- linear logreg: 0.079 -> 0.17.

Degradation levers (each individually lifts AUPRC): tiny n_estimators, shallow
max_depth, and first-N feature cap (excludes EXT_SOURCE_1/2/3). Tree models start
above the floor (~0.11) since they need no feature scaling.
