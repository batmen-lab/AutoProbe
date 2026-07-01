# m5_forecast — LightGBM (lag/calendar features), RMSE (lower better)

Real trained model now (replaces the 1-epoch naive-forecast baseline): each
boosting round = one epoch, so every round has >=3 epochs (baseline 3 rounds;
the agent raised n_estimators to 77-200). Metric = validation RMSE over the
28-day horizon.

Echelon (baseline 3.365 -> best **2.0135**), 8 take-offs incl. big jumps at
r2 (2.9765) and r3 (2.0477, features + depth). Beats the old naive-mean
result (2.1699). Levers: n_estimators (epochs) / feature count (lags, rolling
means, weekday, series ids) / num_leaves.
