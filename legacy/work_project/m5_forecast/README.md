# M5 Forecasting - Accuracy

This workspace targets the **M5 Forecasting - Accuracy** competition: hierarchical
daily-sales forecasting for Walmart products.

## 1) Task Definition

Predict unit sales for the **next 28 days** for each of ~30,490 product/store
series (3,049 products × 10 stores across 3 states, 3 categories, 7 departments).

This is a multi-horizon time-series forecasting problem with a strong hierarchy
(item → dept → cat, store → state, and all aggregations in between).

## 2) Dataset Structure

- `data/sales_train_validation.csv` — one row per series (`id`), columns
  `d_1 … d_1913` hold daily unit sales. Keys: `item_id, dept_id, cat_id,
  store_id, state_id`.
- `data/calendar.csv` — maps each `d_*` to a date, weekday, month, year, events,
  and SNAP flags.
- `data/sell_prices.csv` — weekly selling price per `store_id, item_id`.
- `data/sample_submission.csv` — submission format (28 forecast columns `F1…F28`).

## 3) Evaluation Metric

Official metric: **WRMSSE** (weighted root mean squared scaled error) aggregated
over the 12 hierarchy levels. As a lightweight proxy this baseline reports plain
**RMSE** over a held-out 28-day window — lower is better.

## 4) Baseline Strategy

Strong approaches use LightGBM on lag/rolling features, or recursive/global
neural forecasters (DeepAR, N-BEATS). A reasonable first milestone is a per-series
recursive GBM with calendar + price + lag features.

## 5) Rough Baseline (this workspace)

The shipped `train.py` is a deliberately weak **naive mean forecast**: it holds
out the last 28 days, predicts each series by the mean of the preceding 28 days
(a flat line), and reports validation RMSE. It ignores calendar effects, prices,
trend, seasonality, and the hierarchy — leaving large headroom for the research
pipeline.

## 6) End-to-End Pipeline

1. Load `sales_train_validation.csv`
2. Split off the last 28 days as validation
3. Forecast each series (baseline: recent mean)
4. Evaluate with RMSE (proxy for WRMSSE)
5. Incrementally add calendar/price/lag features and a real model
