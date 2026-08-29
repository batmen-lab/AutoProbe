# Rossmann Store Sales

This workspace targets the **Rossmann Store Sales** competition: forecast daily
sales for 1,115 Rossmann drug stores.

## 1) Task Definition

Predict the `Sales` value for a given store on a given day. Regression problem
over a panel of stores and dates.

## 2) Dataset Structure

- `data/train.csv` — historical daily records. Columns: `Store, DayOfWeek, Date,
  Sales, Customers, Open, Promo, StateHoliday, SchoolHoliday`.
- `data/test.csv` — the dates to predict (no `Sales`/`Customers`).
- `data/store.csv` — per-store metadata: `StoreType, Assortment,
  CompetitionDistance`, competition open date, `Promo2` and its schedule.

## 3) Evaluation Metric

Official metric: **RMSPE** (root mean squared percentage error). Rows with actual
`Sales == 0` are ignored. Lower is better.

`RMSPE = sqrt( mean( ((y - yhat) / y)^2 ) )`  over rows where `y != 0`.

## 4) Baseline Strategy

Competitive solutions use gradient-boosted trees (XGBoost/LightGBM) or entity-
embedding neural nets over engineered date/promo/competition features. A solid
first milestone is a GBM on day-of-week, promo, month, and store features.

## 5) Rough Baseline (this workspace)

The shipped `train.py` is a deliberately weak **per-store mean predictor**: it
splits off the most recent dates as validation and predicts each row with the
mean sales of that store over the training period. It ignores promo, day-of-week,
holidays, seasonality and all `store.csv` features -- leaving large headroom for
the research pipeline.

## 6) End-to-End Pipeline

1. Load `train.csv` (optionally join `store.csv`)
2. Split off the most recent dates as validation
3. Predict sales (baseline: per-store mean)
4. Evaluate with RMSPE (ignoring zero-sales rows)
5. Incrementally add promo/calendar/store features and a real model
