# Case: CXRMultisite / ReWeightCRT (middle)

Subpopulation-shift case harvested from **SubpopBench** (Yang et al., ICML 2023,
*Change is Hard: A Closer Look at Subpopulation Shift*, arXiv:2302.12254).

- **Dominant shift type:** SC *(verify against Table 1 of the paper)*
- **Structure:** pneumonia vs. hospital site
- **Groups:** the cross product `(y, a)` — label × attribute — from the
  `y` and `a` columns of the dataset's metadata CSV. This is the grouping the
  worst-group metric must use. Grouping by class alone is a *different and
  weaker* metric.

## Algorithm under test

This case is one cell of a 16-dataset x 3-algorithm matrix. The three
algorithms per dataset are the **best**, **middle** and **worst** performers on
that dataset among the 17 valid algorithms, ranked by the worst-group score
reported in SubpopBench Appendix E.3.

| | Algorithm | Worst-group score (%) |
|---|---|---|
| best | `CRT` | 61.8 |
| middle | `ReWeightCRT` | 53.1 |
| worst | `Focal` | 50 |

- **This case:** `ReWeightCRT` — the **middle** cell, published score 53.1.
- **Middle is defined as:** Median-ranked algorithm among the 17 valid algorithms by reported worst-group benchmark score.
- **Evidence:** `published` — Appendix E.3 overall benchmark result; deploy split.
- **Source:** https://arxiv.org/pdf/2302.12254

The scores above come from the paper's own runs, not from this workspace. They
are the target to reason about, not a measurement of this case — reproduce a
baseline here before treating any gap as a defect.

### Stage-1 dependency

`ReWeightCRT` freezes a featurizer trained by a separate ERM job and trains only
the head. `train.py` builds that dependency itself:
`ensure_stage1_checkpoint()` runs a full ERM job at this dataset's own budget
(20001 steps, seed 0) and caches the result under
`$AUTOPROBE_STAGE1_CACHE/CXRMultisite/<key>/model.pkl`
(default cache root `/mnt/workspace/stage1_cache`).

- **The first run of this case is therefore two trainings, not one.** Budget for
  it. Other `CXRMultisite` cases with two-stage algorithms reuse the same
  checkpoint.
- The cache key covers dataset, split sizes and group structure, architecture,
  pretraining source, preprocessing, `train_attr`, ERM hyper-parameters and the
  stage-1 seed. Nothing is ever shared across datasets.
- Only the featurizer is loaded — classifier keys are filtered out, matching
  upstream `subpopbench/train.py`. A missing, mismatched or corrupt checkpoint
  **raises**; it never falls back to a frozen ImageNet-init backbone, because
  that would silently report a linear probe as `ReWeightCRT`.
- `output/stage1_checkpoint.json` records the path, cache key, sha256 and the
  full spec for the run.

## Workspace contract

`python train.py`, no arguments. Config constants live at the top of
[`train.py`](train.py). `subpopbench/` is a deep copy of the benchmark source —
each case is standalone, so an agent editing this case cannot affect any other.

## Data

Datasets are **not** duplicated per case. Point every case at one shared root:

```bash
export SUBPOP_DATA_DIR=/mnt/workspace/data
python -m subpopbench.scripts.download --data_path $SUBPOP_DATA_DIR --download
```

`MIMICNoFinding`, `CheXpertNoFinding`, `CXRMultisite` and `MIMICNotes` need
credentialed manual download — see `MedicalData.md` in the original repo.

## Metric guidance for probe design

The scorer is `subpopbench/utils/eval_helper.py::eval_metrics`, unmodified, so
numbers stay definitionally identical to the benchmark. Per epoch it yields:

| Key | Meaning |
|---|---|
| `min_group.accuracy` | **worst-group accuracy** — the headline subpop metric |
| `adjusted_accuracy` | unweighted mean over `(y, a)` groups — same construct, much lower variance |
| `min_attr.AUROC` | worst-attribute AUROC — the headline for `MIMICNotes` / `CXRMultisite` |
| `overall.accuracy` | average accuracy — the utility floor, not a subpop metric |

Worst-group accuracy is a **min over groups**, so its variance is set by the
size of the smallest group, not by the size of the dataset. Before committing
to it as the tracked metric, measure the noise band for this case (see below).
If the band is wide, track `adjusted_accuracy` instead and report worst-group.

## Noise band (fill in before the first real run)

One plain baseline run writes `output/history.json` with a per-epoch
`worst_group_accuracy` series. The standard deviation over the last ~8 epochs
is the jitter the keep/revert rule actually sees — size the revert deadband at
roughly 2x it.

- smallest validation group (n): `TODO`
- std of `worst_group_accuracy`, last 8 epochs: `TODO`
- suggested revert deadband: `TODO`

## Published reference numbers (fill in from the paper)

`train.py` ships `TRAIN_ATTR = "no"`, so the comparable rows are the
**attribute-unknown-in-training** ones. If you flip it to `"yes"`, switch
columns too — the paper's central finding is that this axis moves the numbers
a lot, and mixing columns makes a good result look like a failure.

Feed these into Stage 2 as the dev plan's thresholds:
`standard_threshold` = best published method, `acceptable_threshold` = ERM.

| Method | Worst-group acc | Avg acc | Selection criterion |
|---|---|---|---|
| ERM (baseline) | `TODO` | `TODO` | `TODO` |
| best published (`CRT`) | 61.8 | `TODO` | `TODO` |
| this case (`ReWeightCRT`) | 53.1 | `TODO` | `TODO` |

Selection criteria in the paper: `OracleWorstAcc` (oracle, not a fair target),
`ValWorstAccAttributeYes` (attributes known at validation),
`ValWorstAccAttributeNo` (attributes unknown — groups degenerate to classes).
