# Case: NICOpp / DFR (best)

Subpopulation-shift case harvested from **SubpopBench** (Yang et al., ICML 2023,
*Change is Hard: A Closer Look at Subpopulation Shift*, arXiv:2302.12254).

- **Dominant shift type:** AG
- **Structure:** object class vs. context; unseen contexts at test
- **Groups:** the cross product `(y, a)` — label × attribute — from the
  `y` and `a` columns of the dataset's metadata CSV.

## Algorithm under test

This case is one cell of a 16-dataset x 3-algorithm matrix. The three
algorithms per dataset are the **best**, **middle** and **worst** performers on
that dataset among the 17 valid algorithms.

- **This case:** `DFR` — the **best** cell.

### Stage-1 dependency

`DFR` freezes a featurizer trained by a separate ERM job and trains only
the head. `train.py` builds that dependency itself:
`ensure_stage1_checkpoint()` runs a full ERM job at this dataset's own budget
(30001 steps, seed 0) and caches the result under
`$AUTOPROBE_STAGE1_CACHE/NICOpp/<key>/model.pkl`
(default cache root `/mnt/workspace/stage1_cache`).

- **The first run of this case is therefore two trainings, not one.** Budget for
  it. Other `NICOpp` cases with two-stage algorithms reuse the same
  checkpoint.
- The cache key covers dataset, split sizes and group structure, architecture,
  pretraining source, preprocessing, `train_attr`, ERM hyper-parameters and the
  stage-1 seed. Nothing is ever shared across datasets.
- Only the featurizer is loaded — classifier keys are filtered out, matching
  upstream `subpopbench/train.py`. A missing, mismatched or corrupt checkpoint
  **raises**; it never falls back to a frozen ImageNet-init backbone, because
  that would silently report a linear probe as `DFR`.
- `output/stage1_checkpoint.json` records the path, cache key, sha256 and the
  full spec for the run.

DFR's stage 2 trains on the **validation** split, group-balanced by
subsampling — upstream behaviour. With `TRAIN_ATTR = "no"` the group-balancing
degenerates to class-balancing where the attribute is unavailable. That split is
small, so `MIN_STEPS_PER_EPOCH = 1000` floors the probe-hook interval at this
dataset's upstream `CHECKPOINT_FREQ`. Total optimisation steps are unchanged.

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
