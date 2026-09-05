# subpop_cases_3x16 — 48 AutoProbe case workspaces

16 SubpopBench datasets x 3 algorithms each. For every dataset the three
algorithms are the **best**, **middle** and **worst** performers on that dataset
among the 17 valid algorithms, ranked by the worst-group score in SubpopBench
Appendix E.3 (Yang et al., ICML 2023, arXiv:2302.12254).

Generated from [`matrix.csv`](matrix.csv), which is the authoritative selection
and is copied here verbatim. Each case restates its own row in its `CASE.md`.

## Layout

One directory per cell, named `<Dataset>_<tier>_<Algorithm>`:

```
subpop_cases_3x16/<Dataset>_<tier>_<Algorithm>/
  CASE.md           the dataset, the selected algorithm, its published score
  train.py          self-contained; `python train.py`, no arguments
  subpopbench/      DEEP COPY of the benchmark source, per case
  requirements.txt
  LICENSE
```

`subpopbench/` is a real copy in every case, not a link or an import into
`/mnt/workspace/SubpopBench`. AutoProbe physically edits the case repo, so cases
must not be able to reach each other. The 16 templates come from
`../cases_copy/`, which is the pristine (pre-run) version of `../cases/`.

Point a run at a case the same way as before:

```bash
curl -s -X POST .../api/workspace/open -d '{"path":".../subpop_cases_3x16/<Case>"}'
```

## The matrix

★ marks a two-stage algorithm — see below.

| Dataset | best | middle | worst | evidence |
|---|---|---|---|---|
| `CMNIST` | `LfF` · — | `ERM` · — | `LDAM` · — | `estimated` |
| `Waterbirds` | `DFR` ★ · 89 | `BSoftmax` · 74.1 | `ERM` · 69.1 | `published` |
| `CelebA` | `ReSample` · 74.1 | `CBLoss` · 65.4 | `JTT` · 48.3 | `published` |
| `CivilCommentsFine` | `SqrtReWeight` · 68.6 | `CVaRDRO` · 62.9 | `LDAM` · 28.4 | `published` |
| `MultiNLI` | `ReSample` · 66.8 | `ReWeight` · 64.2 | `CVaRDRO` · 48.2 | `published` |
| `MetaShift` | `ReWeightCRT` ★ · 85.1 | `JTT` · 82.6 | `LfF` · 72.3 | `published` |
| `ImagenetBG` | `CRT` ★ · 78.2 | `ERM` · 76.8 | `LfF` · 70.1 | `published` |
| `NICOpp` | `DFR` ★ · 38 | `GroupDRO` · 31.1 | `ReWeight` · 25 | `published` |
| `MIMICNoFinding` | `Focal` · 68.7 | `ReSample` · 67.5 | `LfF` · 62.2 | `published` |
| `MIMICNotes` | `ReWeight` · 84 | `SqrtReWeight` · 83.1 | `CVaRDRO` · 65.6 | `published` |
| `CXRMultisite` | `CRT` ★ · 61.8 | `ReWeightCRT` ★ · 53.1 | `Focal` · 50 | `published` |
| `CheXpertNoFinding` | `DFR` ★ · 75.8 | `SqrtReWeight` · 68.5 | `LfF` · 13.7 | `published` |
| `Living17` | `ReWeightCRT` ★ · 33.1 | `SqrtReWeight` · 27.7 | `LDAM` · 24.3 | `published` |
| `Entity13` | `ReWeightCRT` ★ · — | `SqrtReWeight` · — | `LDAM` · — | `extrapolated` |
| `Entity30` | `ReWeightCRT` ★ · — | `SqrtReWeight` · — | `LDAM` · — | `extrapolated` |
| `Nonliving26` | `ReWeightCRT` ★ · — | `SqrtReWeight` · — | `LDAM` · — | `extrapolated` |

Scores are worst-group accuracy in percent, as reported in the paper. `—` means
the paper does not report it; those cells are `TODO` in the case's `CASE.md`.

- **`CMNIST` is `estimated`** — the ranking is a qualitative judgement, not a
  measured result. The three algorithms are real; their order is not evidence.
- **`Entity13` / `Entity30` / `Nonliving26` are `extrapolated`** from `Living17`
  because all four are BREEDS tasks with unseen-subclass evaluation. Same
  caveat, and no scores at all.

Treat both groups as hypotheses. Do not set a Stage-2 threshold from them
without checking the paper.

## Two-stage algorithms (11 of the 48 cases)

`DFR`, `CRT` and `ReWeightCRT` are the *second* stage of a two-stage recipe:
they freeze a featurizer that a separate ERM job trained, and learn only a new
head. Upstream SubpopBench supplies that featurizer from a checkpoint written by
a prior `--algorithm ERM` run. Skipping it does not fail — it freezes the
ImageNet-init featurizer and reports a linear probe under the method's name.

So `train.py` in these cases builds the dependency itself, via
`ensure_stage1_checkpoint()`:

1. Runs a **complete** ERM job at the dataset's own budget (`dataset_class.N_STEPS`,
   seed 0) — not a fraction of the stage-2 steps.
2. Caches it at `$AUTOPROBE_STAGE1_CACHE/<Dataset>/<key>/model.pkl`
   (default root `/mnt/workspace/stage1_cache`) with a `meta.json` holding the
   full spec and a sha256.
3. Reuses it only when dataset, split sizes and group structure, architecture,
   pretraining source, preprocessing, `train_attr`, ERM hyper-parameters and the
   stage-1 seed all match. The dataset is in the cache path as well as the key,
   so a checkpoint can never cross datasets.
4. Loads the **featurizer only** — keys containing `classifier` or `network.1.`
   are dropped, matching upstream `subpopbench/train.py` — then freezes it and
   trains the fresh head.
5. **Raises** on a missing, mismatched, incomplete or corrupt checkpoint. There
   is no fallback path.

Each run records its provenance in `output/stage1_checkpoint.json`.

**The first run of any of these cases is two trainings, not one.** Cases that
share a dataset share the checkpoint, so `CXRMultisite_best_CRT` and
`CXRMultisite_middle_ReWeightCRT` only pay for it once.

- `CXRMultisite_best_CRT`
- `CXRMultisite_middle_ReWeightCRT`
- `CheXpertNoFinding_best_DFR`
- `Entity13_best_ReWeightCRT`
- `Entity30_best_ReWeightCRT`
- `ImagenetBG_best_CRT`
- `Living17_best_ReWeightCRT`
- `MetaShift_best_ReWeightCRT`
- `NICOpp_best_DFR`
- `Nonliving26_best_ReWeightCRT`
- `Waterbirds_best_DFR`

`DFR` additionally trains its stage 2 on the **validation** split, group-balanced
by subsampling, per upstream. With `TRAIN_ATTR = "no"` that degenerates to
class-balancing where the attribute is unavailable. That split is small, so DFR
cases set `MIN_STEPS_PER_EPOCH` to the dataset's upstream `CHECKPOINT_FREQ`: an
"epoch" (the probe hook interval) becomes one checkpoint interval instead of one
pass over a few hundred rows. **Total optimisation steps are unchanged**, so the
run stays comparable; only the probe's sampling rate changes.

`JTT` is *not* in this list. It is two-stage too, but SubpopBench implements it
as `AbstractTwoStage`, which trains both stages inside one run and needs no
external checkpoint.

## What differs from `../cases/`

Per case: `ALGORITHM`, the `CASE.md` header and its new "Algorithm under test"
section. Repo-wide, in every one of the 48 `train.py`:

- `build_hparams(algorithm)` — the hyper-parameter construction lifted out of
  `main()` so stage 1 and stage 2 build theirs identically. The stage-1 cache key
  is computed from its output, so editing a hyper-parameter invalidates the
  cached checkpoint rather than silently reusing one.
- `_seed_everything(seed)` — same seeding as before, now reusable, so a stage-1
  build cannot leave the stage-2 run on a different RNG state.
- The `STAGE-1 DEPENDENCY` section: `ensure_stage1_checkpoint`,
  `_train_stage1_erm`, `load_stage1_featurizer` and their cache-key helpers.
- `MIN_STEPS_PER_EPOCH` (`None` everywhere except the DFR cases).

Cases whose algorithm is not two-stage take none of those paths at runtime —
`ALGORITHM in TWO_STAGE_ALGORITHMS` is false and the loop is byte-for-byte the
old one.

## Data

Not duplicated per case. One shared root for all 48:

```bash
export SUBPOP_DATA_DIR=/mnt/workspace/data
```

`MIMICNoFinding`, `CheXpertNoFinding`, `CXRMultisite` and `MIMICNotes` need
credentialed manual download.
