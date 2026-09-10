# Case: MIMICNotes / ReWeight (best)

Subpopulation-shift case harvested from **SubpopBench** (Yang et al., ICML 2023,
*Change is Hard: A Closer Look at Subpopulation Shift*, arXiv:2302.12254).

- **Dominant shift type:** AI
- **Structure:** clinical-note outcome vs. demographic subgroup
- **Groups:** the cross product `(y, a)` — label × attribute — from the
  `y` and `a` columns of the dataset's metadata CSV.

## Algorithm under test

This case is one cell of a 16-dataset x 3-algorithm matrix. The three
algorithms per dataset are the **best**, **middle** and **worst** performers on
that dataset among the 17 valid algorithms.

- **This case:** `ReWeight` — the **best** cell.

## Workspace contract

`python train.py`, no arguments. Config constants live at the top of
[`train.py`](train.py). `subpopbench/` is a deep copy of the benchmark source —
each case is standalone, so an agent editing this case cannot affect any other.

## Environment

No per-case venv. `train.py` runs under the repo venv
(`../../venv/bin/python`, CPython 3.12) — `pipeline/stages.py` resolves that
path explicitly, so it holds however the API server was launched. Build it
once with `make setup` at the repo root.

Verify the import chain without touching the data:

```bash
../../venv/bin/python -c "import subpopbench.learning.algorithms, subpopbench.models.networks"
```

**One upstream patch was needed.** `subpopbench/learning/optimizers.py`
imported `AdamW` from `transformers`; HuggingFace deprecated that re-export
and removed it (gone in 4.57 and in 5.x). It now imports
`torch.optim.AdamW`, which is the same algorithm — HF's class was a copy whose
`correct_bias=True` default is exactly torch's behaviour. MIMICNotes uses
`optimizer: "sgd"` anyway, so the function is never even called here; only the
module-level import was breaking. The rest of the vendored SubpopBench source
is unmodified.

## Data

Datasets are **not** duplicated per case. Point every case at one shared root:

```bash
export SUBPOP_DATA_DIR=/mnt/workspace/data
python -m subpopbench.scripts.download --data_path $SUBPOP_DATA_DIR --download
```

`MIMICNoFinding`, `CheXpertNoFinding`, `CXRMultisite` and `MIMICNotes` need
credentialed manual download — see `MedicalData.md` in the original repo.
