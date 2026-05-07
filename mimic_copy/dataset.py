"""MIMIC-III ICU mortality dataset backed by pre-computed TF-IDF features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from scipy import sparse
from torch.utils.data import Dataset


class MIMICMortalityDataset(Dataset):
    """Loads pre-computed TF-IDF + ethnicity features from disk.

    Expects files produced by ``preprocess.py``:
        - ``{split}_tfidf.npz``  : scipy sparse CSR matrix  (n_samples x n_features)
        - ``{split}_meta.npz``   : numpy archive with keys ``eth`` and ``labels``

    Args:
        data_dir: Directory containing the .npz files.
        split: One of ``"train"``, ``"val"``, ``"test"``.
    """

    def __init__(self, data_dir: str | Path, split: str, use_eth: bool = True) -> None:
        data_dir = Path(data_dir)
        self.tfidf = sparse.load_npz(data_dir / f"{split}_tfidf.npz")
        meta = np.load(data_dir / f"{split}_meta.npz")
        self.eth = torch.tensor(meta["eth"], dtype=torch.float32)
        self.use_eth = use_eth
        self.labels = torch.tensor(meta["labels"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        
        tfidf_vec = torch.tensor(
            self.tfidf[idx].toarray().squeeze(0), dtype=torch.float32
        )
        if self.use_eth:
            features = torch.cat([tfidf_vec, self.eth[idx]])
        else:
            features = tfidf_vec
        return {
            "features": features,
            "label": self.labels[idx],
        }