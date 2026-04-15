"""Dataset loaders and stratified CV splits.

Implements ADR-001-rev1 §4.4:
    * ASAP 1.0 (Latin-1, ftfy repair, anonymized tokens preserved)
    * ASAP 2.0 PERSUADE (UTF-8)
    * Per-prompt min-max label normalization to [0, 1]
    * Stratified 5-fold CV keyed by (essay_set × score_bin)

The label on disk stays integer; models receive the normalized
float in [0, 1] via the ``target`` field. Inverse denormalization is
handled by ``Prompt.denormalize_scores``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


# -------------------------------------------------------------------
# Prompt metadata — static ground truth, never derived from data.
# Values sourced from Kaggle ASAP 2012 rules and EDA_Raporu_v1.
# -------------------------------------------------------------------

@dataclass(frozen=True)
class Prompt:
    essay_set: int
    min_score: int
    max_score: int
    bucket: str  # "short" | "medium" | "long"

    @property
    def num_classes(self) -> int:
        return self.max_score - self.min_score + 1

    def normalize(self, raw: np.ndarray | pd.Series) -> np.ndarray:
        return (np.asarray(raw, dtype=np.float32) - self.min_score) / (self.max_score - self.min_score)

    def denormalize(self, norm: np.ndarray) -> np.ndarray:
        return norm * (self.max_score - self.min_score) + self.min_score

    def to_class_idx(self, raw: np.ndarray | pd.Series) -> np.ndarray:
        """0..K-1 integer class index (for CORN / CORAL heads)."""
        return np.asarray(raw, dtype=np.int64) - self.min_score


ASAP1_PROMPTS: dict[int, Prompt] = {
    1: Prompt(1, 2, 12, "medium"),
    2: Prompt(2, 1, 6, "medium"),
    3: Prompt(3, 0, 3, "short"),
    4: Prompt(4, 0, 3, "short"),
    5: Prompt(5, 0, 4, "short"),
    6: Prompt(6, 0, 4, "short"),
    7: Prompt(7, 0, 30, "medium"),
    8: Prompt(8, 0, 60, "long"),
}


# -------------------------------------------------------------------
# Text cleaning
# -------------------------------------------------------------------

_ANON_RE = re.compile(r"@[A-Z]+\d*")


def repair_text(text: str) -> str:
    """Apply ftfy, collapse whitespace, keep anonymized tokens intact."""
    try:
        import ftfy
        text = ftfy.fix_text(text)
    except ImportError:
        pass
    text = text.replace("\u00a0", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def anonymized_tokens(text: str) -> list[str]:
    return _ANON_RE.findall(text)


# -------------------------------------------------------------------
# Loaders
# -------------------------------------------------------------------

def load_asap1(
    path: str | Path,
    prompts: Iterable[int] | None = None,
    repair: bool = True,
) -> pd.DataFrame:
    """Load ASAP 1.0 training_set_rel3.tsv.

    Returns a DataFrame with columns: essay_id, essay_set, essay, score,
    score_norm (in [0,1]), class_idx.
    """
    df = pd.read_csv(path, sep="\t", encoding="latin-1")
    df = df[["essay_id", "essay_set", "essay", "domain1_score"]].copy()
    df.rename(columns={"domain1_score": "score"}, inplace=True)
    if prompts is not None:
        df = df[df["essay_set"].isin(list(prompts))].copy()
    if repair:
        df["essay"] = df["essay"].astype(str).map(repair_text)
    # Attach per-row normalization
    score_norm = np.zeros(len(df), dtype=np.float32)
    class_idx = np.zeros(len(df), dtype=np.int64)
    for es, sub in df.groupby("essay_set"):
        p = ASAP1_PROMPTS[int(es)]
        idx = sub.index
        score_norm[df.index.get_indexer(idx)] = p.normalize(sub["score"].values)
        class_idx[df.index.get_indexer(idx)] = p.to_class_idx(sub["score"].values)
    df["score_norm"] = score_norm
    df["class_idx"] = class_idx
    return df.reset_index(drop=True)


def load_asap2(path: str | Path, repair: bool = True) -> pd.DataFrame:
    """Load ASAP 2.0 / PERSUADE CSV."""
    df = pd.read_csv(path, encoding="utf-8")
    df = df.rename(columns={"full_text": "essay"}).copy()
    if repair:
        df["essay"] = df["essay"].astype(str).map(repair_text)
    # Holistic score 1..6, 6 classes
    df["score_norm"] = ((df["score"].astype(np.float32) - 1) / 5.0).astype(np.float32)
    df["class_idx"] = (df["score"].astype(np.int64) - 1)
    return df.reset_index(drop=True)


# -------------------------------------------------------------------
# Stratified CV
# -------------------------------------------------------------------

def stratified_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    stratify_cols: Iterable[str] = ("essay_set", "class_idx"),
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic stratified K-fold (ADR §4.5)."""
    key = df[list(stratify_cols)].astype(str).agg("-".join, axis=1).values
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(df)), key))


def iter_folds(
    df: pd.DataFrame,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    for i, (tr, va) in enumerate(folds):
        yield i, df.iloc[tr].reset_index(drop=True), df.iloc[va].reset_index(drop=True)


# -------------------------------------------------------------------
# Fixed stratified train/dev/test split (Pilot Phase — ADR-001-rev2)
# -------------------------------------------------------------------

def fixed_split(
    df: pd.DataFrame,
    ratios: tuple[float, float, float] = (0.70, 0.15, 0.15),
    seed: int = 42,
    stratify_cols: Iterable[str] = ("essay_set", "class_idx"),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Deterministic stratified train/dev/test split.

    Used during Pilot Phase (ADR-001-rev2 addendum): single fixed split +
    multi-seed model training. The split itself is fixed across seeds
    (``seed`` here controls only the split shuffle, not training). Training
    seed variance is measured independently by running the trainer multiple
    times over the SAME (train, dev, test) partition.

    Returns (train_df, dev_df, test_df), each reset-indexed.
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, f"ratios must sum to 1.0, got {ratios}"
    r_tr, r_dev, r_te = ratios

    key = df[list(stratify_cols)].astype(str).agg("-".join, axis=1).values
    # First: carve off test
    from sklearn.model_selection import train_test_split
    idx = np.arange(len(df))
    idx_trdev, idx_test = train_test_split(
        idx, test_size=r_te, random_state=seed, stratify=key,
    )
    # Then: split remainder into train/dev
    key_trdev = key[idx_trdev]
    dev_rel = r_dev / (r_tr + r_dev)
    idx_tr, idx_dev = train_test_split(
        idx_trdev, test_size=dev_rel, random_state=seed, stratify=key_trdev,
    )
    tr_df = df.iloc[idx_tr].reset_index(drop=True)
    dev_df = df.iloc[idx_dev].reset_index(drop=True)
    te_df = df.iloc[idx_test].reset_index(drop=True)
    return tr_df, dev_df, te_df
