"""Metrics — QWK (primary) + fold-level bootstrap CI.

ADR-001-rev1 §4.5:
    * Primary:    Quadratic Weighted Kappa (Cohen 1968)
    * Auxiliary:  MAE, RMSE, Pearson r, Spearman ρ, Macro-F1
    * Bootstrap:  fold-level resample (13a-2), 1000 iterations, 95% CI
"""
from __future__ import annotations

import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score


def qwk(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None) -> float:
    return cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=labels)


def all_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[int] | None = None) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    out = {
        "qwk": float(qwk(y_true, y_pred, labels)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
    }
    if y_true.std() > 0 and y_pred.std() > 0:
        out["pearson_r"] = float(pearsonr(y_true, y_pred)[0])
        out["spearman_rho"] = float(spearmanr(y_true, y_pred).statistic)
    else:
        out["pearson_r"] = 0.0
        out["spearman_rho"] = 0.0
    out["macro_f1"] = float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0))
    return out


def bootstrap_ci_fold_level(
    fold_qwks: np.ndarray,
    n_iter: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Fold-level bootstrap CI for QWK (per 13a-2).

    Args:
        fold_qwks: array of shape (n_folds * n_seeds,) — one QWK per fold×seed observation.
        n_iter:    bootstrap iterations (default 1000).
        alpha:     two-sided confidence level (0.05 → 95% CI).

    Returns:
        (mean, ci_lo, ci_hi)
    """
    rng = rng or np.random.default_rng(42)
    fold_qwks = np.asarray(fold_qwks, dtype=np.float64)
    n = fold_qwks.size
    means = np.empty(n_iter, dtype=np.float64)
    for i in range(n_iter):
        idx = rng.integers(0, n, size=n)
        means[i] = fold_qwks[idx].mean()
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return float(fold_qwks.mean()), lo, hi


def per_group_qwk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group: np.ndarray,
    labels: list[int] | None = None,
) -> dict[str, dict[str, float]]:
    """Per demographic group QWK + signed bias (ASAP 2.0 fairness)."""
    out: dict[str, dict[str, float]] = {}
    for g in np.unique(group):
        mask = group == g
        if mask.sum() < 5:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        out[str(g)] = {
            "n": int(mask.sum()),
            "qwk": float(qwk(yt, yp, labels)),
            "signed_bias": float(yp.mean() - yt.mean()),
        }
    return out
