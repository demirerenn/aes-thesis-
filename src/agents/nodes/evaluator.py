"""Evaluator Agent — reads training artifacts, renders run report, decides gating.

Pipeline:
    1. Load runs/<run_name>/summary_<run_name>.json produced by train_baseline.py.
    2. Load per-sample prediction CSVs (preds-*.csv) if present.
    3. Aggregate confusion matrix, worst predictions, length-bucketed errors.
    4. Apply ADR-001-rev2 gating thresholds (QWK 0.82 / 0.78 / 0.72).
    5. Render report_<run_name>.md via Jinja2 template.
    6. Return a Decision for the LangGraph state.

This node is intentionally deterministic — no LLM call needed. The LLM
wrapper in graph.py can optionally post-process the rendered markdown
for stylistic polish or Turkish-English reconciliation; that is a
separate, optional step.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape


# ----------------------------------------------------------------------
# Gating — single source of truth for pilot/CV decision thresholds
# ----------------------------------------------------------------------

GATING = [
    # (key, label, desc, min_qwk, next_agent)
    ("go",         "GO",         "QWK ≥ 0.82 — ileriye genişlet (sonraki kova / tam CV'ye geç)",       0.82, "thesis_writer"),
    ("iterate_a",  "ITERATE-A",  "0.78 ≤ QWK < 0.82 — hiperparametre arama (LR, epoch, batch)",        0.78, "feedback_strategy"),
    ("iterate_b",  "ITERATE-B",  "0.78 ≤ QWK < 0.82 — loss ablation (CORAL, MSE+rank, label smoothing)", None, "feedback_strategy"),
    ("revise",     "REVISE",     "0.72 ≤ QWK < 0.78 — mimari revizyon (backbone/loss değişimi)",       0.72, "architect"),
    ("rollback",   "ROLLBACK",   "QWK < 0.72 — kök-neden analizi (data-leak, tokenizer, loss bug)",   0.0,  "architect"),
]


# ----------------------------------------------------------------------
# Core helpers (pure functions — unit-testable)
# ----------------------------------------------------------------------

def decide_gating(qwk_mean: float) -> tuple[str, str, str]:
    """Return (decision_key, label, next_agent) for given QWK."""
    if qwk_mean >= 0.82:
        return "go", "GO", "thesis_writer"
    if qwk_mean >= 0.78:
        return "iterate_a", "ITERATE-A", "feedback_strategy"
    if qwk_mean >= 0.72:
        return "revise", "REVISE", "architect"
    return "rollback", "ROLLBACK", "architect"


def build_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            cm[t, p] += 1
    return cm


def render_cm_ascii(cm: np.ndarray, label_prefix: str = "") -> str:
    n = cm.shape[0]
    header = "          " + " ".join(f"{j+1:>5}" for j in range(n))
    lines = [header]
    for i in range(n):
        row = f"Gerçek {i+1} [" + " ".join(f"{cm[i,j]:>5}" for j in range(n)) + " ]"
        lines.append(row)
    return "\n".join(lines)


def length_bucket_errors(preds_df: pd.DataFrame) -> list[dict]:
    """MAE + QWK per length bucket."""
    from src.aes.metrics import qwk as qwk_fn  # local import to avoid circular

    bins = [(0, 150, "very_short"), (150, 300, "short"),
            (300, 600, "medium"), (600, 1200, "long"), (1200, 10**6, "very_long")]
    out = []
    for lo, hi, tag in bins:
        sub = preds_df[(preds_df["essay_len_words"] >= lo) & (preds_df["essay_len_words"] < hi)]
        if len(sub) == 0:
            continue
        mae = float(np.abs(sub["y_true"] - sub["y_pred"]).mean())
        q = float(qwk_fn(sub["y_true"].values, sub["y_pred"].values)) if len(sub) >= 10 else float("nan")
        out.append({"bucket": tag, "n": int(len(sub)), "mae": mae, "qwk": q})
    return out


def diagonal_dominance(cm: np.ndarray) -> float:
    total = cm.sum()
    return float(np.trace(cm) / total * 100.0) if total else 0.0


def top_error_pair(cm: np.ndarray) -> str | None:
    mask = ~np.eye(cm.shape[0], dtype=bool)
    off = cm * mask
    if off.max() == 0:
        return None
    i, j = np.unravel_index(np.argmax(off), off.shape)
    return f"(gerçek={i+1}, tahmin={j+1}) — {int(off[i, j])} adet"


# ----------------------------------------------------------------------
# Main node entry — signature matches LangGraph stub contract
# ----------------------------------------------------------------------

@dataclass
class EvaluatorInput:
    run_dir: Path
    config_path: Path
    template_dir: Path


def evaluate_run(inp: EvaluatorInput) -> dict[str, Any]:
    """Read artifacts, render report, return a Decision dict.

    Output dict keys:
        report_path, decision, next_agent, qwk_mean, summary_path, rationale
    """
    # ---- 1. Load summary + config ----
    summary_candidates = list(inp.run_dir.glob("summary_*.json"))
    if not summary_candidates:
        raise FileNotFoundError(f"no summary_*.json in {inp.run_dir}")
    summary = json.loads(summary_candidates[0].read_text())
    cfg = yaml.safe_load(Path(inp.config_path).read_text())

    # ---- 2. Aggregate per-sample predictions (optional) ----
    pred_files = sorted(inp.run_dir.glob("preds-*.csv"))
    if pred_files:
        preds_all = pd.concat([pd.read_csv(p) for p in pred_files], ignore_index=True)
        n_classes = int(cfg["model"]["num_classes"])
        cm = build_confusion_matrix(preds_all["y_true"].values, preds_all["y_pred"].values, n_classes)
        cm_render = render_cm_ascii(cm)
        worst = preds_all.sort_values("abs_err", ascending=False).head(20).to_dict("records")
        length_mae = length_bucket_errors(preds_all)
        diag = diagonal_dominance(cm)
        top_err = top_error_pair(cm)
    else:
        preds_all = pd.DataFrame()
        cm_render = "(per-sample tahminler bulunamadı; trainer.evaluate_df(..., return_predictions=True) ile yeniden çalıştırın)"
        worst, length_mae, diag, top_err = [], [], 0.0, None

    # ---- 3. Secondary metrics aggregation ----
    records = summary.get("fold_records", [])
    def _agg(key: str) -> tuple[float, float]:
        vals = [r[key] for r in records if key in r]
        if not vals:
            return 0.0, 0.0
        return float(np.mean(vals)), float(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)

    sec_defs = [
        ("MAE",       "mae",  None,   None),
        ("RMSE",      "rmse", None,   None),
        ("Pearson r", "pearson_r",    0.85, "ge"),
        ("Spearman ρ", "spearman_rho", 0.85, "ge"),
        ("Macro-F1",  "macro_f1", None, None),
    ]
    secondary = []
    for name, key, target, op in sec_defs:
        mu, std = _agg(key)
        status = "—"
        if target is not None:
            status = "GEÇTİ" if mu >= target else "GEÇMEDİ"
        secondary.append({"name": name, "value": mu, "std": std, "target": target, "status": status})

    # ---- 4. Gating decision ----
    qwk_mean = float(summary["qwk_mean"])
    decision, label, next_agent = decide_gating(qwk_mean)
    rationale = _build_rationale(qwk_mean, summary, cm, length_mae)

    # ---- 5. Render Jinja2 template ----
    env = Environment(
        loader=FileSystemLoader(str(inp.template_dir)),
        autoescape=select_autoescape(enabled_extensions=()),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tmpl = env.get_template("run_report.md.j2")

    ctx = {
        "run_name": summary["run_name"],
        "mode": summary.get("mode", "cv"),
        "generated_at": dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "env": summary.get("env", {}),
        "data_hashes": summary.get("data_hashes", {}),
        "config": cfg,
        "qwk_mean": qwk_mean,
        "qwk_std":  summary.get("qwk_std", 0.0),
        "qwk_min":  summary.get("qwk_min", qwk_mean),
        "qwk_max":  summary.get("qwk_max", qwk_mean),
        "qwk_ci_lo": summary.get("qwk_ci_lo", qwk_mean),
        "qwk_ci_hi": summary.get("qwk_ci_hi", qwk_mean),
        "n_observations": summary["n_observations"],
        "fold_records": records,
        "secondary_metrics": secondary,
        "confusion_matrix_render": cm_render,
        "worst_predictions": worst,
        "length_bucket_mae": length_mae,
        "diagonal_pct": diag,
        "top_error_pair": top_err,
        "fairness": None,  # populated by Fairness Auditor in ASAP2 runs
        "resources": {"wall_clock": summary.get("wall_clock"),
                      "peak_gpu_mem_gb": summary.get("peak_gpu_mem_gb"),
                      "avg_step_sec": summary.get("avg_step_sec")},
        "comparison": [
            {"name": "Taghipour & Ng 2016 (LSTM)",  "qwk": "0.761", "source": "Literatür"},
            {"name": "R²BERT (Yang 2020)",          "qwk": "0.794", "source": "Literatür"},
            {"name": "Bu run",                      "qwk": f"{qwk_mean:.4f}", "source": "—"},
        ],
        "decision": decision,
        "decision_options": [{"key": k, "label": l, "desc": d} for k, l, d, _, _ in GATING],
        "rationale": rationale,
        "next_agent": next_agent,
        "artifacts": [p.name for p in pred_files] + ["summary_" + summary["run_name"] + ".json"],
        "seeds": sorted({r["seed"] for r in records}),
        "has_test": any("test_qwk" in r for r in records),
        "n_train": "—", "n_dev": "—", "n_test": "—",
        "mlflow_run_id": None,
        "rev": 0,
    }
    rendered = tmpl.render(**ctx)
    report_path = inp.run_dir / f"report_{summary['run_name']}.md"
    report_path.write_text(rendered, encoding="utf-8")

    return {
        "report_path": str(report_path),
        "decision": decision,
        "next_agent": next_agent,
        "qwk_mean": qwk_mean,
        "summary_path": str(summary_candidates[0]),
        "rationale": rationale,
    }


def _build_rationale(qwk: float, summary: dict, cm: np.ndarray, length_mae: list[dict]) -> str:
    parts: list[str] = []
    parts.append(f"Ortalama QWK = {qwk:.4f} (n={summary['n_observations']}, std={summary.get('qwk_std', 0.0):.4f}).")
    if cm.size and cm.sum():
        dd = diagonal_dominance(cm)
        parts.append(f"Diagonal dominance {dd:.1f}%.")
        te = top_error_pair(cm)
        if te:
            parts.append(f"En sık hata: {te}.")
    if length_mae:
        worst_bucket = max(length_mae, key=lambda b: b["mae"])
        parts.append(f"En yüksek MAE uzunluk kovasında '{worst_bucket['bucket']}' (MAE={worst_bucket['mae']:.3f}).")
    return " ".join(parts)
