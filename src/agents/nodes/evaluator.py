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

import matplotlib
matplotlib.use("Agg")  # headless container — no display
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape


# ----------------------------------------------------------------------
# Gating — ADR-001 rev3.2 iki-rejim eşikleri
# ----------------------------------------------------------------------
# Rejim-A: P2-only (single-prompt, Faz 1-2 hızlı iterasyon).
#          Referans R²BERT P2=0.719 üzerinden kalibre edildi.
# Rejim-B: 8-prompt ortalama (Faz 3 confirmatory, literatür-uyumlu).
#          Referans R²BERT 8-prompt avg=0.794.

GATING_REGIMES = {
    "A": [
        ("go",         "GO",         "QWK ≥ 0.72 — R²BERT-P2 seviyesi; 8-prompt confirmatory'ye geç",      0.72, "thesis_writer"),
        ("iterate_a",  "ITERATE-A",  "0.66 ≤ QWK < 0.72 — epoch/LR/regularization tuning",                 0.66, "feedback_strategy"),
        ("revise",     "REVISE",     "0.60 ≤ QWK < 0.66 — mimari revizyon (backbone/loss/seq_len)",        0.60, "architect"),
        ("rollback",   "ROLLBACK",   "QWK < 0.60 — kök-neden analizi (data-leak, tokenizer, loss bug)",    0.0,  "architect"),
    ],
    "B": [
        ("go",         "GO",         "QWK ≥ 0.82 — literatür üstü; thesis-grade raporlama",                0.82, "thesis_writer"),
        ("iterate_a",  "ITERATE-A",  "0.78 ≤ QWK < 0.82 — hiperparametre arama (LR, epoch, batch)",        0.78, "feedback_strategy"),
        ("revise",     "REVISE",     "0.72 ≤ QWK < 0.78 — mimari revizyon (backbone/loss değişimi)",       0.72, "architect"),
        ("rollback",   "ROLLBACK",   "QWK < 0.72 — kök-neden analizi (data-leak, tokenizer, loss bug)",    0.0,  "architect"),
    ],
}

# Default regime for legacy callers that do not specify one.
GATING = GATING_REGIMES["B"]


def _infer_regime(cfg: dict | None) -> str:
    """Rejim-A if exactly one prompt configured, else Rejim-B."""
    if not cfg:
        return "B"
    prompts = (cfg.get("dataset") or {}).get("prompts")
    return "A" if isinstance(prompts, list) and len(prompts) == 1 else "B"


# ----------------------------------------------------------------------
# Core helpers (pure functions — unit-testable)
# ----------------------------------------------------------------------

def decide_gating(qwk_mean: float, regime: str = "B") -> tuple[str, str, str]:
    """Return (decision_key, label, next_agent) for given QWK under regime."""
    if regime == "A":
        if qwk_mean >= 0.72:
            return "go", "GO", "thesis_writer"
        if qwk_mean >= 0.66:
            return "iterate_a", "ITERATE-A", "feedback_strategy"
        if qwk_mean >= 0.60:
            return "revise", "REVISE", "architect"
        return "rollback", "ROLLBACK", "architect"
    # Rejim-B (8-prompt avg)
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
        # For multi-prompt regression (num_classes=1) the raw-score range spans all prompts.
        # Derive CM size from observed values to avoid clipping prompts with wide scales.
        n_classes_cfg = int(cfg["model"]["num_classes"])
        if n_classes_cfg <= 1:
            n_classes = int(max(preds_all["y_true"].max(), preds_all["y_pred"].max())) + 1
        else:
            n_classes = n_classes_cfg
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
    regime = _infer_regime(cfg)
    decision, label, next_agent = decide_gating(qwk_mean, regime=regime)
    rationale = _build_rationale(qwk_mean, summary, cm, length_mae)
    rationale += f" Rejim-{regime} eşikleri uygulandı (ADR-001 rev3.2 §4.0)."

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
        "decision_options": [{"key": k, "label": l, "desc": d} for k, l, d, _, _ in GATING_REGIMES[regime]],
        "rationale": rationale,
        "next_agent": next_agent,
        "regime": regime,
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

    # Learning curves — rendered when per-epoch history is attached to
    # fold_records (ADR-001 rev3 §5). Older runs without history → skipped.
    lc_path = render_learning_curves(
        records,
        inp.run_dir / f"learning_curves_{summary['run_name']}.png",
        regime=regime,
    )

    return {
        "report_path": str(report_path),
        "decision": decision,
        "next_agent": next_agent,
        "qwk_mean": qwk_mean,
        "summary_path": str(summary_candidates[0]),
        "rationale": rationale,
        "learning_curves_path": str(lc_path) if lc_path else None,
    }


def render_learning_curves(records: list[dict], out_path: Path, regime: str = "B") -> Path | None:
    """Plot per-seed train/val QWK curves and train-val gap (ADR-001 rev3).

    Reads per-epoch history attached to each fold_record by the trainer.
    `regime` selects which gating thresholds are drawn as reference lines
    (Rejim-A: 0.72/0.66/0.60 for P2-only; Rejim-B: 0.82/0.78/0.72 for 8-prompt avg).
    Returns path on success, None if history missing (old runs).
    """
    seeds_with_history = [r for r in records if r.get("history")]
    if not seeds_with_history:
        return None

    fig, (ax_qwk, ax_gap) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    cmap = plt.get_cmap("tab10")

    for i, rec in enumerate(seeds_with_history):
        hist = rec["history"]
        epochs = [h["epoch"] for h in hist]
        train_q = [h["train_qwk"] for h in hist]
        val_q = [h["val_qwk"] for h in hist]
        gap = [h["gap_qwk"] for h in hist]
        seed_label = f"seed={rec.get('seed', i)}"
        color = cmap(i % 10)
        ax_qwk.plot(epochs, train_q, linestyle="--", color=color, alpha=0.7,
                    label=f"{seed_label} train")
        ax_qwk.plot(epochs, val_q, linestyle="-", color=color, marker="o",
                    label=f"{seed_label} val")
        ax_gap.plot(epochs, gap, linestyle="-", color=color, marker="s",
                    label=seed_label)

    if regime == "A":
        go_t, iter_t, rev_t = 0.72, 0.66, 0.60
    else:
        go_t, iter_t, rev_t = 0.82, 0.78, 0.72
    ax_qwk.axhline(go_t, color="green", linestyle=":", linewidth=1, label=f"GO ({go_t:.2f})")
    ax_qwk.axhline(iter_t, color="orange", linestyle=":", linewidth=1, label=f"ITERATE ({iter_t:.2f})")
    ax_qwk.axhline(rev_t, color="red", linestyle=":", linewidth=1, label=f"REVISE ({rev_t:.2f})")
    ax_qwk.set_ylabel("QWK")
    ax_qwk.set_title("Learning Curves — Train vs Val QWK per Seed")
    ax_qwk.grid(True, alpha=0.3)
    ax_qwk.legend(loc="lower right", fontsize=8, ncol=2)

    ax_gap.axhline(0.0, color="black", linewidth=0.8)
    ax_gap.set_xlabel("Epoch")
    ax_gap.set_ylabel("train_qwk − val_qwk")
    ax_gap.set_title("Overfitting Diagnostic — Train/Val Gap")
    ax_gap.grid(True, alpha=0.3)
    ax_gap.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


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
