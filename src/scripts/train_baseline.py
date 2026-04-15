"""Baseline training driver.

Usage:
    python -m src.scripts.train_baseline --config configs/baseline_asap1_p2.yaml

Orchestrates ADR-001-rev1 §5 workflow:
    1. Load YAML config
    2. Verify data hashes (13c-2) and environment snapshot (13c-1)
    3. Load dataset, build stratified 5-fold splits
    4. For each (fold, seed) → train + evaluate → log to MLflow
    5. Aggregate fold-level QWK with bootstrap CI (13a-2)
    6. Write report_<run_name>.md from Report_Template_v1.md

This file is wall-clock heavy (≈ 7 h on GX10 per ADR §5).
Designed to be called by the Training Engineer Agent node in LangGraph.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml

# Make the package importable when invoked from project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.aes.data import (  # noqa: E402
    ASAP1_PROMPTS, fixed_split, iter_folds, load_asap1, load_asap2, stratified_folds,
)
from src.aes.metrics import bootstrap_ci_fold_level  # noqa: E402
from src.aes.training import TrainConfig, Trainer  # noqa: E402
from src.aes.utils import capture_env, data_hashes, set_seed  # noqa: E402


# -------------------------------------------------------------------
# Config loading
# -------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_train_cfg(cfg: dict, seed: int) -> TrainConfig:
    t = cfg["training"]
    return TrainConfig(
        run_name=cfg["run_name"],
        backbone=cfg["model"]["backbone"],
        loss=cfg["model"]["loss"],
        num_classes=cfg["model"]["num_classes"],
        max_length=t["max_length"],
        batch_size=t["batch_size"],
        grad_accum=t["grad_accum"],
        lr=float(t["lr"]),
        epochs=t["epochs"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        dropout=cfg["model"].get("dropout", 0.1),
        precision=t["precision"],
        compile_backbone=t.get("compile_backbone", True),
        longformer_global=t.get("longformer_global", False),
        early_stop_patience=t.get("early_stop_patience", 2),
        seed=seed,
        num_workers=t.get("num_workers", 4),
        grad_clip=t.get("grad_clip", 1.0),
        output_dir=cfg["output"]["dir"],
        mlflow_uri=cfg["tracking"]["mlflow_uri"],
    )


# -------------------------------------------------------------------
# Main orchestrator
# -------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="AES baseline training driver")
    ap.add_argument("--config", required=True, help="YAML config path")
    ap.add_argument("--data-dir", default="data", help="Data directory (default: data/)")
    ap.add_argument("--smoke", action="store_true",
                    help="Smoke run — 1 seed × 1 epoch × tiny batch (dry-run, §4.7 talep 13b-1)")
    ap.add_argument("--mode", choices=["auto", "pilot", "cv"], default="auto",
                    help="Evaluation mode. 'auto' reads cfg.eval.mode (default). "
                         "'pilot' = fixed stratified train/dev/test × N seeds (ADR-001-rev2). "
                         "'cv' = 5-fold × N seeds (original ADR-001-rev1 §5).")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    run_name = cfg["run_name"]
    print(f"\n=== {run_name} ===\n")

    # ---------------- Environment & data hashes ----------------
    env = capture_env()
    hashes = data_hashes(args.data_dir)
    print("Environment:", json.dumps(env.as_mlflow_params(), indent=2))
    print("Data hashes:", json.dumps(hashes, indent=2))

    # ---------------- MLflow setup ----------------
    try:
        import mlflow
        mlflow.set_tracking_uri(cfg["tracking"]["mlflow_uri"])
        mlflow.set_experiment(cfg["tracking"]["experiment"])
    except ImportError:
        mlflow = None
        print("[WARN] mlflow not installed; tracking disabled")

    # ---------------- Data ----------------
    ds_spec = cfg["dataset"]
    data_path = Path(args.data_dir) / Path(ds_spec["path"]).name
    if ds_spec["name"] == "asap1":
        df = load_asap1(data_path, prompts=ds_spec.get("prompts"))
    elif ds_spec["name"] == "asap2":
        df = load_asap2(data_path)
    else:
        raise ValueError(f"Unknown dataset: {ds_spec['name']}")
    print(f"Loaded {len(df)} essays from {ds_spec['name']}")

    # ---------------- Evaluation mode resolution ----------------
    mode = args.mode
    if mode == "auto":
        mode = cfg.get("eval", {}).get("mode", "cv")
    print(f"Evaluation mode: {mode}")

    output_dir = Path(cfg["output"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    all_obs: list[float] = []         # one QWK per observation (seed × fold or seed)
    fold_records: list[dict] = []

    def _run_one(tr_df, va_df, te_df, seed: int, tag: str) -> dict:
        """Train one model and evaluate on dev (val) + optional test."""
        print(f"\n--- {run_name} | {tag} | train={len(tr_df)} dev={len(va_df)}"
              + (f" test={len(te_df)}" if te_df is not None else "") + " ---")
        tcfg = build_train_cfg(cfg, seed=seed)
        if args.smoke:
            tcfg.epochs = 1
            tcfg.batch_size = 4
            tcfg.grad_accum = 1
            tcfg.compile_backbone = False

        trainer = Trainer(tcfg)
        if mlflow:
            with mlflow.start_run(run_name=f"{run_name}-{tag}"):
                mlflow.log_params(env.as_mlflow_params())
                mlflow.log_params(hashes)
                mlflow.log_params({
                    "cfg.run_name": run_name,
                    "cfg.mode": mode,
                    "cfg.tag": tag,
                    "cfg.seed": seed,
                    "cfg.backbone": tcfg.backbone,
                    "cfg.loss": tcfg.loss,
                    "cfg.num_classes": tcfg.num_classes,
                    "cfg.max_length": tcfg.max_length,
                    "cfg.batch_size": tcfg.batch_size,
                    "cfg.grad_accum": tcfg.grad_accum,
                    "cfg.lr": tcfg.lr,
                    "cfg.epochs": tcfg.epochs,
                    "cfg.precision": tcfg.precision,
                })
                dev_metrics = trainer.fit(tr_df, va_df)
                for k, v in dev_metrics.items():
                    mlflow.log_metric(f"dev.{k}", float(v))
                if te_df is not None:
                    test_metrics, y_true, y_pred = trainer.evaluate_df(te_df, return_predictions=True)
                    for k, v in test_metrics.items():
                        mlflow.log_metric(f"test.{k}", float(v))
                    dev_metrics["test_qwk"] = float(test_metrics["qwk"])
                    dev_metrics["test_mae"] = float(test_metrics.get("mae", 0.0))
                    # Persist predictions for Evaluator error analysis
                    pred_path = output_dir / f"preds-{tag}.csv"
                    import pandas as pd
                    preds_df = pd.DataFrame({
                        "essay_id": te_df.get("essay_id", pd.Series(range(len(te_df)))).values,
                        "essay_set": te_df.get("essay_set", pd.Series([0]*len(te_df))).values,
                        "y_true": y_true,
                        "y_pred": y_pred,
                        "abs_err": np.abs(y_true - y_pred),
                        "essay_len_words": te_df["essay"].str.split().str.len().values,
                    })
                    preds_df.to_csv(pred_path, index=False)
                    mlflow.log_artifact(str(pred_path))
        else:
            dev_metrics = trainer.fit(tr_df, va_df)
            if te_df is not None:
                test_metrics, y_true, y_pred = trainer.evaluate_df(te_df, return_predictions=True)
                dev_metrics["test_qwk"] = float(test_metrics["qwk"])
                dev_metrics["test_mae"] = float(test_metrics.get("mae", 0.0))
                # Dump per-sample predictions for Evaluator Agent error analysis
                pred_path = output_dir / f"preds-{tag}.csv"
                import pandas as pd
                preds_df = pd.DataFrame({
                    "essay_id": te_df.get("essay_id", pd.Series(range(len(te_df)))).values,
                    "essay_set": te_df.get("essay_set", pd.Series([0]*len(te_df))).values,
                    "y_true": y_true,
                    "y_pred": y_pred,
                    "abs_err": np.abs(y_true - y_pred),
                    "essay_len_words": te_df["essay"].str.split().str.len().values,
                })
                preds_df.to_csv(pred_path, index=False)
                print(f"  → saved {pred_path}")

        ckpt_path = output_dir / f"ckpt-{tag}.pt"
        trainer.save_checkpoint(ckpt_path)
        print(f"  → saved {ckpt_path}")
        return dev_metrics

    if mode == "pilot":
        # ---- Pilot Phase (ADR-001-rev2): fixed split × N training seeds ----
        ev = cfg["eval"]
        split_seed = int(ev.get("split_seed", 42))
        ratios = tuple(ev.get("split_ratios", [0.70, 0.15, 0.15]))
        seeds = ev["seeds"]
        if args.smoke:
            seeds = seeds[:1]

        tr_df, dev_df, te_df = fixed_split(df, ratios=ratios, seed=split_seed)
        print(f"Fixed split (seed={split_seed}, ratios={ratios}): "
              f"train={len(tr_df)} dev={len(dev_df)} test={len(te_df)}")

        for seed in seeds:
            set_seed(seed)
            tag = f"pilot-s{seed}"
            metrics = _run_one(tr_df, dev_df, te_df, seed=seed, tag=tag)
            fold_records.append({"seed": seed, "fold": None, **metrics})
            all_obs.append(metrics["qwk"])

    else:
        # ---- CV Phase (ADR-001-rev1 §5): 5-fold × N seeds ----
        seeds = cfg["cv"]["seeds"]
        n_splits = cfg["cv"]["n_splits"]
        if args.smoke:
            seeds = seeds[:1]
            n_splits = 2

        for seed in seeds:
            set_seed(seed)
            folds = stratified_folds(df, n_splits=n_splits, seed=seed)
            for fold_i, tr_df, va_df in iter_folds(df, folds):
                tag = f"cv-s{seed}-f{fold_i}"
                metrics = _run_one(tr_df, va_df, None, seed=seed, tag=tag)
                fold_records.append({"seed": seed, "fold": fold_i, **metrics})
                all_obs.append(metrics["qwk"])

    # ---------------- Aggregate ----------------
    obs = np.asarray(all_obs)
    # Bootstrap CI is meaningful for n ≥ 5. In pilot mode with 3 seeds we
    # still report it, but also log mean ± std for transparency.
    if obs.size >= 3:
        mu, lo, hi = bootstrap_ci_fold_level(obs, n_iter=1000)
    else:
        mu, lo, hi = float(obs.mean()), float(obs.min()), float(obs.max())
    summary = {
        "run_name": run_name,
        "mode": mode,
        "n_observations": int(obs.size),
        "qwk_mean": float(mu),
        "qwk_std":  float(obs.std(ddof=1)) if obs.size > 1 else 0.0,
        "qwk_min":  float(obs.min()),
        "qwk_max":  float(obs.max()),
        "qwk_ci_lo": float(lo),
        "qwk_ci_hi": float(hi),
        "fold_records": fold_records,
        "env": env.as_mlflow_params(),
        "data_hashes": hashes,
    }
    out_json = output_dir / f"summary_{run_name}.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"\n=== Summary ===\nQWK = {mu:.4f}  [{lo:.4f}, {hi:.4f}]  (n={obs.size})\nWritten: {out_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
