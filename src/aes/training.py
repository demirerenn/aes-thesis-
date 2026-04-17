"""Single-run trainer — orchestrates data / model / loss / metrics.

Designed for the ASUS Ascent GX10 target:
    * BF16 autocast (GB10 native)
    * Gradient accumulation for effective batch control
    * torch.compile for DeBERTa, Flash-Attn-2 for Longformer (§4.7)
    * MLflow logging per epoch; best-QWK checkpoint retention
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from .data import ASAP1_PROMPTS, Prompt
from .losses import CORALLoss, CORNLoss, MSEPlusRankLoss, build_loss
from .metrics import all_metrics, qwk
from .models import AESModel, build_model, global_attention_for_longformer


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------

class EssayDataset(Dataset):
    def __init__(self, texts: list[str], class_idx: list[int], score_norm: list[float],
                 tokenizer, max_length: int,
                 essay_set: list[int] | None = None,
                 score_raw: list[int] | None = None) -> None:
        self.texts = texts
        self.class_idx = class_idx
        self.score_norm = score_norm
        self.essay_set = essay_set
        self.score_raw = score_raw
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        item = {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "class_idx": torch.tensor(self.class_idx[idx], dtype=torch.long),
            "score_norm": torch.tensor(self.score_norm[idx], dtype=torch.float32),
        }
        if self.essay_set is not None:
            item["essay_set"] = torch.tensor(self.essay_set[idx], dtype=torch.long)
        if self.score_raw is not None:
            item["score_raw"] = torch.tensor(self.score_raw[idx], dtype=torch.long)
        return item


def collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].size(0)
        input_ids[i, :n] = b["input_ids"]
        attention_mask[i, :n] = b["attention_mask"]
    out = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "class_idx": torch.stack([b["class_idx"] for b in batch]),
        "score_norm": torch.stack([b["score_norm"] for b in batch]),
    }
    if "essay_set" in batch[0]:
        out["essay_set"] = torch.stack([b["essay_set"] for b in batch])
    if "score_raw" in batch[0]:
        out["score_raw"] = torch.stack([b["score_raw"] for b in batch])
    return out


# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------

@dataclass
class TrainConfig:
    run_name: str
    backbone: str
    loss: str               # "corn" | "coral" | "mse_rank"
    num_classes: int
    max_length: int
    batch_size: int
    grad_accum: int
    lr: float
    epochs: int
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout: float = 0.1
    precision: str = "bf16"
    compile_backbone: bool = True
    longformer_global: bool = False
    early_stop_patience: int = 2
    seed: int = 42
    num_workers: int = 4
    grad_clip: float = 1.0
    output_dir: str = "runs"
    mlflow_uri: str = "file:./mlruns"

    def autocast_dtype(self) -> torch.dtype:
        return {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.precision]


# -------------------------------------------------------------------
# Trainer
# -------------------------------------------------------------------

class Trainer:
    def __init__(self, cfg: TrainConfig, prompt_denormalize: Callable[[np.ndarray], np.ndarray] | None = None,
                 min_class: int = 0,
                 prompt_map: dict[int, Prompt] | None = None) -> None:
        self.cfg = cfg
        self.prompt_denormalize = prompt_denormalize
        self.min_class = min_class  # for mapping class_idx → real score
        self.prompt_map = prompt_map  # multi-prompt evaluation hook (ADR rev3.2 §9)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(cfg.backbone, use_fast=True)
        self.model: AESModel = build_model(cfg.backbone, cfg.num_classes, cfg.loss, cfg.dropout).to(self.device)
        self.criterion: nn.Module = build_loss(cfg.loss, cfg.num_classes).to(self.device)
        self.scaler = torch.amp.GradScaler(enabled=(cfg.precision == "fp16"))

        if cfg.compile_backbone and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead")
            except Exception:
                pass  # Flash-Attn or kernel incompat: fall back silently

    # ---------------- helpers ----------------

    def _make_loader(self, df, shuffle: bool) -> DataLoader:
        essay_set = df["essay_set"].tolist() if "essay_set" in df.columns and self.prompt_map else None
        score_raw = df["score"].tolist() if "score" in df.columns and self.prompt_map else None
        ds = EssayDataset(
            df["essay"].tolist(),
            df["class_idx"].tolist(),
            df["score_norm"].tolist(),
            self.tokenizer,
            self.cfg.max_length,
            essay_set=essay_set,
            score_raw=score_raw,
        )
        pad_id = self.tokenizer.pad_token_id or 0
        return DataLoader(
            ds,
            batch_size=self.cfg.batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.num_workers,
            collate_fn=lambda b: collate(b, pad_id),
            pin_memory=True,
        )

    def _compute_loss(self, batch: dict[str, torch.Tensor], out: torch.Tensor) -> torch.Tensor:
        if isinstance(self.criterion, (CORNLoss, CORALLoss)):
            return self.criterion(out, batch["class_idx"].to(self.device))
        if isinstance(self.criterion, MSEPlusRankLoss):
            return self.criterion(out, batch["score_norm"].to(self.device))
        raise TypeError(f"Unsupported criterion: {type(self.criterion)}")

    def _predict(self, out: torch.Tensor) -> torch.Tensor:
        if isinstance(self.criterion, CORNLoss):
            return CORNLoss.predict_label(out)
        if isinstance(self.criterion, CORALLoss):
            return CORALLoss.predict_label(out)
        # Regression (MSE+rank): sigmoid [0,1] → map to [0,K-1] then round
        y = out.squeeze(-1) * (self.cfg.num_classes - 1)
        return torch.round(y).long()

    # ---------------- training ----------------

    def fit(self, train_df, val_df) -> dict[str, float]:
        train_loader = self._make_loader(train_df, shuffle=True)
        val_loader = self._make_loader(val_df, shuffle=False)
        # Separate no-shuffle loader for per-epoch train-QWK measurement
        # (used as overfitting diagnostic; inference-only, no gradient).
        train_eval_loader = self._make_loader(train_df, shuffle=False)

        total_steps = math.ceil(len(train_loader) / self.cfg.grad_accum) * self.cfg.epochs
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)

        no_decay = ("bias", "LayerNorm.weight")
        params = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             "weight_decay": self.cfg.weight_decay},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             "weight_decay": 0.0},
        ]
        optim = torch.optim.AdamW(params, lr=self.cfg.lr)
        sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

        best_qwk = -1.0
        best_state = None
        patience_left = self.cfg.early_stop_patience
        self.history: list[dict[str, float]] = []

        for epoch in range(1, self.cfg.epochs + 1):
            self.model.train()
            t0 = time.time()
            total = 0.0
            optim.zero_grad(set_to_none=True)
            for step, batch in enumerate(train_loader, 1):
                ids = batch["input_ids"].to(self.device, non_blocking=True)
                mask = batch["attention_mask"].to(self.device, non_blocking=True)
                gam = global_attention_for_longformer(ids) if self.cfg.longformer_global else None

                with torch.amp.autocast(device_type="cuda", dtype=self.cfg.autocast_dtype()):
                    out = self.model(ids, mask, global_attention_mask=gam)
                    loss = self._compute_loss(batch, out) / self.cfg.grad_accum

                if self.cfg.precision == "fp16":
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()

                total += loss.item() * self.cfg.grad_accum

                if step % self.cfg.grad_accum == 0:
                    if self.cfg.grad_clip:
                        if self.cfg.precision == "fp16":
                            self.scaler.unscale_(optim)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
                    if self.cfg.precision == "fp16":
                        self.scaler.step(optim); self.scaler.update()
                    else:
                        optim.step()
                    sched.step()
                    optim.zero_grad(set_to_none=True)

            train_metrics = self.evaluate(train_eval_loader)
            val_metrics = self.evaluate(val_loader)
            took = time.time() - t0
            gap = train_metrics["qwk"] - val_metrics["qwk"]
            print(f"[epoch {epoch:02d}] loss={total/len(train_loader):.4f}  "
                  f"train_qwk={train_metrics['qwk']:.4f}  val_qwk={val_metrics['qwk']:.4f}  "
                  f"gap={gap:+.4f}  val_mae={val_metrics['mae']:.4f}  ({took:.0f}s)")
            self.history.append({
                "epoch": epoch,
                "train_loss": total / len(train_loader),
                "train_qwk": float(train_metrics["qwk"]),
                "train_mae": float(train_metrics["mae"]),
                "val_qwk": float(val_metrics["qwk"]),
                "val_mae": float(val_metrics["mae"]),
                "gap_qwk": float(gap),
                "epoch_sec": float(took),
            })

            if val_metrics["qwk"] > best_qwk:
                best_qwk = val_metrics["qwk"]
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                patience_left = self.cfg.early_stop_patience
            else:
                patience_left -= 1
                if patience_left < 0:
                    print(f"[early stop] no improvement after {self.cfg.early_stop_patience} epochs")
                    break

        if best_state is not None:
            self.model.load_state_dict(best_state)
        return self.evaluate(val_loader)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader,
                 return_predictions: bool = False) -> dict[str, float] | tuple[dict[str, float], np.ndarray, np.ndarray]:
        """Run evaluation loop.

        When ``self.prompt_map`` is set (multi-prompt run), predictions are
        denormalized per-prompt and QWK is computed per-prompt then averaged
        (ADR-001 rev3.2 §9 — literatür-uyumlu protokol).

        If ``return_predictions`` is True, additionally return
        (y_true, y_pred) numpy arrays (raw integer scores) so the Evaluator
        Agent can build a confusion matrix and worst-prediction list.
        """
        self.model.eval()
        pred_norm_chunks: list[np.ndarray] = []
        pred_class_chunks: list[np.ndarray] = []
        y_class_chunks: list[np.ndarray] = []
        essay_set_chunks: list[np.ndarray] = []
        score_raw_chunks: list[np.ndarray] = []

        for batch in loader:
            ids = batch["input_ids"].to(self.device, non_blocking=True)
            mask = batch["attention_mask"].to(self.device, non_blocking=True)
            gam = global_attention_for_longformer(ids) if self.cfg.longformer_global else None
            with torch.amp.autocast(device_type="cuda", dtype=self.cfg.autocast_dtype()):
                out = self.model(ids, mask, global_attention_mask=gam)
            y_class_chunks.append(batch["class_idx"].numpy())
            # Multi-prompt regression uses raw sigmoid output for per-prompt denormalization.
            is_regression = isinstance(self.criterion, MSEPlusRankLoss)
            if self.prompt_map is not None and is_regression:
                pred_norm_chunks.append(out.squeeze(-1).float().cpu().numpy())
            else:
                pred_class_chunks.append(self._predict(out).cpu().numpy())
            if "essay_set" in batch:
                essay_set_chunks.append(batch["essay_set"].numpy())
            if "score_raw" in batch:
                score_raw_chunks.append(batch["score_raw"].numpy())

        y_class = np.concatenate(y_class_chunks)
        yhat_class = np.concatenate(pred_class_chunks) if pred_class_chunks else np.zeros_like(y_class)

        if self.prompt_map is not None and essay_set_chunks and score_raw_chunks:
            essay_set = np.concatenate(essay_set_chunks)
            y_raw = np.concatenate(score_raw_chunks)
            if pred_norm_chunks:
                pred_norm = np.concatenate(pred_norm_chunks)
                yhat_raw = self._denormalize_per_prompt(pred_norm, essay_set)
            else:
                # Ordinal head multi-prompt fallback: map class_idx → raw score by adding min_score.
                yhat_raw = np.array(
                    [yhat_class[i] + self.prompt_map[int(es)].min_score for i, es in enumerate(essay_set)],
                    dtype=np.int64,
                )
            metrics = self._multi_prompt_metrics(y_raw, yhat_raw, essay_set)
            if return_predictions:
                return metrics, y_raw, yhat_raw
            return metrics

        metrics = all_metrics(y_class, yhat_class, labels=list(range(max(2, self.cfg.num_classes))))
        if return_predictions:
            return metrics, y_class, yhat_class
        return metrics

    def _denormalize_per_prompt(self, pred_norm: np.ndarray, essay_set: np.ndarray) -> np.ndarray:
        """Map sigmoid output in [0,1] → integer raw score per prompt."""
        out = np.empty_like(pred_norm, dtype=np.int64)
        for i, es in enumerate(essay_set):
            p = self.prompt_map[int(es)]
            raw = pred_norm[i] * (p.max_score - p.min_score) + p.min_score
            out[i] = int(np.clip(np.round(raw), p.min_score, p.max_score))
        return out

    def _multi_prompt_metrics(self, y_raw: np.ndarray, yhat_raw: np.ndarray,
                              essay_set: np.ndarray) -> dict[str, float]:
        """Per-prompt QWK → average (literature convention)."""
        per_prompt: dict[int, float] = {}
        mae_total, rmse_total_sq, n_total = 0.0, 0.0, 0
        for es in np.unique(essay_set):
            mask = essay_set == es
            p = self.prompt_map[int(es)]
            labels = list(range(p.min_score, p.max_score + 1))
            per_prompt[int(es)] = float(qwk(y_raw[mask], yhat_raw[mask], labels=labels))
            mae_total += float(np.abs(y_raw[mask] - yhat_raw[mask]).sum())
            rmse_total_sq += float(((y_raw[mask] - yhat_raw[mask]) ** 2).sum())
            n_total += int(mask.sum())
        qwk_avg = float(np.mean(list(per_prompt.values())))
        return {
            "qwk": qwk_avg,
            "mae": mae_total / max(1, n_total),
            "rmse": math.sqrt(rmse_total_sq / max(1, n_total)),
            "pearson_r": 0.0,
            "spearman_rho": 0.0,
            "macro_f1": 0.0,
            "per_prompt_qwk": per_prompt,
        }

    def evaluate_df(self, df, return_predictions: bool = False):
        """Convenience wrapper: build loader from DataFrame and evaluate.

        When ``return_predictions`` is True, returns (metrics, y_true, y_pred).
        """
        loader = self._make_loader(df, shuffle=False)
        return self.evaluate(loader, return_predictions=return_predictions)

    def save_checkpoint(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "cfg": self.cfg.__dict__}, path)
