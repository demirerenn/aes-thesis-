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

from .losses import CORALLoss, CORNLoss, MSEPlusRankLoss, build_loss
from .metrics import all_metrics, qwk
from .models import AESModel, build_model, global_attention_for_longformer


# -------------------------------------------------------------------
# Dataset
# -------------------------------------------------------------------

class EssayDataset(Dataset):
    def __init__(self, texts: list[str], class_idx: list[int], score_norm: list[float],
                 tokenizer, max_length: int) -> None:
        self.texts = texts
        self.class_idx = class_idx
        self.score_norm = score_norm
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
        return {
            "input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
            "class_idx": torch.tensor(self.class_idx[idx], dtype=torch.long),
            "score_norm": torch.tensor(self.score_norm[idx], dtype=torch.float32),
        }


def collate(batch: list[dict[str, torch.Tensor]], pad_id: int) -> dict[str, torch.Tensor]:
    max_len = max(b["input_ids"].size(0) for b in batch)
    input_ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), max_len), dtype=torch.long)
    for i, b in enumerate(batch):
        n = b["input_ids"].size(0)
        input_ids[i, :n] = b["input_ids"]
        attention_mask[i, :n] = b["attention_mask"]
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "class_idx": torch.stack([b["class_idx"] for b in batch]),
        "score_norm": torch.stack([b["score_norm"] for b in batch]),
    }


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
                 min_class: int = 0) -> None:
        self.cfg = cfg
        self.prompt_denormalize = prompt_denormalize
        self.min_class = min_class  # for mapping class_idx → real score
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
        ds = EssayDataset(
            df["essay"].tolist(),
            df["class_idx"].tolist(),
            df["score_norm"].tolist(),
            self.tokenizer,
            self.cfg.max_length,
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

            val_metrics = self.evaluate(val_loader)
            took = time.time() - t0
            print(f"[epoch {epoch:02d}] loss={total/len(train_loader):.4f}  "
                  f"val_qwk={val_metrics['qwk']:.4f}  val_mae={val_metrics['mae']:.4f}  ({took:.0f}s)")

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

        If ``return_predictions`` is True, additionally return
        (y_true, y_pred) numpy arrays so the Evaluator Agent can build a
        confusion matrix and worst-prediction list.
        """
        self.model.eval()
        preds, gts = [], []
        for batch in loader:
            ids = batch["input_ids"].to(self.device, non_blocking=True)
            mask = batch["attention_mask"].to(self.device, non_blocking=True)
            gam = global_attention_for_longformer(ids) if self.cfg.longformer_global else None
            with torch.amp.autocast(device_type="cuda", dtype=self.cfg.autocast_dtype()):
                out = self.model(ids, mask, global_attention_mask=gam)
            yhat = self._predict(out).cpu().numpy()
            y = batch["class_idx"].numpy()
            preds.append(yhat); gts.append(y)
        yhat = np.concatenate(preds)
        y = np.concatenate(gts)
        metrics = all_metrics(y, yhat, labels=list(range(self.cfg.num_classes)))
        if return_predictions:
            return metrics, y, yhat
        return metrics

    def evaluate_df(self, df, return_predictions: bool = False):
        """Convenience wrapper: build loader from DataFrame and evaluate.

        When ``return_predictions`` is True, returns (metrics, y_true, y_pred).
        """
        loader = self._make_loader(df, shuffle=False)
        return self.evaluate(loader, return_predictions=return_predictions)

    def save_checkpoint(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": self.model.state_dict(), "cfg": self.cfg.__dict__}, path)
