"""Backbone + ordinal head models.

ADR-001-rev1 §4.2 backbone decisions:
    * short / medium  → microsoft/deberta-v3-large
    * long            → allenai/longformer-large-4096

Head variants (determined by loss):
    * corn / coral  → Linear(hidden, K-1) with mean-pooled [CLS]
    * mse_rank      → Linear(hidden, 1) sigmoid output for [0, 1] regression
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel


@dataclass(frozen=True)
class ModelSpec:
    backbone: str
    num_classes: int
    head: str  # "ordinal" (K-1 logits) | "regression" (1 scalar)
    dropout: float = 0.1
    use_pooler: bool = False

    @property
    def head_dim(self) -> int:
        return self.num_classes - 1 if self.head == "ordinal" else 1


class AESModel(nn.Module):
    """Generic AES wrapper: backbone → dropout → linear head."""

    def __init__(self, spec: ModelSpec) -> None:
        super().__init__()
        self.spec = spec
        config = AutoConfig.from_pretrained(spec.backbone)
        self.backbone = AutoModel.from_pretrained(spec.backbone, config=config)
        self.dropout = nn.Dropout(spec.dropout)
        self.head = nn.Linear(config.hidden_size, spec.head_dim)
        self._init_head()

    def _init_head(self) -> None:
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def _pool(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool non-padding tokens; fall back to [CLS] if mask is None."""
        if attention_mask is None:
            return hidden[:, 0]
        mask = attention_mask.unsqueeze(-1).float()
        summed = (hidden * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp_min(1.0)
        return summed / counts

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        global_attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if global_attention_mask is not None:
            kwargs["global_attention_mask"] = global_attention_mask
        out = self.backbone(**kwargs)
        pooled = self._pool(out.last_hidden_state, attention_mask)
        logits = self.head(self.dropout(pooled))
        if self.spec.head == "regression":
            return torch.sigmoid(logits)  # map to [0, 1]
        return logits


def build_model(backbone: str, num_classes: int, loss_name: str, dropout: float = 0.1) -> AESModel:
    loss_name = loss_name.lower()
    if loss_name in {"corn", "coral"}:
        head = "ordinal"
    elif loss_name in {"mse_rank", "r2bert"}:
        head = "regression"
    else:
        raise ValueError(f"Unknown loss for head resolution: {loss_name}")
    spec = ModelSpec(backbone=backbone, num_classes=num_classes, head=head, dropout=dropout)
    return AESModel(spec)


def global_attention_for_longformer(input_ids: torch.Tensor) -> torch.Tensor:
    """Global attention on [CLS] token only (index 0)."""
    gam = torch.zeros_like(input_ids)
    gam[:, 0] = 1
    return gam
