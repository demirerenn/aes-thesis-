"""Ordinal losses.

Implements ADR-001-rev1 §4.3:
    * CORN (Shi et al. 2022) — primary loss
    * CORAL (Cao et al. 2020) — fallback Plan C
    * MSE + pairwise-rank (R²BERT, Yang 2020) — fallback Plan B

Reference:
    Shi, X., Cao, W., & Raschka, S. (2022).
    Deep Neural Networks for Rank-Consistent Ordinal Regression
    Based on Conditional Probabilities.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------------------------
# CORN (primary)
# -------------------------------------------------------------------

class CORNLoss(nn.Module):
    """Conditional Ordinal Regression for Neural networks.

    Head output shape: (B, K-1) logits for P(y > k | y > k-1).
    Target shape:       (B,) class index in [0, K-1].
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        assert logits.ndim == 2 and logits.shape[1] == self.num_classes - 1, (
            f"CORN expects logits of shape (B, K-1={self.num_classes-1}), got {tuple(logits.shape)}"
        )
        # Build binary labels y_k = 1[target > k] for k = 0..K-2
        B = logits.size(0)
        K = self.num_classes
        device = logits.device

        # y_k[i] = 1 if targets[i] > k else 0
        ks = torch.arange(K - 1, device=device).unsqueeze(0)       # (1, K-1)
        y_k = (targets.unsqueeze(1) > ks).float()                   # (B, K-1)

        # CORN masks: the i-th sample contributes to the k-th task only while
        # y_{k-1} = 1 (i.e. target > k-1). For k=0 every sample contributes.
        mask = torch.ones_like(y_k)
        # Shift y_k left by 1 to get "previous task was positive" flag
        mask[:, 1:] = y_k[:, :-1]

        bce = F.binary_cross_entropy_with_logits(logits, y_k, reduction="none")
        loss = (bce * mask).sum() / mask.sum().clamp_min(1.0)
        return loss

    @staticmethod
    @torch.no_grad()
    def predict_label(logits: torch.Tensor) -> torch.Tensor:
        """Infer integer class index y_pred = Σ_{k} σ(logit_k) (expected-value formulation).

        Rounded to nearest int (ADR-001-rev1 §4.3 talep 13a-1).
        """
        probs = torch.sigmoid(logits)             # (B, K-1)  P(y>k | y>k-1)
        # Cumulative product gives P(y>k) unconditionally
        cum = torch.cumprod(probs, dim=1)         # (B, K-1)  P(y>k)
        y = cum.sum(dim=1)                        # (B,)      E[y] in [0, K-1]
        return torch.round(y).long()


# -------------------------------------------------------------------
# CORAL (fallback Plan C)
# -------------------------------------------------------------------

class CORALLoss(nn.Module):
    """Rank-Consistent Ordinal Regression (Cao et al. 2020).

    Weight-shared head; bias vector of size K-1.
    Head output shape: (B, K-1) logits.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B = logits.size(0)
        K = self.num_classes
        device = logits.device
        ks = torch.arange(K - 1, device=device).unsqueeze(0)
        y_k = (targets.unsqueeze(1) > ks).float()
        return F.binary_cross_entropy_with_logits(logits, y_k, reduction="mean")

    @staticmethod
    @torch.no_grad()
    def predict_label(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        return (probs > 0.5).long().sum(dim=1)


# -------------------------------------------------------------------
# MSE + pairwise rank (R²BERT — fallback Plan B)
# -------------------------------------------------------------------

class MSEPlusRankLoss(nn.Module):
    """Combined MSE regression + margin-ranking loss.

    Expects scalar regression output in (B, 1) and scalar targets in (B,)
    already normalized to [0, 1].
    """

    def __init__(self, mse_weight: float = 1.0, rank_weight: float = 0.5, margin: float = 0.01) -> None:
        super().__init__()
        self.mse_weight = mse_weight
        self.rank_weight = rank_weight
        self.margin = margin
        self.mse = nn.MSELoss()

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        preds = preds.squeeze(-1)
        mse_loss = self.mse(preds, targets)

        # Build all ordered pairs within the batch
        diff_pred = preds.unsqueeze(1) - preds.unsqueeze(0)     # (B, B)
        diff_targ = targets.unsqueeze(1) - targets.unsqueeze(0)
        sign = torch.sign(diff_targ)                             # +1 / -1 / 0
        # MarginRanking: max(0, -sign * diff_pred + margin)
        rank_loss = F.relu(-sign * diff_pred + self.margin)
        # Only keep pairs with nonzero sign (strict ordering)
        mask = (sign.abs() > 0).float()
        rank_loss = (rank_loss * mask).sum() / mask.sum().clamp_min(1.0)

        return self.mse_weight * mse_loss + self.rank_weight * rank_loss


# -------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------

def build_loss(name: str, num_classes: int) -> nn.Module:
    name = name.lower()
    if name == "corn":
        return CORNLoss(num_classes)
    if name == "coral":
        return CORALLoss(num_classes)
    if name in {"mse_rank", "r2bert"}:
        return MSEPlusRankLoss()
    raise ValueError(f"Unknown loss: {name}")
