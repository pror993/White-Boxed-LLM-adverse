"""Mellowmax loss function for discrete token optimization.

Implements the Mellowmax operator over token-level cross-entropy losses:
    L_mellow = (1 / alpha) * log( (1 / |Y|) * sum_{t=1}^{|Y|} exp(alpha * L_t) )

Smoothly interpolates between arithmetic mean cross-entropy (alpha -> 0) and
worst-case maximum token loss (alpha -> infinity), mitigating token-level loss plateaus.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


def mellowmax_loss(
    token_losses: torch.Tensor,
    alpha: float = 1.0,
    dim: int = -1,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Compute the numerically stable Mellowmax loss over a tensor of token losses.

    Args:
        token_losses: Per-token cross-entropy losses of shape `[..., seq_len]`.
        alpha: Temperature / stiffness parameter (default 1.0).
               alpha -> 0 approaches arithmetic mean loss.
               alpha -> inf approaches maximum loss.
        dim: Dimension over which to aggregate token losses.
        eps: Small epsilon to guard against division by zero when alpha -> 0.

    Returns:
        Tensor of aggregated Mellowmax losses with dimension `dim` reduced.
    """
    if abs(alpha) < eps:
        # L'Hopital limit as alpha -> 0 is the arithmetic mean
        return token_losses.mean(dim=dim)

    num_tokens = token_losses.size(dim)
    scaled_losses = alpha * token_losses

    # Numerically stable formulation: (logsumexp(alpha * L) - log(num_tokens)) / alpha
    log_sum_exp = torch.logsumexp(scaled_losses, dim=dim)
    mellow = (log_sum_exp - math.log(num_tokens)) / alpha
    return mellow


class MellowmaxLoss(nn.Module):
    """PyTorch loss module computing Mellowmax over sequence cross-entropy predictions."""

    def __init__(self, alpha: float = 1.0, eps: float = 1e-4):
        super().__init__()
        self.alpha = alpha
        self.eps = eps
        self.base_ce = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        dim: int = -1,
    ) -> torch.Tensor:
        """Calculate Mellowmax loss given prediction logits and target labels.

        Args:
            logits: Prediction logits of shape `[batch, target_len, vocab_size]`
                    or `[target_len, vocab_size]`.
            labels: Ground truth token indices of shape `[batch, target_len]`
                    or `[target_len]`.
            dim: Sequence dimension for aggregation.

        Returns:
            Scalar loss or per-batch sequence losses.
        """
        # Ensure 3D logits: [batch, target_len, vocab_size]
        is_batched = logits.dim() == 3
        if not is_batched:
            logits = logits.unsqueeze(0)
            labels = labels.unsqueeze(0)

        batch_sz, target_len, vocab_sz = logits.shape
        flat_logits = logits.view(-1, vocab_sz)
        flat_labels = labels.view(-1)

        flat_losses = self.base_ce(flat_logits, flat_labels)
        token_losses = flat_losses.view(batch_sz, target_len)

        per_seq_mellow = mellowmax_loss(token_losses, alpha=self.alpha, dim=-1, eps=self.eps)

        if not is_batched:
            return per_seq_mellow.squeeze(0)
        return per_seq_mellow
