"""Linguistic fluency and sequence perplexity regularizers for discrete optimization.

Provides sequence perplexity (PPL) and negative log-likelihood (NLL) estimators
to evaluate and penalize nonsensical token combinations along the Pareto frontier
of adversarial robustness vs. linguistic naturalness.
"""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn


class PerplexityRegularizer(nn.Module):
    """Measures and penalizes high sequence perplexity of discrete tokens."""

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        use_log_ppl: bool = True,
        clamp_max_nll: float = 20.0,
    ):
        super().__init__()
        self.model = model
        self.use_log_ppl = use_log_ppl
        self.clamp_max_nll = clamp_max_nll
        self.loss_fct = nn.CrossEntropyLoss(reduction="none")

    def compute_sequence_nll(
        self,
        token_ids: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Compute the average per-token negative log-likelihood of a token sequence.

        Args:
            token_ids: Discrete token IDs of shape `[batch, seq_len]` or `[1, seq_len]`.
            model: Optional model override.

        Returns:
            Tensor of scalar NLL values per sequence `[batch]`.
        """
        lm = model or self.model
        assert lm is not None, "Model must be provided to compute sequence NLL."

        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        batch_sz, seq_len = token_ids.shape
        if seq_len < 2:
            return torch.zeros(batch_sz, device=token_ids.device)

        with torch.no_grad():
            outputs = lm(token_ids)
            logits = outputs.logits  # [batch, seq_len, vocab_size]

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = token_ids[:, 1:].contiguous()

            _, pred_len, vocab_sz = shift_logits.shape
            flat_nll = self.loss_fct(
                shift_logits.view(-1, vocab_sz),
                shift_labels.view(-1),
            )
            seq_nll = flat_nll.view(batch_sz, pred_len).mean(dim=1)
            seq_nll = torch.clamp(seq_nll, max=self.clamp_max_nll)

        return seq_nll

    def compute_perplexity(
        self,
        token_ids: torch.Tensor,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Compute perplexity = exp(NLL) for token sequences."""
        nll = self.compute_sequence_nll(token_ids, model=model)
        return torch.exp(nll)

    def forward(
        self,
        target_loss: torch.Tensor,
        opt_token_ids: torch.Tensor,
        lambda_weight: float = 0.0,
        model: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Regularize target objective loss with sequence perplexity.

        Args:
            target_loss: Primary objective loss `[batch]` or scalar.
            opt_token_ids: Optimizable sequence token IDs `[batch, opt_len]`.
            lambda_weight: Weight coefficient for perplexity penalty.
            model: Language model to evaluate perplexity.

        Returns:
            Combined regularized loss.
        """
        if lambda_weight <= 0.0:
            return target_loss

        fluency_penalty = (
            self.compute_sequence_nll(opt_token_ids, model=model)
            if self.use_log_ppl
            else self.compute_perplexity(opt_token_ids, model=model)
        )

        return target_loss + lambda_weight * fluency_penalty


class CompositeObjective(nn.Module):
    """Composite objective evaluator combining target loss and regularizers."""

    def __init__(
        self,
        target_loss_fn: Callable[..., torch.Tensor],
        regularizer: Optional[PerplexityRegularizer] = None,
        lambda_weight: float = 0.0,
    ):
        super().__init__()
        self.target_loss_fn = target_loss_fn
        self.regularizer = regularizer
        self.lambda_weight = lambda_weight

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        opt_token_ids: Optional[torch.Tensor] = None,
        model: Optional[nn.Module] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Calculate composite loss and return individual component metrics."""
        primary_loss = self.target_loss_fn(logits, labels)

        reg_val = torch.tensor(0.0, device=logits.device)
        if self.regularizer is not None and self.lambda_weight > 0.0 and opt_token_ids is not None:
            reg_val = self.regularizer.compute_sequence_nll(opt_token_ids, model=model).mean()

        total_loss = primary_loss + self.lambda_weight * reg_val

        metrics = {
            "primary_loss": float(primary_loss.item()) if primary_loss.numel() == 1 else float(primary_loss.mean().item()),
            "reg_loss": float(reg_val.item()),
            "total_loss": float(total_loss.item()) if total_loss.numel() == 1 else float(total_loss.mean().item()),
        }
        return total_loss, metrics
