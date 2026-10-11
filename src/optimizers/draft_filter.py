"""Multi-fidelity candidate screening and draft model filtering for GCG discrete optimization.

Accelerates coordinate gradient search by filtering large candidate pools (e.g. B=512)
using a lightweight draft model (e.g. 0.5B parameters or smaller) prior to evaluation
on the target language model.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.models.causal_lm import CausalLMModelWrapper
from src.utils.device import clear_device_cache, get_best_device

logger = logging.getLogger(__name__)


@dataclass
class DraftFilterConfig:
    """Configuration for two-stage multi-fidelity candidate screening."""

    enabled: bool = True
    filtering_ratio: int = 8  # Retain top B // r candidates (e.g., 512 // 8 = 64)
    draft_micro_batch_size: int = 64
    min_retained_candidates: int = 8
    always_retain_identity: bool = True  # Ensure candidate 0 is always evaluated on target


@dataclass
class DraftScreeningResult:
    """Telemetry and tensor output from draft model candidate screening."""

    filtered_candidate_ids: torch.Tensor  # [K, seq_len]
    selected_indices: torch.Tensor  # [K] indices from original batch
    draft_losses: torch.Tensor  # [K] losses on the draft model
    original_batch_size: int
    filtered_batch_size: int
    screening_time_seconds: float
    effective_filtering_ratio: float


class DraftModelFilter:
    """Hierarchical candidate screener using a lightweight draft causal model."""

    def __init__(
        self,
        draft_model: Union[PreTrainedModel, CausalLMModelWrapper, nn.Module],
        draft_tokenizer: Optional[PreTrainedTokenizerBase] = None,
        config: Optional[DraftFilterConfig] = None,
        device: Optional[Union[torch.device, str]] = None,
    ):
        self.config = config or DraftFilterConfig()
        self.device = torch.device(device) if device is not None else get_best_device()

        if isinstance(draft_model, CausalLMModelWrapper):
            self.model = draft_model.model
            self.tokenizer = draft_model.tokenizer
        else:
            self.model = draft_model
            self.tokenizer = draft_tokenizer

        self.model.to(self.device)
        self.model.eval()

    def filter_candidates(
        self,
        candidate_ids: torch.Tensor,
        loss_slice: slice,
        target_slice: slice,
        filtering_ratio: Optional[int] = None,
    ) -> DraftScreeningResult:
        """Screen and rank candidate sequences, retaining only top-performing fraction.

        Args:
            candidate_ids: Input candidate sequences `[B, seq_len]`.
            loss_slice: Prediction logits slice for target sequence.
            target_slice: Ground truth target token slice.
            filtering_ratio: Optional override for candidate retention ratio $r$.

        Returns:
            DraftScreeningResult containing filtered candidates and telemetry.
        """
        start_time = time.time()
        b_total = candidate_ids.size(0)
        r = filtering_ratio or self.config.filtering_ratio
        k_target = max(self.config.min_retained_candidates, b_total // r)
        k_target = min(k_target, b_total)

        # Micro-batched evaluation across draft model
        micro_bs = self.config.draft_micro_batch_size
        candidate_losses: List[torch.Tensor] = []
        loss_fct = nn.CrossEntropyLoss(reduction="none")

        with torch.no_grad():
            for start_idx in range(0, b_total, micro_bs):
                end_idx = min(start_idx + micro_bs, b_total)
                chunk = candidate_ids[start_idx:end_idx].to(self.device)

                outputs = self.model(chunk)
                logits = outputs.logits

                chunk_target_logits = logits[:, loss_slice, :].contiguous()
                chunk_target_labels = chunk[:, target_slice].contiguous()

                chunk_b, target_len, vocab_sz = chunk_target_logits.shape
                flat_loss = loss_fct(
                    chunk_target_logits.view(-1, vocab_sz),
                    chunk_target_labels.view(-1),
                )
                per_seq_loss = flat_loss.view(chunk_b, target_len).mean(dim=1)
                candidate_losses.append(per_seq_loss.cpu())

                del chunk, outputs, logits, chunk_target_logits, chunk_target_labels

        all_draft_losses = torch.cat(candidate_losses, dim=0)  # [B]

        # Candidate 0 is the baseline identity (unmodified sequence).
        # We always guarantee candidate 0 is retained so greedy search stays monotonic.
        if self.config.always_retain_identity:
            # Sort candidates 1..B
            remaining_losses = all_draft_losses[1:]
            num_to_pick = k_target - 1
            if num_to_pick > 0:
                _, sorted_sub_indices = torch.topk(remaining_losses, num_to_pick, largest=False)
                # Map back to original indices (add 1 because of slice)
                retained_sub_indices = (sorted_sub_indices + 1).tolist()
                selected_indices_list = [0] + retained_sub_indices
            else:
                selected_indices_list = [0]
        else:
            _, sorted_indices = torch.topk(all_draft_losses, k_target, largest=False)
            selected_indices_list = sorted_indices.tolist()

        selected_indices = torch.tensor(selected_indices_list, dtype=torch.long)
        filtered_candidates = candidate_ids[selected_indices]
        filtered_draft_losses = all_draft_losses[selected_indices]

        elapsed = time.time() - start_time

        clear_device_cache(self.device)

        return DraftScreeningResult(
            filtered_candidate_ids=filtered_candidates,
            selected_indices=selected_indices,
            draft_losses=filtered_draft_losses,
            original_batch_size=b_total,
            filtered_batch_size=len(selected_indices_list),
            screening_time_seconds=elapsed,
            effective_filtering_ratio=b_total / max(1, len(selected_indices_list)),
        )
