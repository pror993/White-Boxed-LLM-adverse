"""Discrete perturbation sensitivity profiling and loss curvature analysis.

Quantifies the empirical stability, sharpness, and output degradation of optimized
discrete token sequences under stochastic token substitutions, swaps, and insertions.
"""

from __future__ import annotations

import enum
import logging
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from src.models.causal_lm import CausalLMModelWrapper, TokenSlices
from src.optimizers.coordinate_gradient import evaluate_candidates_batched
from src.utils.device import get_best_device

logger = logging.getLogger(__name__)


class PerturbationType(str, enum.Enum):
    """Categorical discrete perturbation strategies."""

    TOKEN_SUBSTITUTION = "token_substitution"
    ADJACENT_SWAP = "adjacent_swap"
    RANDOM_INSERTION = "random_insertion"


@dataclass
class NoiseLevelEvaluation:
    """Metrics evaluated at a specific perturbation noise probability q."""

    noise_rate: float
    num_samples: int
    mean_loss: float
    std_loss: float
    min_loss: float
    max_loss: float
    loss_delta_from_baseline: float
    target_match_rate: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class SensitivityProfileResult:
    """Master sensitivity profiling metrics across noise rates."""

    baseline_loss: float
    perturbation_type: str
    noise_evaluations: List[NoiseLevelEvaluation] = field(default_factory=list)
    sharpness_index: float = 0.0
    empirical_curvature: float = 0.0
    sharp_minima_confirmed: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "baseline_loss": round(self.baseline_loss, 4),
            "perturbation_type": self.perturbation_type,
            "sharpness_index": round(self.sharpness_index, 4),
            "empirical_curvature": round(self.empirical_curvature, 4),
            "sharp_minima_confirmed": self.sharp_minima_confirmed,
            "evaluations": [e.to_dict() for e in self.noise_evaluations],
        }


def perturb_tokens(
    token_ids: torch.Tensor,
    opt_slice: slice,
    noise_rate: float,
    vocab_size: int,
    strategy: PerturbationType = PerturbationType.TOKEN_SUBSTITUTION,
    filter_mask: Optional[torch.Tensor] = None,
    rng: Optional[random.Random] = None,
) -> torch.Tensor:
    """Apply discrete perturbations strictly within the optimizable token slice.

    Args:
        token_ids: 1D or 2D tensor of token IDs `[1, seq_len]`.
        opt_slice: Coordinates allowed to mutate.
        noise_rate: Mutation probability q in [0, 1].
        vocab_size: Total vocabulary size.
        strategy: Perturbation mechanism.
        filter_mask: Boolean tensor of tokens to avoid sampling.
        rng: Optional Python random instance.

    Returns:
        Perturbed copy of the input token sequence.
    """
    rand = rng or random.Random()
    perturbed = token_ids.clone()

    m_start = opt_slice.start
    m_stop = opt_slice.stop
    slice_len = m_stop - m_start

    if slice_len <= 0 or noise_rate <= 0.0:
        return perturbed

    if strategy == PerturbationType.TOKEN_SUBSTITUTION:
        for offset in range(slice_len):
            if rand.random() < noise_rate:
                pos = m_start + offset
                # Sample random token from vocab that is not in filter_mask
                new_tok = rand.randint(10, vocab_size - 1)
                if filter_mask is not None and filter_mask[new_tok]:
                    new_tok = (new_tok + 1) % vocab_size
                perturbed[0, pos] = new_tok

    elif strategy == PerturbationType.ADJACENT_SWAP:
        for offset in range(slice_len - 1):
            if rand.random() < noise_rate:
                pos = m_start + offset
                # Swap adjacent tokens
                temp = perturbed[0, pos].item()
                perturbed[0, pos] = perturbed[0, pos + 1]
                perturbed[0, pos + 1] = temp

    return perturbed


class DiscreteSensitivityProfiler:
    """Profiles the loss landscape geometry around optimized discrete token sequences."""

    def __init__(
        self,
        model_wrapper: CausalLMModelWrapper,
        device: Optional[Union[torch.device, str]] = None,
    ):
        self.wrapper = model_wrapper
        self.model = model_wrapper.model
        self.tokenizer = model_wrapper.tokenizer
        self.device = torch.device(device) if device is not None else get_best_device()

    def profile_sequence(
        self,
        input_ids: torch.Tensor,
        slices: TokenSlices,
        target_text: str,
        noise_levels: Optional[List[float]] = None,
        samples_per_noise_level: int = 16,
        strategy: PerturbationType = PerturbationType.TOKEN_SUBSTITUTION,
        micro_batch_size: int = 16,
    ) -> SensitivityProfileResult:
        """Run Monte Carlo discrete noise perturbation to assess loss curvature.

        Args:
            input_ids: Optimized token sequence `[1, seq_len]`.
            slices: Aligned token slices.
            target_text: Ground-truth target completion to test generation match.
            noise_levels: Sequence of perturbation rates q (e.g. [0.05, 0.10, 0.15, 0.20]).
            samples_per_noise_level: Number of random draws per noise level.
            strategy: Perturbation mechanism.
            micro_batch_size: Forward pass chunk size.

        Returns:
            SensitivityProfileResult capturing empirical curvature and loss variances.
        """
        levels = noise_levels or [0.05, 0.10, 0.15, 0.20]
        vocab_sz = self.wrapper.vocab_size
        filter_mask = self.wrapper.build_non_ascii_mask(self.device)

        # 1. Measure baseline loss
        with torch.no_grad():
            baseline_batch = evaluate_candidates_batched(
                model=self.model,
                candidate_ids=input_ids.to(self.device),
                loss_slice=slices.loss_slice,
                target_slice=slices.target_slice,
                micro_batch_size=1,
                device=self.device,
            )
            baseline_loss = float(baseline_batch.best_loss)

        profile_evals: List[NoiseLevelEvaluation] = []
        curvatures: List[float] = []

        # 2. Monte Carlo evaluations per noise rate
        for q in levels:
            perturbed_candidates_list: List[torch.Tensor] = []
            for _ in range(samples_per_noise_level):
                cand = perturb_tokens(
                    token_ids=input_ids,
                    opt_slice=slices.opt_slice,
                    noise_rate=q,
                    vocab_size=vocab_sz,
                    strategy=strategy,
                    filter_mask=filter_mask,
                )
                perturbed_candidates_list.append(cand[0])

            candidate_batch = torch.stack(perturbed_candidates_list, dim=0).to(self.device)

            # Evaluate forward losses
            eval_batch = evaluate_candidates_batched(
                model=self.model,
                candidate_ids=candidate_batch,
                loss_slice=slices.loss_slice,
                target_slice=slices.target_slice,
                micro_batch_size=micro_batch_size,
                device=self.device,
            )

            losses = eval_batch.exact_losses.numpy() if eval_batch.exact_losses is not None else np.array([0.0])
            mean_l = float(np.mean(losses))
            std_l = float(np.std(losses))
            min_l = float(np.min(losses))
            max_l = float(np.max(losses))
            delta_l = mean_l - baseline_loss

            # Check generation match degradation on first candidate sample
            with torch.no_grad():
                gen_text = self.wrapper.generate(
                    input_ids=candidate_batch[0:1, : slices.opt_slice.stop],
                    max_new_tokens=len(self.tokenizer(target_text).input_ids) + 5,
                )
                target_match = float(target_text.strip().lower() in gen_text.lower())

            eval_entry = NoiseLevelEvaluation(
                noise_rate=q,
                num_samples=samples_per_noise_level,
                mean_loss=mean_l,
                std_loss=std_l,
                min_loss=min_l,
                max_loss=max_l,
                loss_delta_from_baseline=delta_l,
                target_match_rate=target_match,
            )
            profile_evals.append(eval_entry)

            if q > 0:
                curvatures.append(delta_l / q)

        # Average slope delta_L / q gives empirical sharpness index
        sharpness_idx = float(np.mean(curvatures)) if curvatures else 0.0
        # Second-order empirical curvature (rate of slope increase)
        emp_curvature = float(np.var(curvatures)) if len(curvatures) > 1 else 0.0
        sharp_minima = sharpness_idx > 2.0  # Sharp minima threshold

        return SensitivityProfileResult(
            baseline_loss=baseline_loss,
            perturbation_type=strategy.value,
            noise_evaluations=profile_evals,
            sharpness_index=sharpness_idx,
            empirical_curvature=emp_curvature,
            sharp_minima_confirmed=sharp_minima,
        )
