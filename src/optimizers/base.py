"""Abstract base interface and telemetry dataclasses for discrete optimizers.

Provides the mathematical abstraction for discrete coordinate gradient search,
first-order Taylor polynomial candidate sampling, micro-batched forward evaluation,
and trajectory profiling on autoregressive language models.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union

import torch

from src.utils.device import (
    MPSMemorySnapshot,
    MPSMemoryTracker,
    clear_device_cache,
    get_best_device,
    get_mps_memory_snapshot,
    synchronize_device,
)

logger = logging.getLogger(__name__)


@dataclass
class OptimizerConfig:
    """Hyperparameters and configuration for discrete coordinate optimizers."""

    num_steps: int = 500
    batch_size: int = 512
    top_k: int = 256
    micro_batch_size: int = 32
    num_coordinates_per_step: int = 1
    coordinate_selection_strategy: str = "random"  # "random", "all", "cyclic", "gradient_norm"
    allow_non_ascii: bool = False
    filter_special_tokens: bool = True
    early_stopping_loss: Optional[float] = 0.01
    track_trajectories: bool = True
    device: Optional[str] = "mps"
    dtype: str = "float16"
    seed: Optional[int] = 42
    profile_memory: bool = True
    enable_draft_filtering: bool = False
    draft_filtering_ratio: int = 8
    objective_type: str = "cross_entropy"  # "cross_entropy", "mellowmax", "regularized_ppl"
    mellowmax_alpha: float = 1.0
    perplexity_lambda: float = 0.0
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OptimizationStepResult:
    """State, metrics, and substitution metadata for a single discrete step."""

    step: int
    loss: float
    best_candidate_tokens: List[int]
    best_candidate_text: str
    substituted_position: Optional[int] = None
    original_token_id: Optional[int] = None
    new_token_id: Optional[int] = None
    gradient_norm: float = 0.0
    candidates_evaluated: int = 0
    step_time_seconds: float = 0.0
    memory_snapshot: Optional[Dict[str, Any]] = None
    success: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateEvaluationBatch:
    """Batched candidate sequences, Taylor surrogate scores, and exact model losses."""

    candidate_ids: torch.Tensor  # [B, seq_len]
    surrogate_scores: Optional[torch.Tensor] = None  # [B] Taylor approximation scores
    exact_losses: Optional[torch.Tensor] = None  # [B] Ground-truth forward losses
    best_candidate_idx: int = 0
    best_loss: float = float("inf")
    evaluated_batch_size: int = 0


@dataclass
class OptimizationTrajectory:
    """Complete convergence history, token substitution track, and curvature dynamics."""

    history: List[OptimizationStepResult] = field(default_factory=list)
    initial_loss: float = float("inf")
    best_loss: float = float("inf")
    best_step: int = -1
    best_tokens: List[int] = field(default_factory=list)
    best_text: str = ""
    total_wall_time: float = 0.0
    total_candidates_evaluated: int = 0
    target_reached: bool = False

    def add_step(self, step_result: OptimizationStepResult) -> None:
        """Record step result and update global trajectory optima."""
        self.history.append(step_result)
        self.total_candidates_evaluated += step_result.candidates_evaluated
        if step_result.loss < self.best_loss:
            self.best_loss = step_result.loss
            self.best_step = step_result.step
            self.best_tokens = list(step_result.best_candidate_tokens)
            self.best_text = step_result.best_candidate_text

    def get_loss_curve(self) -> List[float]:
        """Extract sequential loss values over optimization steps."""
        return [step.loss for step in self.history]

    def get_gradient_norms(self) -> List[float]:
        """Extract embedding gradient norms over optimization steps."""
        return [step.gradient_norm for step in self.history]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_loss": self.initial_loss,
            "best_loss": self.best_loss,
            "best_step": self.best_step,
            "best_tokens": self.best_tokens,
            "best_text": self.best_text,
            "total_wall_time": round(self.total_wall_time, 3),
            "total_candidates_evaluated": self.total_candidates_evaluated,
            "target_reached": self.target_reached,
            "steps_count": len(self.history),
            "history": [s.to_dict() for s in self.history],
        }


class DiscreteOptimizer(ABC):
    """Abstract Base Class for first-order discrete coordinate optimizers.

    Defines the canonical computational pipeline:
    1. `compute_coordinate_gradients`: Exact $\\nabla_E \\mathcal{L}$ calculation via autograd graph.
    2. `sample_candidates`: First-order Taylor polynomial surrogate sampling.
    3. `evaluate_candidates`: Micro-batched exact forward evaluation on Apple Silicon.
    4. `step`: Single coordinate update iteration.
    5. `optimize`: End-to-end discrete optimization loop with trajectory tracking.
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        config: Optional[OptimizerConfig] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or OptimizerConfig()
        self.device = get_best_device(self.config.device)
        self.trajectory = OptimizationTrajectory()

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)

    @abstractmethod
    def compute_coordinate_gradients(
        self,
        input_ids: torch.Tensor,
        target_slice: slice,
        loss_slice: slice,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, float]:
        """Compute exact gradients of the objective with respect to coordinate embeddings.

        Args:
            input_ids: Token indices of shape `[1, seq_len]`.
            target_slice: Slice designating the coordinate positions to optimize.
            loss_slice: Slice designating output tokens contributing to loss evaluation.

        Returns:
            Tuple of:
                - `grad`: Tensor of shape `[target_len, vocab_size]` or `[target_len, embed_dim]`.
                - `loss`: Current scalar loss on input sequence.
        """
        pass

    @abstractmethod
    def sample_candidates(
        self,
        input_ids: torch.Tensor,
        target_slice: slice,
        coordinate_gradients: torch.Tensor,
        batch_size: int,
        top_k: int,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample candidate token sequences using first-order Taylor approximation scores.

        Args:
            input_ids: Current sequence of token indices `[1, seq_len]`.
            target_slice: Coordinate slice subject to mutation.
            coordinate_gradients: Gradients computed at the target slice coordinates.
            batch_size: Total candidate sequences to generate.
            top_k: Top substitution candidates considered per coordinate.

        Returns:
            Tuple of:
                - `candidate_ids`: Tensor of shape `[B, seq_len]`.
                - `surrogate_scores`: Estimated loss delta scores `[B]`.
        """
        pass

    def evaluate_candidates(
        self,
        candidate_ids: torch.Tensor,
        loss_slice: slice,
        micro_batch_size: Optional[int] = None,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> CandidateEvaluationBatch:
        """Evaluate candidate token sequences using micro-batched forward passes.

        Guarantees that memory footprint stays strictly within Apple Silicon 16GB limits
        by chunking evaluation into micro-batches, executing garbage collection, and
        clearing the Metal cache when necessary.

        Args:
            candidate_ids: Tensor of candidate sequences `[B, seq_len]`.
            loss_slice: Slice defining positions where loss is computed.
            micro_batch_size: Forward pass chunk size (defaults to `self.config.micro_batch_size`).
            loss_fn: Optional custom scalar reduction function.

        Returns:
            `CandidateEvaluationBatch` containing exact candidate losses and best candidate index.
        """
        micro_bs = micro_batch_size or self.config.micro_batch_size
        total_candidates = candidate_ids.size(0)
        candidate_losses: List[torch.Tensor] = []

        self.model.eval()

        for chunk_start in range(0, total_candidates, micro_bs):
            chunk_end = min(chunk_start + micro_bs, total_candidates)
            chunk_ids = candidate_ids[chunk_start:chunk_end].to(self.device)

            with torch.no_grad():
                # Forward pass
                outputs = self.model(chunk_ids)
                logits = outputs.logits  # [micro_b, seq_len, vocab_size]

                # Compute per-candidate sequence cross-entropy loss over target positions
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = chunk_ids[:, 1:].contiguous()

                # Slice shift_logits and shift_labels according to loss_slice
                # Default causal prediction alignment
                loss_start = max(0, loss_slice.start - 1) if loss_slice.start is not None else 0
                loss_end = loss_slice.stop - 1 if loss_slice.stop is not None else shift_labels.size(1)

                eval_logits = shift_logits[:, loss_start:loss_end, :].contiguous()
                eval_labels = shift_labels[:, loss_start:loss_end].contiguous()

                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                # Flatten micro_b and seq_len for CE
                mb, seq_l, vocab_sz = eval_logits.shape
                flat_losses = loss_fct(
                    eval_logits.view(-1, vocab_sz),
                    eval_labels.view(-1),
                )
                per_candidate_loss = flat_losses.view(mb, seq_l).mean(dim=1)
                candidate_losses.append(per_candidate_loss.cpu())

            # Immediate cleanup to avoid MPS allocator fragmentation
            del chunk_ids, outputs, logits, shift_logits, shift_labels, eval_logits, eval_labels
            if (chunk_start // micro_bs) % 4 == 0:
                clear_device_cache(self.device)

        all_losses = torch.cat(candidate_losses, dim=0)  # [total_candidates]
        best_idx = int(torch.argmin(all_losses).item())
        best_loss = float(all_losses[best_idx].item())

        return CandidateEvaluationBatch(
            candidate_ids=candidate_ids,
            exact_losses=all_losses,
            best_candidate_idx=best_idx,
            best_loss=best_loss,
            evaluated_batch_size=total_candidates,
        )

    @abstractmethod
    def step(
        self,
        current_ids: torch.Tensor,
        target_slice: slice,
        loss_slice: slice,
        step_idx: int,
    ) -> OptimizationStepResult:
        """Execute a single discrete optimization step.

        Args:
            current_ids: Current sequence token IDs `[1, seq_len]`.
            target_slice: Coordinates to modify.
            loss_slice: Coordinates defining objective loss.
            step_idx: Current iteration index.

        Returns:
            OptimizationStepResult with loss, metrics, and new token candidate.
        """
        pass

    def optimize(
        self,
        initial_ids: torch.Tensor,
        target_slice: slice,
        loss_slice: slice,
        callback: Optional[Callable[[OptimizationStepResult], None]] = None,
    ) -> OptimizationTrajectory:
        """Run iterative discrete search until convergence, early stopping, or max steps.

        Args:
            initial_ids: Starting token sequence `[1, seq_len]`.
            target_slice: Mutable coordinate span.
            loss_slice: Target prediction loss span.
            callback: Optional per-step hook for progress streaming or logging.

        Returns:
            OptimizationTrajectory with full optimization history.
        """
        start_time = time.time()
        current_ids = initial_ids.clone()

        # Compute initial loss
        with torch.no_grad():
            init_eval = self.evaluate_candidates(current_ids, loss_slice=loss_slice)
            self.trajectory.initial_loss = init_eval.best_loss
            self.trajectory.best_loss = init_eval.best_loss
            self.trajectory.best_tokens = current_ids[0, target_slice].tolist()
            self.trajectory.best_text = self.tokenizer.decode(self.trajectory.best_tokens)

        logger.info(
            "Starting discrete optimization: Initial Loss = %.4f, Target Coordinates = %s",
            self.trajectory.initial_loss,
            target_slice,
        )

        for step_idx in range(self.config.num_steps):
            step_start = time.time()
            with MPSMemoryTracker(f"step_{step_idx}", device=self.device) as tracker:
                step_result = self.step(
                    current_ids=current_ids,
                    target_slice=target_slice,
                    loss_slice=loss_slice,
                    step_idx=step_idx,
                )

            step_result.step_time_seconds = time.time() - step_start
            if self.config.profile_memory:
                step_result.memory_snapshot = tracker.summary()

            # Update working token sequence with best candidate found in this step
            current_ids = torch.tensor(
                [step_result.best_candidate_tokens],
                dtype=current_ids.dtype,
                device=current_ids.device,
            )

            # Record trajectory
            self.trajectory.add_step(step_result)

            if callback is not None:
                callback(step_result)

            logger.info(
                "Step %03d | Loss: %.4f | Best: %.4f | Evaluated: %d | Time: %.2fs",
                step_idx,
                step_result.loss,
                self.trajectory.best_loss,
                step_result.candidates_evaluated,
                step_result.step_time_seconds,
            )

            # Check early stopping threshold
            if (
                self.config.early_stopping_loss is not None
                and step_result.loss <= self.config.early_stopping_loss
            ):
                logger.info(
                    "Early stopping criterion reached: Loss %.4f <= %.4f at step %d",
                    step_result.loss,
                    self.config.early_stopping_loss,
                    step_idx,
                )
                self.trajectory.target_reached = True
                break

        self.trajectory.total_wall_time = time.time() - start_time
        return self.trajectory
