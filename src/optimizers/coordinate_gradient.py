"""Greedy Coordinate Gradient (GCG) discrete optimizer for causal language models.

Implements first-order Taylor series approximations over discrete token substitutions,
embedding gradient projection, candidate sampling, and micro-batched forward evaluation
optimized for Apple Silicon Metal (MPS) architectures.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table

from src.models.causal_lm import CausalLMModelWrapper, TokenSlices
from src.optimizers.base import (
    CandidateEvaluationBatch,
    DiscreteOptimizer,
    OptimizationStepResult,
    OptimizationTrajectory,
    OptimizerConfig,
)
from src.utils.device import (
    MPSMemoryTracker,
    clear_device_cache,
    get_best_device,
    safe_mps_context,
    synchronize_device,
)

logger = logging.getLogger(__name__)
console = Console()


def extract_embedding_matrix(model: nn.Module) -> torch.Tensor:
    """Retrieve the token embedding weight matrix from a causal LM."""
    if hasattr(model, "get_input_embeddings"):
        layer = model.get_input_embeddings()
        if hasattr(layer, "weight"):
            return layer.weight

    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and "embed" in name.lower():
            return module.weight

    raise AttributeError(f"Could not locate embedding weight matrix in model {type(model)}")


def compute_token_gradients(
    model: nn.Module,
    input_ids: torch.Tensor,
    opt_slice: slice,
    target_slice: slice,
    loss_slice: slice,
    embedding_matrix: Optional[torch.Tensor] = None,
    filter_mask: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, float]:
    """Calculate exact gradients of target loss with respect to discrete token coordinates.

    Args:
        model: Hugging Face CausalLM module.
        input_ids: Full sequence token tensor `[1, seq_len]`.
        opt_slice: Coordinates of mutable tokens `slice(opt_start, opt_end)`.
        target_slice: Ground truth target token coordinates `slice(target_start, target_end)`.
        loss_slice: Corresponding causal prediction logits slice `slice(target_start - 1, target_end - 1)`.
        embedding_matrix: Cached embedding weights `[vocab_size, hidden_dim]`.
        filter_mask: Boolean tensor `[vocab_size]` where True indicates tokens to suppress.
        device: Active PyTorch compute device.

    Returns:
        Tuple of:
            - `token_grads`: Projected first-order Taylor gradients `[m, vocab_size]`.
            - `current_loss`: Current scalar cross-entropy loss on the sequence.
    """
    dev = device or input_ids.device
    model.eval()

    if embedding_matrix is None:
        embedding_matrix = extract_embedding_matrix(model)

    # 1. Obtain input embeddings with gradient tracking
    # Use one_hot projection or detached embedding lookup with requires_grad=True
    with torch.no_grad():
        raw_embeds = model.get_input_embeddings()(input_ids.to(dev))

    inputs_embeds = raw_embeds.detach().clone().requires_grad_(True)

    # 2. Forward pass through causal transformer
    outputs = model(inputs_embeds=inputs_embeds)
    logits = outputs.logits  # [1, seq_len, vocab_size]

    # 3. Compute cross-entropy loss over target tokens
    # Causal shift: logits at [target_start - 1 : target_end - 1] predict target_ids at [target_start : target_end]
    target_logits = logits[0, loss_slice, :].contiguous()  # [target_len, vocab_size]
    target_labels = input_ids[0, target_slice].to(dev).contiguous()  # [target_len]

    loss_fct = nn.CrossEntropyLoss(reduction="mean")
    loss = loss_fct(target_logits, target_labels)
    current_loss_val = float(loss.item())

    # 4. Backward pass to extract embedding gradients
    loss.backward()

    # inputs_embeds.grad has shape [1, seq_len, hidden_dim]
    assert inputs_embeds.grad is not None, "Autograd failed to propagate gradients to inputs_embeds"
    opt_embed_grad = inputs_embeds.grad[0, opt_slice, :]  # [m, hidden_dim]

    # 5. First-order Taylor projection: dot product with embedding matrix
    # Taylor linear term: delta_L ≈ <grad, e(v) - e(x_i)> = <grad, e(v)> - <grad, e(x_i)>
    # To minimize loss (delta_L < 0), we seek tokens v with most negative <grad, e(v)>.
    # token_grads[i, v] = <grad_i, e(v)>
    # Shape: [m, hidden_dim] @ [hidden_dim, vocab_size] -> [m, vocab_size]
    token_grads = torch.matmul(opt_embed_grad, embedding_matrix.T)

    # 6. Apply filtering (e.g. non-ASCII or special tokens)
    if filter_mask is not None:
        # Assign high positive gradient so they will never be selected for minimization
        token_grads[:, filter_mask.to(token_grads.device)] = float("inf")

    # Clean intermediate autograd buffers
    del inputs_embeds, outputs, logits, target_logits
    return token_grads, current_loss_val


def sample_candidate_substitutions(
    current_ids: torch.Tensor,
    opt_slice: slice,
    token_gradients: torch.Tensor,
    batch_size: int = 512,
    top_k: int = 256,
    num_coordinates_per_step: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample candidate sequences from the top-k first-order Taylor replacement pool.

    Args:
        current_ids: Current sequence token IDs `[1, seq_len]`.
        opt_slice: Slice defining mutable tokens `[opt_start, opt_end]`.
        token_gradients: Taylor gradient scores `[m, vocab_size]` (lower is better).
        batch_size: Number of candidate sequences to sample.
        top_k: Top candidate substitutions considered per coordinate.
        num_coordinates_per_step: Number of coordinates mutated per sequence (default 1).

    Returns:
        Tuple of:
            - `candidate_ids`: Sampled candidate token IDs `[B, seq_len]`.
            - `sampled_scores`: First-order surrogate score deltas `[B]`.
    """
    device = current_ids.device
    seq_len = current_ids.size(1)
    m = opt_slice.stop - opt_slice.start
    vocab_size = token_gradients.size(1)

    # Top-k tokens with the lowest (most negative) gradient values
    actual_top_k = min(top_k, vocab_size)
    _, topk_indices = torch.topk(-token_gradients, actual_top_k, dim=1)  # [m, top_k]

    # Pre-allocate candidate tensor
    candidate_ids = current_ids.repeat(batch_size, 1)

    # Candidate 0 is always the identity (unmodified current sequence) to guarantee monotonicity
    # Sample coordinates to mutate for remaining candidates
    sample_size = batch_size - 1
    sampled_coords = torch.randint(0, m, (sample_size, num_coordinates_per_step), device=device)
    sampled_tok_ranks = torch.randint(0, actual_top_k, (sample_size, num_coordinates_per_step), device=device)

    # Extract chosen replacement token IDs
    # topk_indices: [m, actual_top_k]
    # For each sample s and coordinate c: coord = sampled_coords[s, c], rank = sampled_tok_ranks[s, c]
    for step_c in range(num_coordinates_per_step):
        coords = sampled_coords[:, step_c]  # [sample_size]
        ranks = sampled_tok_ranks[:, step_c]  # [sample_size]
        new_token_ids = topk_indices[coords, ranks]  # [sample_size]

        # Map relative coordinate to absolute sequence position
        abs_positions = coords + opt_slice.start
        candidate_ids[1:, abs_positions] = new_token_ids

    # Compute surrogate first-order scores
    surrogate_scores = torch.zeros(batch_size, device=device)
    return candidate_ids, surrogate_scores


def evaluate_candidates_batched(
    model: nn.Module,
    candidate_ids: torch.Tensor,
    loss_slice: slice,
    target_slice: slice,
    micro_batch_size: int = 64,
    device: Optional[torch.device] = None,
) -> CandidateEvaluationBatch:
    """Evaluate candidate sequences in micro-batches to strictly observe memory limits.

    Args:
        model: CausalLM model.
        candidate_ids: Tensor of candidate sequences `[B, seq_len]`.
        loss_slice: Logit slice corresponding to target prediction.
        target_slice: Ground truth target token slice.
        micro_batch_size: Batch chunk size for each forward pass.
        device: Execution device (e.g. MPS).

    Returns:
        `CandidateEvaluationBatch` with exact candidate losses and minimum loss index.
    """
    dev = device or candidate_ids.device
    total_candidates = candidate_ids.size(0)
    candidate_losses: List[torch.Tensor] = []

    model.eval()
    loss_fct = nn.CrossEntropyLoss(reduction="none")

    with torch.no_grad():
        for start_idx in range(0, total_candidates, micro_batch_size):
            end_idx = min(start_idx + micro_batch_size, total_candidates)
            batch_chunk = candidate_ids[start_idx:end_idx].to(dev)

            outputs = model(batch_chunk)
            logits = outputs.logits  # [chunk_b, seq_len, vocab_size]

            # Slice target prediction logits and labels
            chunk_target_logits = logits[:, loss_slice, :].contiguous()  # [chunk_b, target_len, vocab_size]
            chunk_target_labels = batch_chunk[:, target_slice].contiguous()  # [chunk_b, target_len]

            chunk_b, target_len, vocab_sz = chunk_target_logits.shape
            flat_loss = loss_fct(
                chunk_target_logits.view(-1, vocab_sz),
                chunk_target_labels.view(-1),
            )
            per_seq_loss = flat_loss.view(chunk_b, target_len).mean(dim=1)
            candidate_losses.append(per_seq_loss.cpu())

            del batch_chunk, outputs, logits, chunk_target_logits, chunk_target_labels
            if (start_idx // micro_batch_size) % 4 == 0:
                clear_device_cache(dev)

    all_losses = torch.cat(candidate_losses, dim=0)  # [B]
    best_idx = int(torch.argmin(all_losses).item())
    best_loss_val = float(all_losses[best_idx].item())

    return CandidateEvaluationBatch(
        candidate_ids=candidate_ids,
        exact_losses=all_losses,
        best_candidate_idx=best_idx,
        best_loss=best_loss_val,
        evaluated_batch_size=total_candidates,
    )


class CoordinateGradientOptimizer(DiscreteOptimizer):
    """Greedy Coordinate Gradient (GCG) discrete optimizer implementation.

    Conducts iterative first-order Taylor approximation search over mutable token coordinates
    to minimize target generation loss on autoregressive causal language models.
    """

    def __init__(
        self,
        model_wrapper: CausalLMModelWrapper,
        config: Optional[OptimizerConfig] = None,
    ):
        super().__init__(
            model=model_wrapper.model,
            tokenizer=model_wrapper.tokenizer,
            config=config,
        )
        self.wrapper = model_wrapper
        self.embedding_matrix = self.wrapper.get_embedding_matrix()
        self.filter_mask = (
            self.wrapper.build_non_ascii_mask(device=self.device)
            if self.config.filter_special_tokens or not self.config.allow_non_ascii
            else None
        )

    def compute_coordinate_gradients(
        self,
        input_ids: torch.Tensor,
        target_slice: slice,
        loss_slice: slice,
        opt_slice: Optional[slice] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, float]:
        """Compute exact gradients projected onto discrete vocabulary."""
        assert opt_slice is not None, "opt_slice must be provided"
        return compute_token_gradients(
            model=self.model,
            input_ids=input_ids,
            opt_slice=opt_slice,
            target_slice=target_slice,
            loss_slice=loss_slice,
            embedding_matrix=self.embedding_matrix,
            filter_mask=self.filter_mask,
            device=self.device,
        )

    def sample_candidates(
        self,
        input_ids: torch.Tensor,
        target_slice: slice,
        coordinate_gradients: torch.Tensor,
        batch_size: int,
        top_k: int,
        opt_slice: Optional[slice] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample candidate sequences from top-k Taylor replacement candidates."""
        assert opt_slice is not None, "opt_slice must be provided"
        return sample_candidate_substitutions(
            current_ids=input_ids,
            opt_slice=opt_slice,
            token_gradients=coordinate_gradients,
            batch_size=batch_size,
            top_k=top_k,
            num_coordinates_per_step=self.config.num_coordinates_per_step,
        )

    def step(
        self,
        current_ids: torch.Tensor,
        target_slice: slice,
        loss_slice: slice,
        step_idx: int,
        opt_slice: Optional[slice] = None,
    ) -> OptimizationStepResult:
        """Execute single discrete coordinate gradient update step."""
        assert opt_slice is not None, "opt_slice is required"

        step_start = time.time()

        # 1. Compute exact token coordinate gradients
        token_grads, current_loss = self.compute_coordinate_gradients(
            input_ids=current_ids,
            target_slice=target_slice,
            loss_slice=loss_slice,
            opt_slice=opt_slice,
        )
        grad_norm = float(torch.norm(token_grads[token_grads != float("inf")]).item())

        # 2. Sample candidate substitutions
        candidate_ids, _ = self.sample_candidates(
            input_ids=current_ids,
            target_slice=target_slice,
            coordinate_gradients=token_grads,
            batch_size=self.config.batch_size,
            top_k=self.config.top_k,
            opt_slice=opt_slice,
        )

        # 3. Micro-batched forward evaluation
        eval_batch = evaluate_candidates_batched(
            model=self.model,
            candidate_ids=candidate_ids,
            loss_slice=loss_slice,
            target_slice=target_slice,
            micro_batch_size=self.config.micro_batch_size,
            device=self.device,
        )

        best_cand_ids = eval_batch.candidate_ids[eval_batch.best_candidate_idx].unsqueeze(0)
        best_loss = eval_batch.best_loss
        best_opt_tokens = best_cand_ids[0, opt_slice].tolist()
        best_opt_text = self.tokenizer.decode(best_opt_tokens, skip_special_tokens=True)

        # Check which coordinate changed compared to current_ids
        diff = (best_cand_ids[0, opt_slice] != current_ids[0, opt_slice]).nonzero(as_tuple=False)
        sub_pos = int(diff[0].item()) + opt_slice.start if len(diff) > 0 else None
        orig_id = int(current_ids[0, sub_pos].item()) if sub_pos is not None else None
        new_id = int(best_cand_ids[0, sub_pos].item()) if sub_pos is not None else None

        step_elapsed = time.time() - step_start

        return OptimizationStepResult(
            step=step_idx,
            loss=best_loss,
            best_candidate_tokens=best_cand_ids[0].tolist(),
            best_candidate_text=best_opt_text,
            substituted_position=sub_pos,
            original_token_id=orig_id,
            new_token_id=new_id,
            gradient_norm=grad_norm,
            candidates_evaluated=eval_batch.evaluated_batch_size,
            step_time_seconds=step_elapsed,
            success=(
                best_loss <= self.config.early_stopping_loss
                if self.config.early_stopping_loss
                else False
            ),
        )

    def optimize_sequence(
        self,
        current_ids: torch.Tensor,
        slices: TokenSlices,
        log_interval: int = 1,
    ) -> OptimizationTrajectory:
        """Run complete coordinate gradient search loop on formatted token slices.

        Args:
            current_ids: Initial token sequence `[1, seq_len]`.
            slices: Pre-computed token slices (`opt_slice`, `target_slice`, `loss_slice`).
            log_interval: Step frequency for terminal display logging.

        Returns:
            OptimizationTrajectory tracking full convergence history.
        """
        start_time = time.time()
        working_ids = current_ids.clone().to(self.device)

        # Compute initial loss
        with torch.no_grad():
            initial_eval = evaluate_candidates_batched(
                model=self.model,
                candidate_ids=working_ids,
                loss_slice=slices.loss_slice,
                target_slice=slices.target_slice,
                micro_batch_size=1,
                device=self.device,
            )
            self.trajectory.initial_loss = initial_eval.best_loss
            self.trajectory.best_loss = initial_eval.best_loss
            self.trajectory.best_tokens = working_ids[0, slices.opt_slice].tolist()
            self.trajectory.best_text = self.tokenizer.decode(
                self.trajectory.best_tokens, skip_special_tokens=True
            )

        console.print(
            f"[bold cyan]Starting GCG Optimization:[/bold cyan] "
            f"Initial Loss = [bold yellow]{self.trajectory.initial_loss:.4f}[/bold yellow] | "
            f"Mutable Slice: [white]{slices.opt_slice}[/white] ({slices.opt_len} tokens)"
        )

        for step_idx in range(self.config.num_steps):
            with MPSMemoryTracker(f"step_{step_idx}", device=self.device) as mem_tracker:
                step_result = self.step(
                    current_ids=working_ids,
                    target_slice=slices.target_slice,
                    loss_slice=slices.loss_slice,
                    step_idx=step_idx,
                    opt_slice=slices.opt_slice,
                )

            if self.config.profile_memory:
                step_result.memory_snapshot = mem_tracker.summary()

            # Update working sequence with best candidate
            working_ids = torch.tensor(
                [step_result.best_candidate_tokens],
                dtype=torch.long,
                device=self.device,
            )

            self.trajectory.add_step(step_result)

            if step_idx % log_interval == 0 or step_result.success:
                table = Table(show_header=True, header_style="bold magenta", expand=False)
                table.add_column("Step", style="dim", width=6)
                table.add_column("Loss", justify="right", style="green", width=10)
                table.add_column("Best Loss", justify="right", style="bold yellow", width=10)
                table.add_column("Candidates", justify="right", width=10)
                table.add_column("Time (s)", justify="right", width=8)
                table.add_column("Optimized Suffix Preview", style="cyan")

                preview = (
                    step_result.best_candidate_text[:40] + "..."
                    if len(step_result.best_candidate_text) > 40
                    else step_result.best_candidate_text
                )
                table.add_row(
                    str(step_idx),
                    f"{step_result.loss:.4f}",
                    f"{self.trajectory.best_loss:.4f}",
                    str(step_result.candidates_evaluated),
                    f"{step_result.step_time_seconds:.2f}",
                    preview,
                )
                console.print(table)

            if (
                self.config.early_stopping_loss is not None
                and step_result.loss <= self.config.early_stopping_loss
            ):
                console.print(
                    f"[bold green]Early stopping achieved:[/bold green] "
                    f"Loss {step_result.loss:.4f} <= {self.config.early_stopping_loss} at step {step_idx}"
                )
                self.trajectory.target_reached = True
                break

        self.trajectory.total_wall_time = time.time() - start_time
        return self.trajectory
