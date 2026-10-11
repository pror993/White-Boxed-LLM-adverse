"""Unit tests for discrete coordinate gradient optimization and token slicing.

Validates exact token index alignment, continuous-to-discrete gradient projection,
top-k candidate sampling guarantees, and micro-batched candidate evaluation on MPS.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from src.models.causal_lm import CausalLMModelWrapper, PromptDecomposition, TokenSlices
from src.optimizers.base import OptimizerConfig
from src.optimizers.coordinate_gradient import (
    CoordinateGradientOptimizer,
    compute_token_gradients,
    evaluate_candidates_batched,
    extract_embedding_matrix,
    sample_candidate_substitutions,
)
from src.utils.device import get_best_device


@pytest.fixture(scope="module")
def shared_device():
    return get_best_device("mps")


@pytest.fixture(scope="module")
def small_lm_and_wrapper(shared_device):
    """Provide a consistent small causal model and wrapper for testing."""
    config = GPT2Config(
        vocab_size=500,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
        use_cache=False,
    )
    model = GPT2LMHeadModel(config).to(device=shared_device, dtype=torch.float32)
    model.eval()

    class SyntheticTokenizer:
        """Lightweight deterministic tokenizer for testing exact token alignment."""
        def __init__(self, vocab_size=500):
            self.vocab_size = vocab_size
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.bos_token_id = 2
            self.all_special_ids = [0, 1, 2]
            self.chat_template = None

        def __call__(self, text, add_special_tokens=True, return_tensors=None):
            # Deterministic mapping: ASCII code % (vocab_size - 10) + 10
            words = text.strip().split()
            toks = []
            if add_special_tokens:
                toks.append(self.bos_token_id)
            for w in words:
                toks.append(min(self.vocab_size - 1, max(10, sum(ord(c) for c in w) % self.vocab_size)))
            
            class Output:
                def __init__(self, ids):
                    self.input_ids = ids
            return Output(toks)

        def decode(self, token_ids, skip_special_tokens=False):
            return " ".join(f"tok_{t}" for t in token_ids if not (skip_special_tokens and t in self.all_special_ids))

    tokenizer = SyntheticTokenizer(vocab_size=500)
    wrapper = CausalLMModelWrapper(model=model, tokenizer=tokenizer, device=shared_device)
    return model, tokenizer, wrapper


class TestTokenSlices:
    """Validate token slice integrity and assertions."""

    def test_token_slices_validation_success(self):
        slices = TokenSlices(
            prefix_slice=slice(0, 5),
            opt_slice=slice(5, 10),
            target_slice=slice(12, 16),
            loss_slice=slice(11, 15),
            delim_slice=slice(10, 12),
        )
        assert slices.opt_len == 5
        assert slices.target_len == 4
        slices.validate(total_seq_len=20)

    def test_token_slices_validation_failure_on_overlap(self):
        slices = TokenSlices(
            prefix_slice=slice(0, 5),
            opt_slice=slice(5, 12),
            target_slice=slice(10, 15),  # Overlaps with opt_slice!
            loss_slice=slice(9, 14),
        )
        with pytest.raises(AssertionError):
            slices.validate(total_seq_len=20)

    def test_token_slices_validation_failure_on_loss_mismatch(self):
        slices = TokenSlices(
            prefix_slice=slice(0, 5),
            opt_slice=slice(5, 10),
            target_slice=slice(10, 15),
            loss_slice=slice(10, 15),  # Must be target.start - 1 : target.stop - 1!
        )
        with pytest.raises(AssertionError):
            slices.validate(total_seq_len=20)

    def test_build_slices_raw(self, small_lm_and_wrapper):
        _, _, wrapper = small_lm_and_wrapper
        decomp = wrapper.build_prompt_slices(
            context="Tell me a joke",
            optim_str="! ! ! !",
            target="Why did the chicken cross",
            use_chat_template=False,
        )
        assert isinstance(decomp, PromptDecomposition)
        assert decomp.slices.opt_len > 0
        assert decomp.slices.target_len > 0
        assert decomp.input_ids.size(1) >= decomp.slices.target_slice.stop


class TestCoordinateGradientComputation:
    """Validate exact gradient propagation and vocabulary projection on MPS."""

    def test_compute_token_gradients_shape_and_finite(self, small_lm_and_wrapper, shared_device):
        model, _, _ = small_lm_and_wrapper
        seq_len = 16
        input_ids = torch.randint(10, 400, (1, seq_len), device=shared_device)

        opt_slice = slice(4, 9)
        target_slice = slice(11, 15)
        loss_slice = slice(10, 14)

        token_grads, current_loss = compute_token_gradients(
            model=model,
            input_ids=input_ids,
            opt_slice=opt_slice,
            target_slice=target_slice,
            loss_slice=loss_slice,
            device=shared_device,
        )

        assert token_grads.device.type == shared_device.type
        assert token_grads.shape == (5, 500), f"Expected [5, 500], got {token_grads.shape}"
        assert isinstance(current_loss, float)
        assert current_loss > 0.0
        assert torch.isfinite(token_grads).all(), "Unmasked gradients must be finite."

    def test_compute_token_gradients_with_filter_mask(self, small_lm_and_wrapper, shared_device):
        model, _, _ = small_lm_and_wrapper
        seq_len = 16
        input_ids = torch.randint(10, 400, (1, seq_len), device=shared_device)

        opt_slice = slice(4, 8)
        target_slice = slice(10, 14)
        loss_slice = slice(9, 13)

        # Mask out tokens 0 to 15
        mask = torch.zeros(500, dtype=torch.bool, device=shared_device)
        mask[:15] = True

        token_grads, _ = compute_token_gradients(
            model=model,
            input_ids=input_ids,
            opt_slice=opt_slice,
            target_slice=target_slice,
            loss_slice=loss_slice,
            filter_mask=mask,
            device=shared_device,
        )

        # Filtered tokens must be set to infinity
        assert (token_grads[:, :15] == float("inf")).all()
        # Unfiltered tokens must be finite
        assert torch.isfinite(token_grads[:, 15:]).all()


class TestCandidateSampling:
    """Validate discrete candidate sampling and monotonicity guarantees."""

    def test_sample_candidate_substitutions_invariants(self, shared_device):
        seq_len = 20
        batch_size = 64
        top_k = 32
        vocab_size = 500

        current_ids = torch.randint(10, 400, (1, seq_len), device=shared_device)
        opt_slice = slice(5, 10)  # length 5

        # Mock token gradients: lower is better
        token_grads = torch.randn(5, vocab_size, device=shared_device)

        candidates, scores = sample_candidate_substitutions(
            current_ids=current_ids,
            opt_slice=opt_slice,
            token_gradients=token_grads,
            batch_size=batch_size,
            top_k=top_k,
        )

        assert candidates.shape == (batch_size, seq_len)
        assert candidates.device.type == shared_device.type

        # Invariant 1: Candidate 0 MUST be identical to current_ids (monotonicity anchor)
        assert torch.equal(candidates[0], current_ids[0]), "Candidate 0 must be unmodified identity."

        # Invariant 2: Prefix [0:5] and Suffix [10:20] must remain completely unchanged across all candidates
        assert torch.equal(candidates[:, :5], current_ids[:, :5].repeat(batch_size, 1))
        assert torch.equal(candidates[:, 10:], current_ids[:, 10:].repeat(batch_size, 1))

        # Invariant 3: Candidates 1..B must have modifications strictly inside opt_slice [5:10]
        mutations = candidates[1:, 5:10] != current_ids[0, 5:10]
        assert mutations.any(), "Candidates must introduce token substitutions inside opt_slice."


class TestMicroBatchedEvaluation:
    """Validate micro-batched forward pass and loss extraction."""

    def test_evaluate_candidates_batched(self, small_lm_and_wrapper, shared_device):
        model, _, _ = small_lm_and_wrapper
        seq_len = 14
        batch_size = 20
        candidate_ids = torch.randint(10, 400, (batch_size, seq_len), device=shared_device)

        target_slice = slice(8, 12)
        loss_slice = slice(7, 11)

        result = evaluate_candidates_batched(
            model=model,
            candidate_ids=candidate_ids,
            loss_slice=loss_slice,
            target_slice=target_slice,
            micro_batch_size=6,  # Non-divisible batch size
            device=shared_device,
        )

        assert result.evaluated_batch_size == batch_size
        assert result.exact_losses is not None
        assert result.exact_losses.shape == (batch_size,)
        assert 0 <= result.best_candidate_idx < batch_size
        assert result.best_loss == float(result.exact_losses[result.best_candidate_idx].item())
        assert torch.isfinite(result.exact_losses).all()


class TestEndToEndGCGOptimizer:
    """Validate full GCG optimizer loop execution."""

    def test_optimizer_step_and_monotonicity(self, small_lm_and_wrapper, shared_device):
        _, _, wrapper = small_lm_and_wrapper
        seq_len = 16
        input_ids = torch.randint(10, 400, (1, seq_len), device=shared_device)

        opt_slice = slice(4, 8)
        target_slice = slice(11, 15)
        loss_slice = slice(10, 14)

        slices = TokenSlices(
            prefix_slice=slice(0, 4),
            opt_slice=opt_slice,
            target_slice=target_slice,
            loss_slice=loss_slice,
        )

        cfg = OptimizerConfig(
            num_steps=3,
            batch_size=16,
            top_k=8,
            micro_batch_size=8,
            device=str(shared_device),
            early_stopping_loss=0.001,
            seed=42,
        )
        opt = CoordinateGradientOptimizer(model_wrapper=wrapper, config=cfg)

        traj = opt.optimize_sequence(
            current_ids=input_ids,
            slices=slices,
            log_interval=1,
        )

        assert len(traj.history) == 3
        assert traj.best_loss <= traj.initial_loss, "Greedy search must not degrade initial loss."
        assert traj.total_candidates_evaluated == 3 * 16
