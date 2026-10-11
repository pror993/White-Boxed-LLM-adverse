"""Comprehensive tests for multi-fidelity draft filtering, Mellowmax, perplexity, and sensitivity profiling.

Validates mathematical invariants, numerical stability, and empirical curvature metrics.
"""

from __future__ import annotations

import math
import pytest
import torch
import torch.nn as nn
from transformers import GPT2Config, GPT2LMHeadModel

from src.eval.sensitivity import (
    DiscreteSensitivityProfiler,
    NoiseLevelEvaluation,
    PerturbationType,
    SensitivityProfileResult,
    perturb_tokens,
)
from src.models.causal_lm import CausalLMModelWrapper, TokenSlices
from src.objectives.mellowmax import MellowmaxLoss, mellowmax_loss
from src.objectives.regularizers import CompositeObjective, PerplexityRegularizer
from src.optimizers.draft_filter import DraftFilterConfig, DraftModelFilter
from src.utils.device import get_best_device


@pytest.fixture(scope="module")
def device():
    return get_best_device("mps")


@pytest.fixture(scope="module")
def synthetic_lm(device):
    """Fixture providing a lightweight causal LM for testing."""
    config = GPT2Config(vocab_size=300, n_positions=64, n_embd=48, n_layer=2, n_head=2, use_cache=False)
    model = GPT2LMHeadModel(config).to(device=device, dtype=torch.float32)
    model.eval()

    class MockTokenizer:
        def __init__(self, vocab_size=300):
            self.vocab_size = vocab_size
            self.pad_token_id = 0
            self.eos_token_id = 1
            self.all_special_ids = [0, 1]
            self.chat_template = None

        def decode(self, ids, skip_special_tokens=False):
            return " ".join(str(i) for i in ids)

        def __call__(self, text, **kwargs):
            class Out:
                input_ids = [10, 20, 30]
            return Out()

    wrapper = CausalLMModelWrapper(model=model, tokenizer=MockTokenizer(300), device=device)
    return model, wrapper


class TestDraftModelFilter:
    """Validate two-stage draft screening mechanics and monotonicity."""

    def test_draft_screening_retention_and_monotonicity(self, synthetic_lm, device):
        model, _ = synthetic_lm
        screener = DraftModelFilter(
            draft_model=model,
            config=DraftFilterConfig(filtering_ratio=4, min_retained_candidates=4, always_retain_identity=True),
            device=device,
        )

        b_total = 32
        seq_len = 16
        candidates = torch.randint(5, 250, (b_total, seq_len), device=device)
        loss_slice = slice(8, 12)
        target_slice = slice(9, 13)

        result = screener.filter_candidates(
            candidate_ids=candidates,
            loss_slice=loss_slice,
            target_slice=target_slice,
            filtering_ratio=4,
        )

        assert result.original_batch_size == 32
        assert result.filtered_batch_size == 8  # 32 // 4 = 8
        assert result.filtered_candidate_ids.shape == (8, seq_len)
        # Invariant: candidate 0 must be retained at index 0
        assert torch.equal(result.filtered_candidate_ids[0], candidates[0])
        assert result.selected_indices[0].item() == 0


class TestMellowmaxLoss:
    """Validate mathematical properties, temperature interpolation, and stability."""

    def test_mellowmax_alpha_zero_approaches_mean(self):
        token_losses = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
        # alpha -> 0 should equal arithmetic mean = 2.5
        val = mellowmax_loss(token_losses, alpha=1e-5, dim=-1)
        expected = token_losses.mean(dim=-1)
        assert torch.allclose(val, expected, atol=1e-3)

    def test_mellowmax_large_alpha_approaches_max(self):
        token_losses = torch.tensor([[1.0, 2.0, 3.0, 10.0]])
        # alpha large (e.g. 20.0) should approach max token loss = 10.0
        val = mellowmax_loss(token_losses, alpha=20.0, dim=-1)
        expected_max = torch.tensor([10.0])
        assert torch.allclose(val, expected_max, atol=0.2)

    def test_mellowmax_loss_module_batched(self, device):
        mellow_mod = MellowmaxLoss(alpha=1.5)
        batch = 4
        target_len = 6
        vocab_size = 50

        logits = torch.randn(batch, target_len, vocab_size, device=device)
        labels = torch.randint(0, vocab_size, (batch, target_len), device=device)

        loss = mellow_mod(logits, labels)
        assert loss.shape == (batch,)
        assert torch.isfinite(loss).all()
        assert (loss > 0).all()


class TestPerplexityRegularizer:
    """Validate sequence perplexity and composite objective evaluation."""

    def test_perplexity_nll_and_regularization(self, synthetic_lm, device):
        model, _ = synthetic_lm
        reg = PerplexityRegularizer(model=model, use_log_ppl=True)

        token_ids = torch.randint(5, 250, (2, 8), device=device)
        nll = reg.compute_sequence_nll(token_ids)
        assert nll.shape == (2,)
        assert (nll >= 0.0).all()
        assert torch.isfinite(nll).all()

        ppl = reg.compute_perplexity(token_ids)
        assert (ppl >= 1.0).all()

        # Test composite objective
        target_loss = torch.tensor([3.0, 2.5], device=device)
        reg_loss = reg(target_loss=target_loss, opt_token_ids=token_ids, lambda_weight=0.1)
        assert (reg_loss > target_loss).all()


class TestDiscreteSensitivityProfiler:
    """Validate discrete perturbation algorithms and sensitivity profiling."""

    def test_perturb_tokens_confinement(self, device):
        seq_len = 16
        token_ids = torch.randint(10, 200, (1, seq_len), device=device)
        opt_slice = slice(4, 10)

        perturbed = perturb_tokens(
            token_ids=token_ids,
            opt_slice=opt_slice,
            noise_rate=0.5,
            vocab_size=300,
            strategy=PerturbationType.TOKEN_SUBSTITUTION,
        )

        assert perturbed.shape == token_ids.shape
        # Outside slice must be strictly untouched
        assert torch.equal(perturbed[0, :4], token_ids[0, :4])
        assert torch.equal(perturbed[0, 10:], token_ids[0, 10:])

    def test_sensitivity_profiling_execution(self, synthetic_lm, device):
        _, wrapper = synthetic_lm
        seq_len = 16
        input_ids = torch.randint(10, 250, (1, seq_len), device=device)

        slices = TokenSlices(
            prefix_slice=slice(0, 4),
            opt_slice=slice(4, 9),
            target_slice=slice(11, 15),
            loss_slice=slice(10, 14),
        )

        profiler = DiscreteSensitivityProfiler(model_wrapper=wrapper, device=device)
        result = profiler.profile_sequence(
            input_ids=input_ids,
            slices=slices,
            target_text="test target",
            noise_levels=[0.05, 0.10],
            samples_per_noise_level=2,
            micro_batch_size=2,
        )

        assert isinstance(result, SensitivityProfileResult)
        assert result.baseline_loss > 0.0
        assert len(result.noise_evaluations) == 2
        assert isinstance(result.sharpness_index, float)
        assert isinstance(result.empirical_curvature, float)
