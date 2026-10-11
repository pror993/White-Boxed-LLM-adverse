"""Sanity tests verifying Apple Silicon MPS device, memory tracking, and autograd graph.

Ensures that Metal Performance Shaders (MPS) is available, embedding projection
gradients correctly propagate through causal language model computational graphs,
and memory tracking functions reliably within 16GB unified memory environments.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

from src.optimizers.base import (
    CandidateEvaluationBatch,
    DiscreteOptimizer,
    OptimizationStepResult,
    OptimizationTrajectory,
    OptimizerConfig,
)
from src.utils.device import (
    DeviceConfig,
    MPSMemorySnapshot,
    MPSMemoryTracker,
    clear_device_cache,
    estimate_safe_micro_batch_size,
    get_best_device,
    get_device_config,
    get_mps_memory_snapshot,
    is_apple_silicon,
    is_mps_available,
    safe_mps_context,
    synchronize_device,
)


class TestEnvironmentAndDevice:
    """Validate Apple Silicon host environment and MPS runtime."""

    def test_apple_silicon_detection(self) -> None:
        """Verify host machine is identified as Apple Silicon ARM64."""
        assert is_apple_silicon() is True, "Host machine should be Apple Silicon (Darwin arm64)."

    def test_mps_availability(self) -> None:
        """Verify MPS is built and available in PyTorch."""
        assert torch.backends.mps.is_built() is True, "PyTorch must be built with MPS support."
        assert torch.backends.mps.is_available() is True, "MPS device must be available."
        assert is_mps_available() is True, "is_mps_available() probe must return True."

    def test_device_resolution(self) -> None:
        """Verify device fallback and best device resolution."""
        device = get_best_device("mps")
        assert device.type == "mps", f"Expected 'mps' device, got: {device}"

        cpu_device = get_best_device(preferred="cpu")
        assert cpu_device.type == "cpu", f"Expected 'cpu' device, got: {cpu_device}"

    def test_device_config(self) -> None:
        """Verify default DeviceConfig initialization."""
        cfg = get_device_config()
        assert cfg.preferred_device in ("mps", "cpu")
        assert cfg.default_dtype in ("float16", "float32")
        assert cfg.mps_memory_fraction == 0.75


class TestMPSMemoryManagement:
    """Validate Metal Performance Shaders telemetry and unified memory accounting."""

    def test_mps_memory_snapshot(self) -> None:
        """Verify memory snapshots return populated telemetry."""
        snapshot = get_mps_memory_snapshot("mps")
        assert isinstance(snapshot, MPSMemorySnapshot)
        assert snapshot.device == "mps"
        assert snapshot.system_total_bytes > 0
        assert snapshot.system_available_bytes > 0
        assert snapshot.process_rss_bytes > 0
        assert snapshot.system_total_mb > 1000.0  # At least 1GB

        data_dict = snapshot.to_dict()
        assert "allocated_mb" in data_dict
        assert "driver_allocated_mb" in data_dict
        assert "process_rss_mb" in data_dict

    def test_mps_memory_tracker_context(self) -> None:
        """Verify memory tracker context manager records allocations."""
        device = torch.device("mps")
        with MPSMemoryTracker("tensor_alloc_test", device=device) as tracker:
            # Allocate a 20MB tensor on MPS
            t = torch.randn(5000, 1000, device=device, dtype=torch.float32)
            synchronize_device(device)
            assert t.shape == (5000, 1000)
            del t

        summary = tracker.summary()
        assert summary["name"] == "tensor_alloc_test"
        assert summary["device"] == "mps"
        assert "delta_allocated_mb" in summary

    def test_safe_mps_context_and_cache_clear(self) -> None:
        """Verify safe MPS execution context and cache eviction."""
        device = torch.device("mps")
        with safe_mps_context(device=device):
            x = torch.ones(100, 100, device=device)
            y = x @ x
            assert y.sum().item() == 1000000.0

        # Memory cleanup should succeed without error
        clear_device_cache(device)
        synchronize_device(device)

    def test_micro_batch_size_estimator(self) -> None:
        """Verify safe micro-batch sizing respects 16GB memory bounds."""
        # 1B parameter model with seq_len 128
        batch_1b = estimate_safe_micro_batch_size(
            model_param_count=int(1.1e9),
            seq_len=128,
            total_memory_gb=16.0,
            safety_margin_gb=4.0,
        )
        assert 8 <= batch_1b <= 128
        assert batch_1b % 8 == 0 or batch_1b == 1

        # Very large hypothetical model (e.g. 70B) should clamp to 1
        batch_large = estimate_safe_micro_batch_size(
            model_param_count=int(70e9),
            seq_len=128,
            total_memory_gb=16.0,
            safety_margin_gb=4.0,
        )
        assert batch_large >= 1


class TestMPSAutogradAndGradientPropagation:
    """Validate causal LM embedding gradient propagation on MPS.

    Discrete optimization (GCG, HotFlip) fundamentally requires computing
    dL / d(embedding) at input positions, then projecting onto the embedding matrix
    to evaluate first-order Taylor approximations of token substitutions.
    """

    @pytest.fixture
    def small_causal_lm(self) -> nn.Module:
        """Instantiate a lightweight CausalLM for fast, self-contained graph testing."""
        from transformers import GPT2Config, GPT2LMHeadModel

        config = GPT2Config(
            vocab_size=1000,
            n_positions=128,
            n_embd=64,
            n_layer=2,
            n_head=2,
            use_cache=False,
        )
        model = GPT2LMHeadModel(config)
        return model

    def test_embedding_gradient_propagation_on_mps(self, small_causal_lm: nn.Module) -> None:
        """Verify exact gradient propagation through embeddings on MPS."""
        device = torch.device("mps")
        model = small_causal_lm.to(device)
        model.eval()

        seq_len = 16
        input_ids = torch.randint(0, 1000, (1, seq_len), device=device)

        # Retrieve embedding weights
        embedding_layer = model.get_input_embeddings()
        embeddings_weight = embedding_layer.weight  # [vocab_size, hidden_size]

        # Discrete optimizers hook into one-hot or embedding space:
        # Create input_embeds with gradient tracking
        input_embeds = embedding_layer(input_ids).detach().clone().requires_grad_(True)
        assert input_embeds.device.type == "mps"
        assert input_embeds.requires_grad is True

        # Forward pass through causal LM with input_embeds
        outputs = model(inputs_embeds=input_embeds)
        logits = outputs.logits  # [1, seq_len, vocab_size]
        assert logits.shape == (1, seq_len, 1000)

        # Causal language modeling target prediction (predicting next token)
        # Shift logits and shift labels for loss computation
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()

        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(shift_logits.view(-1, 1000), shift_labels.view(-1))
        assert torch.isfinite(loss)
        assert loss.item() > 0.0

        # Backward pass on MPS
        loss.backward()

        # Gradient with respect to input embeddings must exist and be on MPS
        assert input_embeds.grad is not None, "Gradients must propagate back to inputs_embeds."
        assert input_embeds.grad.device.type == "mps"
        assert input_embeds.grad.shape == (1, seq_len, 64)
        assert torch.isfinite(input_embeds.grad).all(), "Gradients must be finite."
        assert (input_embeds.grad != 0).any(), "Gradients should have non-zero components."

        # Verify first-order Taylor projection onto vocabulary:
        # For a coordinate i, grad_i has shape [1, hidden_dim].
        # Dot product with embedding matrix: [hidden_dim] @ [hidden_dim, vocab_size] -> [vocab_size]
        coord_grad = input_embeds.grad[0, 5, :]  # [64]
        token_gradient_scores = coord_grad @ embeddings_weight.T  # [1000]
        assert token_gradient_scores.shape == (1000,)
        assert token_gradient_scores.device.type == "mps"

    def test_candidate_evaluation_micro_batching(self, small_causal_lm: nn.Module) -> None:
        """Verify micro-batched forward evaluation in DiscreteOptimizer."""
        device = torch.device("mps")
        model = small_causal_lm.to(device)

        # Define a mock tokenizer for testing
        class MockTokenizer:
            def decode(self, token_ids):
                return " ".join(str(t) for t in token_ids)

        class ConcreteOptimizer(DiscreteOptimizer):
            def compute_coordinate_gradients(self, input_ids, target_slice, loss_slice, **kwargs):
                return torch.zeros(1, 64, device=self.device), 1.0

            def sample_candidates(self, input_ids, target_slice, coordinate_gradients, batch_size, top_k, **kwargs):
                candidates = input_ids.repeat(batch_size, 1)
                scores = torch.zeros(batch_size, device=self.device)
                return candidates, scores

            def step(self, current_ids, target_slice, loss_slice, step_idx):
                return OptimizationStepResult(
                    step=step_idx,
                    loss=0.5,
                    best_candidate_tokens=current_ids[0].tolist(),
                    best_candidate_text="test",
                    candidates_evaluated=10,
                )

        cfg = OptimizerConfig(batch_size=32, micro_batch_size=8, device="mps")
        opt = ConcreteOptimizer(model=model, tokenizer=MockTokenizer(), config=cfg)

        # Create candidate tensor of 24 sequences of length 12
        candidate_ids = torch.randint(0, 1000, (24, 12))
        loss_slice = slice(6, 12)

        batch_result = opt.evaluate_candidates(
            candidate_ids=candidate_ids,
            loss_slice=loss_slice,
            micro_batch_size=8,
        )

        assert isinstance(batch_result, CandidateEvaluationBatch)
        assert batch_result.evaluated_batch_size == 24
        assert batch_result.exact_losses is not None
        assert batch_result.exact_losses.shape == (24,)
        assert 0 <= batch_result.best_candidate_idx < 24
        assert torch.isfinite(batch_result.exact_losses).all()


class TestConfigurations:
    """Validate YAML configuration loading and parsing."""

    def test_load_yaml_configs(self) -> None:
        from configs.config import ExperimentConfig

        llama_cfg = ExperimentConfig.from_yaml("configs/llama_3_2_1b.yaml")
        assert llama_cfg.model.model_name_or_path == "meta-llama/Llama-3.2-1B-Instruct"
        assert llama_cfg.optimizer.micro_batch_size == 16
        assert llama_cfg.model.device_map == "mps"

        qwen_cfg = ExperimentConfig.from_yaml("configs/qwen_2_5_1_5b.yaml")
        assert qwen_cfg.model.model_name_or_path == "Qwen/Qwen2.5-1.5B-Instruct"
        assert qwen_cfg.optimizer.batch_size == 256
        assert qwen_cfg.model.torch_dtype == "bfloat16"
