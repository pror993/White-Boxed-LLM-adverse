# White-Boxed LLM Adverse: Discrete Optimization & Adversarial Robustness

A research framework and library for analyzing **first-order discrete optimization methods**, **loss curvature**, and **adversarial robustness dynamics** in autoregressive causal language models.

Optimized for **Apple Silicon Metal acceleration (`device = torch.device("mps")`)** within a 16GB Unified Memory envelope.

---

## 🔬 Research & Mathematical Scope

1. **Discrete Optimization over Finite Vocabularies ($\mathcal{V}$)**:
   - Coordinate gradient search (Greedy Coordinate Gradient / GCG, HotFlip).
   - First-order Taylor series approximations of token substitutions:
     $$\Delta \mathcal{L}(x_{i \to v}) \approx \langle \nabla_{e(x_i)} \mathcal{L}(x), e(v) - e(x_i) \rangle$$
   - Embedding projection and exact linear surrogate ranking across top-$k$ vocabulary coordinates.
2. **Loss Landscape & Sensitivity Profiling**:
   - Model sensitivity benchmarks, token curvature introspection, and gradient norm tracking.
   - Randomized smoothing and empirical robustness dynamics on small-scale instruction models (1B–3B parameters, e.g., LLaMA-3.2-1B, Qwen-2.5-1.5B).
3. **Apple Silicon Hardware Optimization**:
   - Native PyTorch Metal Performance Shaders (`mps`) execution.
   - Exact autograd graph backpropagation through embedding tensors to avoid full vocabulary graph unrolling.
   - Configurable micro-batch candidate evaluation preventing unified memory exhaustion.

---

## 📁 Repository Structure

```text
├── configs/                  # Model targets, optimizer hyperparameters, evaluation configs
│   ├── config.py             # Dataclass schemas and YAML loader
│   ├── llama_3_2_1b.yaml     # Llama-3.2-1B MPS configuration
│   └── qwen_2_5_1_5b.yaml    # Qwen-2.5-1.5B MPS configuration
├── src/
│   ├── optimizers/           # Discrete search and coordinate gradient optimizers
│   │   ├── base.py           # DiscreteOptimizer ABC, Trajectory & StepResult dataclasses
│   │   └── coordinate_gradient.py # Coordinate gradient search algorithms
│   ├── models/               # Model wrappers, embedding hooks, token slicing
│   ├── objectives/           # Cross-entropy, Mellowmax, Perplexity regularizers
│   └── utils/                # MPS telemetry, memory trackers, device management
│       └── device.py         # Apple Silicon MPS detection, snapshotting, cache management
├── experiments/              # Convergence, sensitivity profiling, robustness suites
├── tests/                    # Test suite for token alignment, gradients, MPS autograd
│   └── test_environment.py   # Sanity verification for MPS and autograd graph
├── pyproject.toml            # Project packaging specification
└── requirements.txt          # Production dependencies
```

---

## ⚡ Quickstart

### 1. Activate Environment

```bash
source .venv/bin/activate
```

### 2. Verify Metal (MPS) Acceleration & Test Suite

Run the full automated test suite verifying device detection, unified memory tracking, and embedding gradient propagation on MPS:

```bash
pytest tests/test_environment.py -v
```

---

## 📊 Core Interfaces

### Device & Memory Profiling (`src/utils/device.py`)

- `get_best_device()`: Returns `torch.device("mps")` on Apple Silicon with fallback to CPU.
- `get_mps_memory_snapshot()`: Captures driver-allocated Metal memory and host unified RAM.
- `MPSMemoryTracker`: Context manager profiling peak memory delta across tensor operations.
- `estimate_safe_micro_batch_size()`: Heuristic estimator sizing candidate evaluation chunks within memory ceilings.

### Discrete Optimizer Contract (`src/optimizers/base.py`)

- `compute_coordinate_gradients(...)`: Calculates exact $\nabla_E \mathcal{L}$ through input embeddings.
- `sample_candidates(...)`: Generates $B$ candidate sequences via first-order Taylor approximation scores.
- `evaluate_candidates(...)`: Runs micro-batched forward evaluation on MPS.
- `optimize(...)`: Complete iterative discrete trajectory tracking with early-stopping criteria.
