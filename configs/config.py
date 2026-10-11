"""Configuration dataclasses and loaders for models, discrete optimizers, and evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


@dataclass
class ModelTargetConfig:
    """Target causal language model configuration."""

    model_name_or_path: str = "meta-llama/Llama-3.2-1B-Instruct"
    torch_dtype: str = "float16"
    device_map: Optional[str] = "mps"
    trust_remote_code: bool = True
    attn_implementation: Optional[str] = "sdpa"
    use_cache: bool = False
    max_sequence_length: int = 1024


@dataclass
class OptimizerHyperparameters:
    """Discrete coordinate gradient optimization hyperparameters."""

    num_steps: int = 250
    batch_size: int = 256
    top_k: int = 128
    micro_batch_size: int = 16
    coordinate_selection_strategy: str = "random"  # "random", "all", "gradient_norm"
    early_stopping_loss: float = 0.05
    allow_non_ascii: bool = False
    filter_special_tokens: bool = True
    seed: int = 42


@dataclass
class EvaluationSuiteConfig:
    """Empirical robustness and loss curvature evaluation suite configuration."""

    benchmark_name: str = "sensitivity_profiling"
    num_samples: int = 20
    output_dir: str = "experiments/results"
    save_trajectories: bool = True
    profile_memory: bool = True


@dataclass
class ExperimentConfig:
    """Master experiment configuration container."""

    model: ModelTargetConfig = field(default_factory=ModelTargetConfig)
    optimizer: OptimizerHyperparameters = field(default_factory=OptimizerHyperparameters)
    evaluation: EvaluationSuiteConfig = field(default_factory=EvaluationSuiteConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> ExperimentConfig:
        """Load experiment configuration from YAML file."""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}

        model_cfg = ModelTargetConfig(**data.get("model", {}))
        opt_cfg = OptimizerHyperparameters(**data.get("optimizer", {}))
        eval_cfg = EvaluationSuiteConfig(**data.get("evaluation", {}))
        return cls(model=model_cfg, optimizer=opt_cfg, evaluation=eval_cfg)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Serialize configuration to YAML file."""
        data = {
            "model": self.model.__dict__,
            "optimizer": self.optimizer.__dict__,
            "evaluation": self.evaluation.__dict__,
        }
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
