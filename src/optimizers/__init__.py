"""Discrete search and coordinate gradient optimizers for LLM sensitivity and robustness."""

from src.optimizers.base import (
    CandidateEvaluationBatch,
    DiscreteOptimizer,
    OptimizationStepResult,
    OptimizationTrajectory,
    OptimizerConfig,
)
from src.optimizers.coordinate_gradient import (
    CoordinateGradientOptimizer,
    compute_token_gradients,
    evaluate_candidates_batched,
    extract_embedding_matrix,
    sample_candidate_substitutions,
)
from src.optimizers.draft_filter import (
    DraftFilterConfig,
    DraftModelFilter,
    DraftScreeningResult,
)

__all__ = [
    "DiscreteOptimizer",
    "OptimizationStepResult",
    "OptimizationTrajectory",
    "CandidateEvaluationBatch",
    "OptimizerConfig",
    "CoordinateGradientOptimizer",
    "extract_embedding_matrix",
    "compute_token_gradients",
    "sample_candidate_substitutions",
    "evaluate_candidates_batched",
    "DraftFilterConfig",
    "DraftModelFilter",
    "DraftScreeningResult",
]
