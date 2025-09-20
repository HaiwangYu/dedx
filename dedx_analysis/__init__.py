"""High-level API for dedx analysis package."""

from .analysis import DedxAnalysisConfig, run_analysis
from .pipeline import (
    PIDBandResult,
    SpeciesEvaluationResult,
    evaluate_pid_bands,
    generate_pid_bands,
)

__all__ = [
    "DedxAnalysisConfig",
    "run_analysis",
    "PIDBandResult",
    "SpeciesEvaluationResult",
    "generate_pid_bands",
    "evaluate_pid_bands",
]

__version__ = "0.1.0"
