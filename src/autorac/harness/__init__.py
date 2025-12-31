# Encoding Harness
# Feedback loop for AI-assisted statute encoding

from .experiment_db import (
    ExperimentDB,
    EncodingRun,
    ComplexityFactors,
    IterationError,
    Iteration,
    FinalScores,
)
from .validator_pipeline import (
    ValidatorPipeline,
    ValidationResult,
    PipelineResult,
    validate_file,
)

# Note: encoder_harness and metrics need updating for new journey-based model
# They still use the old prediction-based approach

__all__ = [
    # Experiment DB
    "ExperimentDB",
    "EncodingRun",
    "ComplexityFactors",
    "IterationError",
    "Iteration",
    "FinalScores",
    # Validator Pipeline
    "ValidatorPipeline",
    "ValidationResult",
    "PipelineResult",
    "validate_file",
]
