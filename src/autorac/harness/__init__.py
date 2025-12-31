# Encoding Harness
# Feedback loop for AI-assisted statute encoding

from .experiment_db import (
    ExperimentDB,
    EncodingRun,
    ComplexityFactors,
    IterationError,
    Iteration,
    FinalScores,
    PredictedScores,
)
from .validator_pipeline import (
    ValidatorPipeline,
    ValidationResult,
    PipelineResult,
    validate_file,
)
from .backends import (
    EncoderBackend,
    ClaudeCodeBackend,
    AgentSDKBackend,
    EncoderRequest,
    EncoderResponse,
    PredictionScores,
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
    "PredictedScores",
    # Validator Pipeline
    "ValidatorPipeline",
    "ValidationResult",
    "PipelineResult",
    "validate_file",
    # Encoder Backends
    "EncoderBackend",
    "ClaudeCodeBackend",
    "AgentSDKBackend",
    "EncoderRequest",
    "EncoderResponse",
    "PredictionScores",
]
