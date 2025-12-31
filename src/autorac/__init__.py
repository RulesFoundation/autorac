# AutoRAC - AI-assisted RAC encoding
# Feedback loop infrastructure for automated statute encoding

from .harness import (
    # Experiment DB
    ExperimentDB,
    EncodingRun,
    ComplexityFactors,
    IterationError,
    Iteration,
    FinalScores,
    PredictedScores,
    # Validator Pipeline
    ValidatorPipeline,
    ValidationResult,
    PipelineResult,
    validate_file,
    # Encoder Backends
    EncoderBackend,
    ClaudeCodeBackend,
    AgentSDKBackend,
    EncoderRequest,
    EncoderResponse,
    PredictionScores,
)

__all__ = [
    "ExperimentDB",
    "EncodingRun",
    "ComplexityFactors",
    "IterationError",
    "Iteration",
    "FinalScores",
    "PredictedScores",
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
