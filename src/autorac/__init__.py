# AutoRAC - AI-assisted RAC encoding
# Feedback loop infrastructure for automated statute encoding

from .constants import DEFAULT_CLI_MODEL, DEFAULT_MODEL, REVIEWER_CLI_MODEL
from .harness import (
    ActualScores,
    AgentSDKBackend,
    AgentSuggestion,
    # Calibration Metrics
    CalibrationMetrics,
    CalibrationSnapshot,
    ClaudeCodeBackend,
    ComplexityFactors,
    # Encoder Backends
    EncoderBackend,
    EncoderRequest,
    EncoderResponse,
    EncodingRun,
    # Experiment DB
    ExperimentDB,
    FinalScores,
    Iteration,
    IterationError,
    PipelineResult,
    PredictedScores,
    PredictionScores,
    ReviewResult,
    ReviewResults,
    ValidationResult,
    # Validator Pipeline
    ValidatorPipeline,
    compute_calibration,
    create_run,
    get_calibration_trend,
    print_calibration_report,
    save_calibration_snapshot,
    validate_file,
)

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_CLI_MODEL",
    "REVIEWER_CLI_MODEL",
    "ExperimentDB",
    "EncodingRun",
    "ComplexityFactors",
    "IterationError",
    "Iteration",
    "ReviewResult",
    "ReviewResults",
    "FinalScores",
    "PredictedScores",
    "ActualScores",
    "AgentSuggestion",
    "create_run",
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
    # Calibration Metrics
    "CalibrationMetrics",
    "CalibrationSnapshot",
    "compute_calibration",
    "print_calibration_report",
    "save_calibration_snapshot",
    "get_calibration_trend",
]
