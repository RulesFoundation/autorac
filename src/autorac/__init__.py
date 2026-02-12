# AutoRAC - AI-assisted RAC encoding
# Feedback loop infrastructure for automated statute encoding

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

# Full model ID for Agent SDK (API) backend
DEFAULT_MODEL = "claude-opus-4-6"

# Short-form model names for Claude Code CLI backend
DEFAULT_CLI_MODEL = "opus"
REVIEWER_CLI_MODEL = "opus"

__all__ = [
    "DEFAULT_MODEL",
    "DEFAULT_CLI_MODEL",
    "REVIEWER_CLI_MODEL",
    "ExperimentDB",
    "EncodingRun",
    "ComplexityFactors",
    "IterationError",
    "Iteration",
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
