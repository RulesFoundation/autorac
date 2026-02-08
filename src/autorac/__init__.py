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
    ActualScores,
    AgentSuggestion,
    create_run,
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
    # Calibration Metrics
    CalibrationMetrics,
    CalibrationSnapshot,
    compute_calibration,
    print_calibration_report,
    save_calibration_snapshot,
    get_calibration_trend,
)

__all__ = [
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
