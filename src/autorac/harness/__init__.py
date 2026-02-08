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
    ActualScores,
    AgentSuggestion,
    create_run,
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
from .metrics import (
    CalibrationMetrics,
    CalibrationSnapshot,
    compute_calibration,
    print_calibration_report,
    save_calibration_snapshot,
    get_calibration_trend,
)

__all__ = [
    # Experiment DB
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
    # Calibration Metrics
    "CalibrationMetrics",
    "CalibrationSnapshot",
    "compute_calibration",
    "print_calibration_report",
    "save_calibration_snapshot",
    "get_calibration_trend",
]
