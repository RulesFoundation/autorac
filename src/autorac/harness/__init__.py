# Encoding Harness
# Feedback loop for AI-assisted statute encoding

from .experiment_db import (
    ExperimentDB,
    EncodingRun,
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
from .encoder_harness import (
    EncoderHarness,
    EncoderConfig,
    run_encoding_experiment,
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
    "PredictedScores",
    "ActualScores",
    "AgentSuggestion",
    "create_run",
    # Validator Pipeline
    "ValidatorPipeline",
    "ValidationResult",
    "PipelineResult",
    "validate_file",
    # Encoder Harness
    "EncoderHarness",
    "EncoderConfig",
    "run_encoding_experiment",
    # Metrics
    "CalibrationMetrics",
    "CalibrationSnapshot",
    "compute_calibration",
    "print_calibration_report",
    "save_calibration_snapshot",
    "get_calibration_trend",
]
