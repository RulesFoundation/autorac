# AutoRAC - AI-assisted RAC encoding
# Feedback loop infrastructure for automated statute encoding

from .harness import (
    # Experiment DB
    ExperimentDB,
    EncodingRun,
    PredictedScores,
    ActualScores,
    AgentSuggestion,
    create_run,
    # Validator Pipeline
    ValidatorPipeline,
    ValidationResult,
    PipelineResult,
    validate_file,
    # Encoder Harness
    EncoderHarness,
    EncoderConfig,
    run_encoding_experiment,
    # Metrics
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
    "PredictedScores",
    "ActualScores",
    "AgentSuggestion",
    "create_run",
    "ValidatorPipeline",
    "ValidationResult",
    "PipelineResult",
    "validate_file",
    "EncoderHarness",
    "EncoderConfig",
    "run_encoding_experiment",
    "CalibrationMetrics",
    "CalibrationSnapshot",
    "compute_calibration",
    "print_calibration_report",
    "save_calibration_snapshot",
    "get_calibration_trend",
]
