"""
Pytest fixtures for autorac tests.
"""

import pytest
import tempfile
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, MagicMock
from typing import Optional

# Add src to path for imports - make src accessible as 'autorac'
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from autorac import (
    ExperimentDB,
    EncodingRun,
    PredictedScores,
    ValidatorPipeline,
    ValidationResult,
    PipelineResult,
    FinalScores,
)


@pytest.fixture
def temp_db_path():
    """Create a temporary database path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_experiments.db"


@pytest.fixture
def experiment_db(temp_db_path):
    """Create a temporary experiment database."""
    return ExperimentDB(temp_db_path)


@pytest.fixture
def sample_predicted_scores():
    """Sample predicted scores for testing."""
    return PredictedScores(
        rac=7.5,
        formula=7.0,
        param=8.0,
        integration=7.5,
        iterations=1,
        time_minutes=5.0,
        confidence=0.6,
    )


@pytest.fixture
def sample_final_scores():
    """Sample final scores for testing."""
    return FinalScores(
        rac_reviewer=8.0,
        formula_reviewer=6.5,
        parameter_reviewer=7.5,
        integration_reviewer=8.0,
        policyengine_match=0.88,
        taxsim_match=0.82,
    )


@pytest.fixture
def sample_encoding_run(sample_predicted_scores, sample_final_scores):
    """Create a sample encoding run."""
    run = EncodingRun(
        file_path="/path/to/statute.rac",
        citation="26 USC 32",
        agent_type="autorac:encoder",
        agent_model="claude-opus-4-5-20251101",
        rac_content="# EITC variable\nvariable EarnedIncome:\n  dtype: Money\n",
        statute_text="Sample statute text for EITC",
    )
    run.predicted_scores = sample_predicted_scores
    run.final_scores = sample_final_scores
    run.total_duration_ms = 4500
    return run


@pytest.fixture
def mock_validation_result():
    """Create a mock validation result that passes."""
    return PipelineResult(
        results={
            "ci": ValidationResult(
                validator_name="ci",
                passed=True,
                score=None,
                issues=[],
                duration_ms=100,
            ),
            "rac_reviewer": ValidationResult(
                validator_name="rac_reviewer",
                passed=True,
                score=8.0,
                issues=[],
                duration_ms=500,
            ),
            "formula_reviewer": ValidationResult(
                validator_name="formula_reviewer",
                passed=True,
                score=7.5,
                issues=[],
                duration_ms=500,
            ),
            "parameter_reviewer": ValidationResult(
                validator_name="parameter_reviewer",
                passed=True,
                score=8.5,
                issues=[],
                duration_ms=500,
            ),
            "integration_reviewer": ValidationResult(
                validator_name="integration_reviewer",
                passed=True,
                score=8.0,
                issues=[],
                duration_ms=500,
            ),
            "policyengine": ValidationResult(
                validator_name="policyengine",
                passed=True,
                score=0.95,
                issues=[],
                duration_ms=1000,
            ),
            "taxsim": ValidationResult(
                validator_name="taxsim",
                passed=True,
                score=0.92,
                issues=[],
                duration_ms=1000,
            ),
        },
        total_duration_ms=2000,
        all_passed=True,
    )


@pytest.fixture
def mock_failing_validation_result():
    """Create a mock validation result that fails."""
    return PipelineResult(
        results={
            "ci": ValidationResult(
                validator_name="ci",
                passed=False,
                score=None,
                issues=["Parse error: unexpected token"],
                duration_ms=100,
                error="Parse error: unexpected token",
            ),
            "rac_reviewer": ValidationResult(
                validator_name="rac_reviewer",
                passed=True,
                score=5.0,
                issues=["Missing citation reference"],
                duration_ms=500,
            ),
            "formula_reviewer": ValidationResult(
                validator_name="formula_reviewer",
                passed=True,
                score=4.5,
                issues=["Formula logic incorrect"],
                duration_ms=500,
            ),
            "parameter_reviewer": ValidationResult(
                validator_name="parameter_reviewer",
                passed=True,
                score=6.0,
                issues=[],
                duration_ms=500,
            ),
            "integration_reviewer": ValidationResult(
                validator_name="integration_reviewer",
                passed=True,
                score=5.5,
                issues=[],
                duration_ms=500,
            ),
        },
        total_duration_ms=1500,
        all_passed=False,
    )


@pytest.fixture
def temp_rac_file():
    """Create a temporary RAC file for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        rac_file = Path(tmpdir) / "test.rac"
        rac_file.write_text("""
# Test variable
variable TestIncome:
  entity: Person
  dtype: Money
  period: Year
  formula: |
    return 1000
""")
        yield rac_file


@pytest.fixture
def mock_agent():
    """Create a mock agent for testing encoder harness."""
    mock = Mock()
    mock.predict.return_value = PredictedScores(
        rac=8.0,
        formula=7.5,
        param=8.5,
        integration=8.0,
        iterations=1,
        time_minutes=5.0,
        confidence=0.7,
    )
    mock.encode.return_value = "# Encoded content\n"
    mock.suggest.return_value = []
    return mock
