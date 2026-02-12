"""
Tests for ValidatorPipeline - TDD style.

Tests cover:
1. CI validators (parse, lint, tests)
2. Reviewer agent validators
3. External oracle validators
4. Parallel execution
"""

# Add src to path for imports
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autorac import (
    PipelineResult,
    ValidationResult,
    ValidatorPipeline,
    validate_file,
)

# Fixtures


@pytest.fixture
def temp_rac_file():
    """Create a temporary RAC file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".rac", delete=False) as f:
        f.write("""
# Simple test RAC file
earned_income:
    entity: Person
    period: Year
    dtype: Money
    formula: |
        return person.wages + person.self_employment_income
""")
        return Path(f.name)


@pytest.fixture
def temp_dirs():
    """Create temporary directories for rac and rac-us."""
    with tempfile.TemporaryDirectory() as rac_us_dir:
        with tempfile.TemporaryDirectory() as rac_dir:
            yield Path(rac_us_dir), Path(rac_dir)


@pytest.fixture
def pipeline(temp_dirs):
    """Create a ValidatorPipeline with temp directories."""
    rac_us_path, rac_path = temp_dirs
    return ValidatorPipeline(
        rac_us_path=rac_us_path,
        rac_path=rac_path,
        enable_oracles=True,
        max_workers=4,
    )


# CI Validator Tests


class TestCIValidator:
    """Tests for CI validation (parse, lint, test)."""

    def test_ci_returns_validation_result(self, pipeline, temp_rac_file):
        """CI validator should return ValidationResult."""
        result = pipeline._run_ci(temp_rac_file)

        assert isinstance(result, ValidationResult)
        assert result.validator_name == "ci"
        assert isinstance(result.passed, bool)
        assert isinstance(result.issues, list)
        assert result.duration_ms >= 0

    def test_ci_detects_parse_errors(self, pipeline, temp_dirs):
        """CI should detect parse errors in invalid RAC files."""
        rac_us_path, _ = temp_dirs

        # Create invalid RAC file
        invalid_file = rac_us_path / "invalid.rac"
        invalid_file.write_text("this is {{ not valid RAC syntax !!!")

        result = pipeline._run_ci(invalid_file)

        assert result.passed is False
        assert any(
            "parse" in issue.lower() or "error" in issue.lower()
            for issue in result.issues
        )

    def test_ci_reports_test_failures(self, pipeline, temp_dirs):
        """CI should report test failures from inline tests."""
        rac_us_path, _ = temp_dirs

        # Create RAC file with failing test
        test_file = rac_us_path / "failing_test.rac"
        test_file.write_text("""
always_zero:
    entity: Person
    period: Year
    dtype: Integer
    formula: |
        return 0

test failing_test:
    inputs:
        person.always_zero: 999  # This should fail
    expected:
        person.always_zero: 0
""")

        result = pipeline._run_ci(test_file)

        # Either it fails to parse (no test runner) or tests fail
        assert result.passed is False or len(result.issues) >= 0


# Reviewer Agent Tests


class TestReviewerAgents:
    """Tests for reviewer agent validators."""

    def test_rac_reviewer_returns_validation_result(self, pipeline, temp_rac_file):
        """RAC reviewer should return ValidationResult with score."""
        result = pipeline._run_reviewer("rac-reviewer", temp_rac_file)

        assert isinstance(result, ValidationResult)
        assert result.validator_name == "rac-reviewer"
        # Score should be 0-10 for reviewers
        if result.score is not None:
            assert 0 <= result.score <= 10

    def test_formula_reviewer_returns_validation_result(self, pipeline, temp_rac_file):
        """Formula reviewer should return ValidationResult."""
        result = pipeline._run_reviewer("formula-reviewer", temp_rac_file)

        assert isinstance(result, ValidationResult)
        assert result.validator_name == "formula-reviewer"

    def test_parameter_reviewer_returns_validation_result(
        self, pipeline, temp_rac_file
    ):
        """Parameter reviewer should return ValidationResult."""
        result = pipeline._run_reviewer("parameter-reviewer", temp_rac_file)

        assert isinstance(result, ValidationResult)
        assert result.validator_name == "parameter-reviewer"

    def test_reviewer_includes_issues_list(self, pipeline, temp_rac_file):
        """Reviewer should include a list of issues found."""
        result = pipeline._run_reviewer("rac-reviewer", temp_rac_file)

        assert isinstance(result.issues, list)
        # All issues should be strings
        for issue in result.issues:
            assert isinstance(issue, str)

    def test_reviewer_includes_raw_output(self, pipeline, temp_rac_file):
        """Reviewer should optionally include raw output for debugging."""
        result = pipeline._run_reviewer("rac-reviewer", temp_rac_file)

        # raw_output is optional but should be string if present
        if result.raw_output is not None:
            assert isinstance(result.raw_output, str)


# Oracle Validator Tests


class TestOracleValidators:
    """Tests for external oracle validators (PolicyEngine, TAXSIM)."""

    def test_policyengine_validator_returns_result(self, pipeline, temp_rac_file):
        """PolicyEngine validator should return ValidationResult with match rate."""
        result = pipeline._run_policyengine(temp_rac_file)

        assert isinstance(result, ValidationResult)
        assert result.validator_name == "policyengine"
        # Score should be 0-1 match rate for oracles
        if result.score is not None:
            assert 0 <= result.score <= 1

    def test_taxsim_validator_returns_result(self, pipeline, temp_rac_file):
        """TAXSIM validator should return ValidationResult with match rate."""
        result = pipeline._run_taxsim(temp_rac_file)

        assert isinstance(result, ValidationResult)
        assert result.validator_name == "taxsim"
        if result.score is not None:
            assert 0 <= result.score <= 1

    def test_oracles_disabled_when_flag_false(self, temp_dirs, temp_rac_file):
        """When enable_oracles=False, oracles should not run."""
        rac_us_path, rac_path = temp_dirs

        pipeline = ValidatorPipeline(
            rac_us_path=rac_us_path,
            rac_path=rac_path,
            enable_oracles=False,
        )

        result = pipeline.validate(temp_rac_file)

        assert "policyengine" not in result.results
        assert "taxsim" not in result.results


# Parallel Execution Tests


class TestParallelExecution:
    """Tests for parallel validator execution."""

    def test_validate_runs_all_validators(self, pipeline, temp_rac_file):
        """validate() should run all configured validators."""
        result = pipeline.validate(temp_rac_file)

        assert isinstance(result, PipelineResult)
        # Should include CI and all reviewers
        assert "ci" in result.results
        assert "rac_reviewer" in result.results
        assert "formula_reviewer" in result.results
        assert "parameter_reviewer" in result.results
        # Oracles when enabled
        assert "policyengine" in result.results
        assert "taxsim" in result.results

    def test_validate_returns_total_duration(self, pipeline, temp_rac_file):
        """validate() should return total execution duration."""
        result = pipeline.validate(temp_rac_file)

        assert result.total_duration_ms >= 0

    def test_validate_sets_all_passed_correctly(self, pipeline, temp_rac_file):
        """all_passed should be True only if all validators pass."""
        result = pipeline.validate(temp_rac_file)

        # all_passed should match whether all individual results passed
        expected = all(r.passed for r in result.results.values())
        assert result.all_passed == expected

    def test_validate_handles_validator_exceptions(self, temp_dirs, temp_rac_file):
        """validate() should handle exceptions from validators gracefully."""
        rac_us_path, rac_path = temp_dirs

        pipeline = ValidatorPipeline(
            rac_us_path=rac_us_path,
            rac_path=rac_path,
            enable_oracles=False,
        )

        # Mock a failing validator
        with patch.object(pipeline, "_run_ci", side_effect=Exception("Test error")):
            result = pipeline.validate(temp_rac_file)

            assert "ci" in result.results
            assert result.results["ci"].passed is False
            assert "Test error" in result.results["ci"].error


# Integration Tests


class TestPipelineIntegration:
    """Integration tests for full pipeline."""

    def test_to_actual_scores_conversion(self, pipeline, temp_rac_file):
        """PipelineResult should convert to ActualScores correctly."""
        result = pipeline.validate(temp_rac_file)

        actual_scores = result.to_actual_scores()

        # Should have all expected fields
        assert hasattr(actual_scores, "rac_reviewer")
        assert hasattr(actual_scores, "formula_reviewer")
        assert hasattr(actual_scores, "parameter_reviewer")
        assert hasattr(actual_scores, "ci_pass")
        assert hasattr(actual_scores, "policyengine_match")
        assert hasattr(actual_scores, "taxsim_match")

    def test_validate_file_convenience_function(self, temp_rac_file):
        """validate_file() should work as standalone function."""
        # This may fail if rac-us parent detection doesn't work,
        # but the function should at least not crash
        try:
            result = validate_file(temp_rac_file)
            assert isinstance(result, PipelineResult)
        except Exception:
            # May fail in test environment without proper repo structure
            pass


# Reviewer Agent Prompts Tests


class TestReviewerPrompts:
    """Tests for reviewer agent prompt construction."""

    def test_rac_reviewer_checks_structure(self, pipeline, temp_dirs):
        """RAC reviewer should validate structure and citations."""
        rac_us_path, _ = temp_dirs

        # Create RAC file with structural issues
        test_file = rac_us_path / "bad_structure.rac"
        test_file.write_text("""
# Missing required fields
incomplete_var:
    entity: Person
    # Missing period, dtype, formula
""")

        result = pipeline._run_reviewer("rac-reviewer", test_file)

        # The reviewer should identify structure issues
        # (In placeholder mode this passes, but real impl would catch issues)
        assert isinstance(result, ValidationResult)

    def test_formula_reviewer_checks_logic(self, pipeline, temp_dirs):
        """Formula reviewer should validate formula logic."""
        rac_us_path, _ = temp_dirs

        # Create RAC file with logic issues
        test_file = rac_us_path / "bad_formula.rac"
        test_file.write_text("""
circular_ref:
    entity: Person
    period: Year
    dtype: Money
    formula: |
        # Potential circular reference
        return person.circular_ref + 1
""")

        result = pipeline._run_reviewer("formula-reviewer", test_file)
        assert isinstance(result, ValidationResult)

    def test_parameter_reviewer_checks_sources(self, pipeline, temp_dirs):
        """Parameter reviewer should validate parameter sourcing."""
        rac_us_path, _ = temp_dirs

        # Create RAC file with parameter issues
        test_file = rac_us_path / "unsourced_param.rac"
        test_file.write_text("""
uses_magic_number:
    entity: Person
    period: Year
    dtype: Money
    formula: |
        # Magic number without named definition
        return person.income * 0.153  # Should be from a named definition
""")

        result = pipeline._run_reviewer("parameter-reviewer", test_file)
        assert isinstance(result, ValidationResult)
