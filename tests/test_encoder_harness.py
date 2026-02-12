"""
Tests for encoder harness with Claude Code CLI integration.

These tests verify:
1. _get_predictions() correctly calls Claude CLI and parses JSON response
2. _encode() generates valid RAC content via Claude CLI
3. _get_suggestions() analyzes validation failures and returns improvements
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from autorac.harness.encoder_harness import (
    EncoderConfig,
    EncoderHarness,
    run_claude_code,
)
from autorac.harness.validator_pipeline import PipelineResult, ValidationResult


@pytest.fixture
def temp_config():
    """Create a temporary encoder configuration."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        (tmpdir_path / "rac-us").mkdir()
        (tmpdir_path / "rac").mkdir()
        yield EncoderConfig(
            rac_us_path=tmpdir_path / "rac-us",
            rac_path=tmpdir_path / "rac",
            db_path=tmpdir_path / "experiments.db",
            enable_oracles=False,  # Disable oracles for faster tests
        )


class TestRunClaudeCode:
    """Tests for run_claude_code function."""

    def test_returns_output_and_returncode(self):
        """Test that function returns tuple of (output, returncode)."""
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="test output", stderr="", returncode=0)

            output, code = run_claude_code("test prompt")

            assert "test output" in output
            assert code == 0

    def test_handles_timeout(self):
        """Test timeout handling."""
        import subprocess

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=60)

            output, code = run_claude_code("test", timeout=60)

            assert "Timeout" in output
            assert code == 1

    def test_handles_missing_cli(self):
        """Test handling when claude CLI is not installed."""
        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            output, code = run_claude_code("test")

            assert "not found" in output
            assert code == 1


class TestGetPredictions:
    """Tests for _get_predictions method."""

    def test_parses_valid_json_response(self, temp_config):
        """Test that valid JSON predictions are correctly parsed."""
        harness = EncoderHarness(temp_config)

        prediction_json = json.dumps(
            {
                "rac_reviewer": 8.5,
                "formula_reviewer": 7.0,
                "parameter_reviewer": 8.0,
                "integration_reviewer": 7.5,
                "ci_pass": True,
                "policyengine_match": 0.92,
                "taxsim_match": 0.88,
                "confidence": 0.75,
                "reasoning": "Simple statute with clear calculation",
            }
        )

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_claude:
            mock_claude.return_value = (prediction_json, 0)

            result = harness._get_predictions("26 USC 32", "Sample statute")

            assert result.rac_reviewer == 8.5
            assert result.formula_reviewer == 7.0
            assert result.ci_pass is True
            assert result.confidence == 0.75

    def test_handles_cli_failure_gracefully(self, temp_config):
        """Test that CLI failures return conservative defaults."""
        harness = EncoderHarness(temp_config)

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_claude:
            mock_claude.return_value = ("Error: something went wrong", 1)

            result = harness._get_predictions("26 USC 32", "Sample statute")

            # Should return conservative defaults
            assert result.rac_reviewer == 6.0
            assert result.ci_pass is False
            assert result.confidence == 0.3

    def test_extracts_json_from_wrapped_response(self, temp_config):
        """Test that JSON is extracted even when wrapped in text."""
        harness = EncoderHarness(temp_config)

        # Response with JSON wrapped in text
        wrapped_response = """Based on my analysis, here are my predictions:

{"rac_reviewer": 7.5, "formula_reviewer": 6.5, "parameter_reviewer": 7.0, "integration_reviewer": 7.0, "ci_pass": true, "policyengine_match": 0.85, "taxsim_match": 0.80, "confidence": 0.6}

These scores reflect the moderate complexity of this statute."""

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_claude:
            mock_claude.return_value = (wrapped_response, 0)

            result = harness._get_predictions("26 USC 24", "Child tax credit text")

            assert result.rac_reviewer == 7.5
            assert result.formula_reviewer == 6.5


class TestEncode:
    """Tests for _encode method."""

    def test_generates_valid_rac_content(self, temp_config):
        """Test that valid RAC content is generated and saved."""
        harness = EncoderHarness(temp_config)

        expected_rac = '''"""
Sample statute text
"""

sample_var:
  entity: TaxUnit
  period: Year
  dtype: Money
  label: "Sample Variable"
  formula: |
    return 0
  default: 0
'''

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_claude:
            mock_claude.return_value = (expected_rac, 0)

            output_path = temp_config.rac_us_path / "test.rac"
            result = harness._encode("26 USC 1", "Sample statute text", output_path)

            assert '"""' in result
            assert "sample_var:" in result
            assert output_path.exists()

    def test_strips_markdown_code_blocks(self, temp_config):
        """Test that markdown code blocks are stripped from response."""
        harness = EncoderHarness(temp_config)

        response_with_markdown = '''```yaml
"""
Statute text
"""

test:
  dtype: Money
```'''

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_claude:
            mock_claude.return_value = (response_with_markdown, 0)

            output_path = temp_config.rac_us_path / "test.rac"
            result = harness._encode("26 USC 1", "Statute text", output_path)

            assert not result.startswith("```")
            assert not result.endswith("```")
            assert '"""' in result

    def test_creates_output_directory(self, temp_config):
        """Test that output directory is created if it doesn't exist."""
        harness = EncoderHarness(temp_config)

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_claude:
            mock_claude.return_value = ("text: test", 0)

            output_path = temp_config.rac_us_path / "nested" / "dir" / "test.rac"
            harness._encode("26 USC 1", "Statute text", output_path)

            assert output_path.parent.exists()

    def test_returns_fallback_on_cli_failure(self, temp_config):
        """Test that a valid fallback is returned on CLI failure."""
        harness = EncoderHarness(temp_config)

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_claude:
            mock_claude.side_effect = Exception("CLI error")

            output_path = temp_config.rac_us_path / "fallback.rac"
            result = harness._encode("26 USC 32", "EITC statute", output_path)

            assert '"""' in result
            assert "TODO: Implement formula" in result
            assert output_path.exists()


class TestGetSuggestions:
    """Tests for _get_suggestions method."""

    def test_returns_empty_list_when_all_passed(self, temp_config):
        """Test that no suggestions are generated when all validators pass."""
        harness = EncoderHarness(temp_config)

        passing_result = PipelineResult(
            results={
                "ci": ValidationResult("ci", True, None, [], 100),
                "rac_reviewer": ValidationResult("rac_reviewer", True, 8.0, [], 500),
            },
            total_duration_ms=600,
            all_passed=True,
        )

        result = harness._get_suggestions("26 USC 32", "rac content", passing_result)

        assert result == []

    def test_parses_valid_suggestions_json(self, temp_config):
        """Test that valid suggestions JSON is correctly parsed."""
        harness = EncoderHarness(temp_config)

        failing_result = PipelineResult(
            results={
                "ci": ValidationResult(
                    "ci", False, None, ["Parse error"], 100, error="Parse error"
                ),
            },
            total_duration_ms=100,
            all_passed=False,
        )

        suggestions_json = json.dumps(
            [
                {
                    "category": "documentation",
                    "description": "Add clearer examples for formula syntax",
                    "predicted_impact": "high",
                    "specific_change": "Add section on conditionals",
                },
                {
                    "category": "agent_prompt",
                    "description": "Emphasize no markdown formatting",
                    "predicted_impact": "medium",
                    "specific_change": None,
                },
            ]
        )

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_claude:
            mock_claude.return_value = (suggestions_json, 0)

            result = harness._get_suggestions(
                "26 USC 32", "rac content", failing_result
            )

            assert len(result) == 2
            assert result[0].category == "documentation"
            assert result[0].predicted_impact == "high"
            assert result[1].category == "agent_prompt"

    def test_handles_cli_failure_gracefully(self, temp_config):
        """Test that CLI failures return basic suggestions based on failures."""
        harness = EncoderHarness(temp_config)

        failing_result = PipelineResult(
            results={
                "ci": ValidationResult(
                    "ci", False, None, ["Parse error"], 100, error="Parse error"
                ),
                "rac_reviewer": ValidationResult(
                    "rac_reviewer",
                    False,
                    4.0,
                    ["Missing imports"],
                    500,
                    error="Missing imports",
                ),
            },
            total_duration_ms=600,
            all_passed=False,
        )

        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_claude:
            mock_claude.side_effect = Exception("CLI error")

            result = harness._get_suggestions(
                "26 USC 32", "rac content", failing_result
            )

            assert len(result) == 2
            assert all(s.category == "validator" for s in result)


class TestEncodeWithFeedback:
    """Integration tests for encode_with_feedback method."""

    def test_full_encode_cycle(self, temp_config):
        """Test the full encode-validate-log cycle."""
        harness = EncoderHarness(temp_config)

        prediction_json = json.dumps(
            {
                "rac_reviewer": 8.0,
                "formula_reviewer": 7.5,
                "parameter_reviewer": 8.0,
                "integration_reviewer": 7.5,
                "ci_pass": True,
                "confidence": 0.7,
            }
        )

        rac_content = '''"""
Test statute
"""

test:
  entity: TaxUnit
  dtype: Money
  period: Year
  formula: |
    return 0
'''

        # Mock run_claude_code for both encoder and validator
        with patch("autorac.harness.encoder_harness.run_claude_code") as mock_encoder:
            with patch(
                "autorac.harness.validator_pipeline.run_claude_code"
            ) as mock_validator:
                # Encoder returns prediction then RAC content
                mock_encoder.side_effect = [
                    (prediction_json, 0),  # _get_predictions
                    (rac_content, 0),  # _encode
                ]

                # Validators return passing scores
                mock_validator.return_value = (
                    '{"score": 8.0, "passed": true, "issues": [], "reasoning": "Good"}',
                    0,
                )

                output_path = temp_config.rac_us_path / "statute" / "26" / "1.rac"

                run, result = harness.encode_with_feedback(
                    citation="26 USC 1",
                    statute_text="Test statute text",
                    output_path=output_path,
                )

                assert run.citation == "26 USC 1"
                assert run.predicted is not None
                assert run.actual is not None
                assert output_path.exists()
