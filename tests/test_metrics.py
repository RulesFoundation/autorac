"""
Tests for calibration metrics.
"""

import sys
from datetime import datetime
from pathlib import Path

import pytest

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autorac import (
    ActualScores,
    CalibrationMetrics,
    CalibrationSnapshot,
    ExperimentDB,
    PredictedScores,
    compute_calibration,
    create_run,
    get_calibration_trend,
    print_calibration_report,
    save_calibration_snapshot,
)

# Private function import for testing internals
from autorac.harness.metrics import _compute_metric


class TestComputeMetric:
    """Tests for the _compute_metric helper function."""

    def test_empty_pairs_returns_zeros(self):
        """Test that empty pairs return zero metrics."""
        result = _compute_metric("test", [])
        assert result.n_samples == 0
        assert result.mse == 0
        assert result.mae == 0
        assert result.bias == 0

    def test_mse_calculation(self):
        """Test MSE (Mean Squared Error) calculation."""
        # Pairs: (predicted, actual)
        # Errors: (8-10)^2=4, (6-4)^2=4, (7-7)^2=0
        # MSE = (4+4+0)/3 = 2.6667
        pairs = [(8.0, 10.0), (6.0, 4.0), (7.0, 7.0)]
        result = _compute_metric("test", pairs)
        assert result.mse == pytest.approx(2.6667, rel=0.01)

    def test_mae_calculation(self):
        """Test MAE (Mean Absolute Error) calculation."""
        # Errors: |8-10|=2, |6-4|=2, |7-7|=0
        # MAE = (2+2+0)/3 = 1.333
        pairs = [(8.0, 10.0), (6.0, 4.0), (7.0, 7.0)]
        result = _compute_metric("test", pairs)
        assert result.mae == pytest.approx(1.3333, rel=0.01)

    def test_bias_calculation_overconfident(self):
        """Test bias when agent is overconfident (predicts higher than actual)."""
        # Predicted mean = 8, Actual mean = 6
        # Bias = 8 - 6 = 2 (positive = overconfident)
        pairs = [(8.0, 6.0), (8.0, 6.0), (8.0, 6.0)]
        result = _compute_metric("test", pairs)
        assert result.bias == pytest.approx(2.0, rel=0.01)

    def test_bias_calculation_underconfident(self):
        """Test bias when agent is underconfident (predicts lower than actual)."""
        # Predicted mean = 5, Actual mean = 8
        # Bias = 5 - 8 = -3 (negative = underconfident)
        pairs = [(5.0, 8.0), (5.0, 8.0), (5.0, 8.0)]
        result = _compute_metric("test", pairs)
        assert result.bias == pytest.approx(-3.0, rel=0.01)

    def test_bias_calculation_zero(self):
        """Test bias when predictions match actual."""
        pairs = [(7.0, 7.0), (8.0, 8.0), (6.0, 6.0)]
        result = _compute_metric("test", pairs)
        assert result.bias == pytest.approx(0.0, abs=0.001)

    def test_correlation_perfect_positive(self):
        """Test perfect positive correlation."""
        pairs = [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        result = _compute_metric("test", pairs)
        assert result.correlation == pytest.approx(1.0, abs=0.01)

    def test_correlation_perfect_negative(self):
        """Test perfect negative correlation."""
        pairs = [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]
        result = _compute_metric("test", pairs)
        assert result.correlation == pytest.approx(-1.0, abs=0.01)

    def test_correlation_requires_3_samples(self):
        """Test that correlation requires at least 3 samples."""
        pairs = [(1.0, 1.0), (2.0, 2.0)]
        result = _compute_metric("test", pairs)
        assert result.correlation is None

    def test_correlation_none_for_constant_values(self):
        """Test that correlation is None when all values are constant."""
        pairs = [(5.0, 5.0), (5.0, 5.0), (5.0, 5.0)]
        result = _compute_metric("test", pairs)
        assert result.correlation is None


class TestComputeCalibration:
    """Tests for the compute_calibration function."""

    def test_empty_database_returns_empty_snapshot(self, experiment_db):
        """Test that empty database returns empty snapshot."""
        snapshot = compute_calibration(experiment_db)
        assert snapshot.total_runs == 0
        assert snapshot.pass_rate == 0.0
        assert snapshot.metrics == {}

    def test_compute_calibration_with_data(self, experiment_db):
        """Test calibration computation with sample data."""
        # Create runs with both predicted and actual scores
        for i in range(15):
            run = create_run(
                file_path=f"/path/to/file{i}.rac",
                citation="26 USC 32",
                agent_type="autorac:encoder",
                agent_model="claude-opus-4-5-20251101",
                rac_content="# content",
            )
            run.predicted = PredictedScores(
                rac_reviewer=7.0 + (i % 3),
                formula_reviewer=7.0,
                parameter_reviewer=8.0,
                integration_reviewer=7.5,
                ci_pass=True,
                policyengine_match=0.90,
                taxsim_match=0.85,
                confidence=0.6,
            )
            run.actual = ActualScores(
                rac_reviewer=7.5 + (i % 2),
                formula_reviewer=7.0,
                parameter_reviewer=7.5,
                integration_reviewer=8.0,
                ci_pass=True,
                policyengine_match=0.88,
                taxsim_match=0.82,
            )
            experiment_db.log_run(run)

        snapshot = compute_calibration(experiment_db, min_samples=10)

        assert snapshot.total_runs == 15
        assert "rac_reviewer" in snapshot.metrics
        assert snapshot.metrics["rac_reviewer"].n_samples == 15

    def test_min_samples_filter(self, experiment_db):
        """Test that metrics with too few samples are excluded."""
        # Create only 5 runs (less than default min_samples=10)
        for i in range(5):
            run = create_run(
                file_path=f"/path/to/file{i}.rac",
                citation="26 USC 32",
                agent_type="autorac:encoder",
                agent_model="claude-opus-4-5-20251101",
                rac_content="# content",
            )
            run.predicted = PredictedScores(
                rac_reviewer=7.5,
                formula_reviewer=7.0,
                parameter_reviewer=8.0,
                integration_reviewer=7.5,
                ci_pass=True,
                policyengine_match=0.90,
                taxsim_match=0.85,
                confidence=0.6,
            )
            run.actual = ActualScores(
                rac_reviewer=8.0,
                formula_reviewer=7.0,
                parameter_reviewer=7.5,
                integration_reviewer=8.0,
                ci_pass=True,
            )
            experiment_db.log_run(run)

        snapshot = compute_calibration(experiment_db, min_samples=10)
        assert "rac_reviewer" not in snapshot.metrics

        # With lower threshold, metrics should appear
        snapshot = compute_calibration(experiment_db, min_samples=3)
        assert "rac_reviewer" in snapshot.metrics

    def test_pass_rate_calculation(self, experiment_db):
        """Test pass rate is calculated correctly."""
        # Create 4 runs, 3 passing
        for i in range(4):
            run = create_run(
                file_path=f"/path/to/file{i}.rac",
                citation="26 USC 32",
                agent_type="autorac:encoder",
                agent_model="claude-opus-4-5-20251101",
                rac_content="# content",
            )
            run.predicted = PredictedScores(
                rac_reviewer=7.5,
                formula_reviewer=7.0,
                parameter_reviewer=8.0,
                integration_reviewer=7.5,
                ci_pass=True,
                confidence=0.6,
            )
            run.actual = ActualScores(
                rac_reviewer=8.0,
                formula_reviewer=7.0,
                parameter_reviewer=7.5,
                integration_reviewer=8.0,
                ci_pass=(i != 3),  # Last one fails
                ci_error="Parse error" if i == 3 else None,
            )
            experiment_db.log_run(run)

        snapshot = compute_calibration(experiment_db, min_samples=1)
        assert snapshot.pass_rate == pytest.approx(0.75, rel=0.01)


class TestCalibrationReport:
    """Tests for the calibration report generation."""

    def test_print_calibration_report_empty(self):
        """Test report generation with no data."""
        snapshot = CalibrationSnapshot(
            timestamp=datetime.now(),
            metrics={},
            total_runs=0,
            pass_rate=0.0,
        )
        report = print_calibration_report(snapshot)
        assert "CALIBRATION REPORT" in report
        assert "Total Runs: 0" in report
        assert "No calibration data available yet" in report

    def test_print_calibration_report_with_data(self):
        """Test report generation with metrics."""
        snapshot = CalibrationSnapshot(
            timestamp=datetime.now(),
            metrics={
                "rac_reviewer": CalibrationMetrics(
                    metric_name="rac_reviewer",
                    n_samples=50,
                    predicted_mean=7.5,
                    actual_mean=8.0,
                    mse=0.25,
                    mae=0.5,
                    bias=-0.5,
                    correlation=0.85,
                ),
            },
            total_runs=50,
            pass_rate=0.85,
        )
        report = print_calibration_report(snapshot)
        assert "Total Runs: 50" in report
        assert "Pass Rate: 85.0%" in report
        assert "rac_reviewer" in report
        assert "Bias > 0: Agent overconfident" in report


class TestCalibrationTrend:
    """Tests for calibration trend analysis."""

    def test_save_and_get_calibration_trend(self, temp_db_path, experiment_db):
        """Test saving and retrieving calibration trends."""
        # First need to initialize the calibration_snapshots table
        # by creating the experiment db
        ExperimentDB(temp_db_path)

        # Create a snapshot and save it
        snapshot = CalibrationSnapshot(
            timestamp=datetime.now(),
            metrics={
                "rac_reviewer": CalibrationMetrics(
                    metric_name="rac_reviewer",
                    n_samples=50,
                    predicted_mean=7.5,
                    actual_mean=8.0,
                    mse=0.25,
                    mae=0.5,
                    bias=-0.5,
                ),
            },
            total_runs=50,
            pass_rate=0.85,
        )
        save_calibration_snapshot(temp_db_path, snapshot)

        # Retrieve the trend
        trend = get_calibration_trend(temp_db_path, "rac_reviewer")
        assert len(trend) == 1
        ts, pred, actual = trend[0]
        assert pred == pytest.approx(7.5, rel=0.01)
        assert actual == pytest.approx(8.0, rel=0.01)

    def test_get_calibration_trend_limit(self, temp_db_path, experiment_db):
        """Test that trend limit works correctly."""
        import time

        ExperimentDB(temp_db_path)

        # Create multiple snapshots
        for i in range(5):
            snapshot = CalibrationSnapshot(
                timestamp=datetime.now(),
                metrics={
                    "rac_reviewer": CalibrationMetrics(
                        metric_name="rac_reviewer",
                        n_samples=50 + i,
                        predicted_mean=7.0 + i * 0.1,
                        actual_mean=8.0 + i * 0.1,
                        mse=0.25,
                        mae=0.5,
                        bias=-0.5,
                    ),
                },
                total_runs=50 + i,
                pass_rate=0.85,
            )
            save_calibration_snapshot(temp_db_path, snapshot)
            time.sleep(0.01)  # Ensure different timestamps

        # Get only last 3
        trend = get_calibration_trend(temp_db_path, "rac_reviewer", limit=3)
        assert len(trend) == 3

    def test_get_calibration_trend_nonexistent_metric(
        self, temp_db_path, experiment_db
    ):
        """Test getting trend for non-existent metric."""
        trend = get_calibration_trend(temp_db_path, "nonexistent_metric")
        assert trend == []


class TestSamplePredictedVsActualData:
    """Integration tests with realistic predicted vs actual data scenarios."""

    def test_well_calibrated_agent(self, experiment_db):
        """Test metrics for a well-calibrated agent."""
        # Create runs where predictions closely match actuals
        for i in range(20):
            run = create_run(
                file_path=f"/path/to/file{i}.rac",
                citation="26 USC 32",
                agent_type="autorac:encoder",
                agent_model="claude-opus-4-5-20251101",
                rac_content="# content",
            )
            # Small random-ish variation
            base = 7.5 + (i % 3) * 0.5
            run.predicted = PredictedScores(
                rac_reviewer=base,
                formula_reviewer=7.0,
                parameter_reviewer=8.0,
                integration_reviewer=7.5,
                ci_pass=True,
                confidence=0.8,
            )
            run.actual = ActualScores(
                rac_reviewer=base + 0.2,  # Small difference
                formula_reviewer=7.1,
                parameter_reviewer=7.9,
                integration_reviewer=7.6,
                ci_pass=True,
            )
            experiment_db.log_run(run)

        snapshot = compute_calibration(experiment_db, min_samples=10)
        assert snapshot.metrics["rac_reviewer"].mse < 0.5  # Low MSE
        assert abs(snapshot.metrics["rac_reviewer"].bias) < 0.5  # Low bias

    def test_overconfident_agent(self, experiment_db):
        """Test metrics for an overconfident agent."""
        for i in range(20):
            run = create_run(
                file_path=f"/path/to/file{i}.rac",
                citation="26 USC 32",
                agent_type="autorac:encoder",
                agent_model="claude-opus-4-5-20251101",
                rac_content="# content",
            )
            run.predicted = PredictedScores(
                rac_reviewer=9.0,  # Always predicts high
                formula_reviewer=9.0,
                parameter_reviewer=9.0,
                integration_reviewer=9.0,
                ci_pass=True,
                confidence=0.95,
            )
            run.actual = ActualScores(
                rac_reviewer=6.5,  # Actually lower
                formula_reviewer=6.0,
                parameter_reviewer=6.5,
                integration_reviewer=6.0,
                ci_pass=True,
            )
            experiment_db.log_run(run)

        snapshot = compute_calibration(experiment_db, min_samples=10)
        assert snapshot.metrics["rac_reviewer"].bias > 2.0  # Positive bias
        assert snapshot.metrics["rac_reviewer"].mse > 5.0  # High MSE

    def test_underconfident_agent(self, experiment_db):
        """Test metrics for an underconfident agent."""
        for i in range(20):
            run = create_run(
                file_path=f"/path/to/file{i}.rac",
                citation="26 USC 32",
                agent_type="autorac:encoder",
                agent_model="claude-opus-4-5-20251101",
                rac_content="# content",
            )
            run.predicted = PredictedScores(
                rac_reviewer=5.0,  # Always predicts low
                formula_reviewer=5.0,
                parameter_reviewer=5.0,
                integration_reviewer=5.0,
                ci_pass=True,
                confidence=0.3,
            )
            run.actual = ActualScores(
                rac_reviewer=8.5,  # Actually higher
                formula_reviewer=8.0,
                parameter_reviewer=8.5,
                integration_reviewer=8.0,
                ci_pass=True,
            )
            experiment_db.log_run(run)

        snapshot = compute_calibration(experiment_db, min_samples=10)
        assert snapshot.metrics["rac_reviewer"].bias < -3.0  # Negative bias
