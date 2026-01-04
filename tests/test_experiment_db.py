"""
Tests for the experiment database.
"""

import pytest
import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autorac import (
    ExperimentDB,
    EncodingRun,
    PredictedScores,
    FinalScores,
)
from autorac.harness.encoder_harness import create_run


class TestCreateRun:
    """Tests for the create_run factory function."""

    def test_create_run_generates_id(self):
        """Test that create_run generates a unique ID."""
        run = create_run(
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-5-20251101",
            rac_content="# content",
        )
        assert run.id is not None
        assert len(run.id) == 8  # UUID[:8]

    def test_create_run_sets_timestamp(self):
        """Test that create_run sets current timestamp."""
        before = datetime.now()
        run = create_run(
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-5-20251101",
            rac_content="# content",
        )
        after = datetime.now()
        assert before <= run.timestamp <= after

    def test_create_run_sets_iteration_1_for_new_run(self):
        """Test that new runs have iteration=1."""
        run = create_run(
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-5-20251101",
            rac_content="# content",
        )
        assert run.iteration == 1
        assert run.parent_run_id is None

    def test_create_run_sets_iteration_2_for_revision(self):
        """Test that revisions have iteration=2."""
        run = create_run(
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-5-20251101",
            rac_content="# content",
            parent_run_id="abc12345",
        )
        assert run.iteration == 2
        assert run.parent_run_id == "abc12345"


class TestExperimentDBInit:
    """Tests for ExperimentDB initialization."""

    def test_creates_database_file(self, temp_db_path):
        """Test that database file is created."""
        db = ExperimentDB(temp_db_path)
        assert temp_db_path.exists()

    def test_creates_tables(self, experiment_db, temp_db_path):
        """Test that required tables are created."""
        import sqlite3

        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()

        # Check encoding_runs table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='encoding_runs'"
        )
        assert cursor.fetchone() is not None

        # Check calibration_snapshots table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='calibration_snapshots'"
        )
        assert cursor.fetchone() is not None

        conn.close()


class TestLogAndRetrieveRuns:
    """Tests for logging and retrieving encoding runs."""

    def test_log_run_and_retrieve(self, experiment_db, sample_encoding_run):
        """Test logging a run and retrieving it."""
        run_id = experiment_db.log_run(sample_encoding_run)

        retrieved = experiment_db.get_run(run_id)

        assert retrieved is not None
        assert retrieved.id == sample_encoding_run.id
        assert retrieved.file_path == sample_encoding_run.file_path
        assert retrieved.citation == sample_encoding_run.citation
        assert retrieved.agent_type == sample_encoding_run.agent_type
        assert retrieved.agent_model == sample_encoding_run.agent_model
        assert retrieved.rac_content == sample_encoding_run.rac_content

    def test_log_run_with_predicted_scores(
        self, experiment_db, sample_predicted_scores
    ):
        """Test logging a run with predicted scores."""
        run = create_run(
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-5-20251101",
            rac_content="# content",
        )
        run.predicted = sample_predicted_scores

        experiment_db.log_run(run)
        retrieved = experiment_db.get_run(run.id)

        assert retrieved.predicted is not None
        assert retrieved.predicted.rac_reviewer == sample_predicted_scores.rac_reviewer
        assert retrieved.predicted.ci_pass == sample_predicted_scores.ci_pass
        assert retrieved.predicted.confidence == sample_predicted_scores.confidence

    def test_log_run_with_actual_scores(
        self, experiment_db, sample_actual_scores
    ):
        """Test logging a run with actual scores."""
        run = create_run(
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-5-20251101",
            rac_content="# content",
        )
        run.actual = sample_actual_scores

        experiment_db.log_run(run)
        retrieved = experiment_db.get_run(run.id)

        assert retrieved.actual is not None
        assert retrieved.actual.rac_reviewer == sample_actual_scores.rac_reviewer
        assert retrieved.actual.ci_pass == sample_actual_scores.ci_pass
        assert retrieved.actual.reviewer_issues == sample_actual_scores.reviewer_issues

    def test_log_run_with_suggestions(self, experiment_db):
        """Test logging a run with suggestions."""
        run = create_run(
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-5-20251101",
            rac_content="# content",
        )
        run.suggestions = [
            AgentSuggestion(
                category="documentation",
                description="Add example for brackets",
                predicted_impact="high",
                specific_change="Add bracket example to section 3",
            ),
            AgentSuggestion(
                category="validator",
                description="CI check too strict",
                predicted_impact="low",
            ),
        ]

        experiment_db.log_run(run)
        retrieved = experiment_db.get_run(run.id)

        assert len(retrieved.suggestions) == 2
        assert retrieved.suggestions[0].category == "documentation"
        assert retrieved.suggestions[0].predicted_impact == "high"
        assert retrieved.suggestions[1].category == "validator"

    def test_get_nonexistent_run_returns_none(self, experiment_db):
        """Test that getting a nonexistent run returns None."""
        result = experiment_db.get_run("nonexistent-id")
        assert result is None


class TestUpdateActualScores:
    """Tests for updating actual scores after validation."""

    def test_update_actual_scores(self, experiment_db, sample_actual_scores):
        """Test updating a run with actual scores."""
        run = create_run(
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-5-20251101",
            rac_content="# content",
        )
        experiment_db.log_run(run)

        # Initially no actual scores
        retrieved = experiment_db.get_run(run.id)
        assert retrieved.actual is None

        # Update with actual scores
        experiment_db.update_actual_scores(run.id, sample_actual_scores)

        # Now has actual scores
        retrieved = experiment_db.get_run(run.id)
        assert retrieved.actual is not None
        assert retrieved.actual.rac_reviewer == 8.0
        assert retrieved.actual.ci_pass is True
        assert retrieved.actual.policyengine_match == 0.88


class TestListRunsWithFilters:
    """Tests for listing runs with various filters."""

    def test_get_runs_for_citation(self, experiment_db):
        """Test getting all runs for a specific citation."""
        # Create runs for different citations
        for i in range(3):
            run = create_run(
                file_path=f"/path/to/file{i}.rac",
                citation="26 USC 32",
                agent_type="autorac:encoder",
                agent_model="claude-opus-4-5-20251101",
                rac_content=f"# content {i}",
            )
            experiment_db.log_run(run)

        run_other = create_run(
            file_path="/path/to/other.rac",
            citation="26 USC 24",  # Different citation
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-5-20251101",
            rac_content="# other content",
        )
        experiment_db.log_run(run_other)

        # Get runs for 26 USC 32
        runs = experiment_db.get_runs_for_citation("26 USC 32")
        assert len(runs) == 3
        assert all(r.citation == "26 USC 32" for r in runs)

        # Get runs for 26 USC 24
        runs = experiment_db.get_runs_for_citation("26 USC 24")
        assert len(runs) == 1
        assert runs[0].citation == "26 USC 24"

    def test_get_recent_runs(self, experiment_db):
        """Test getting most recent runs with limit."""
        # Create 5 runs
        for i in range(5):
            run = create_run(
                file_path=f"/path/to/file{i}.rac",
                citation=f"26 USC {i}",
                agent_type="autorac:encoder",
                agent_model="claude-opus-4-5-20251101",
                rac_content=f"# content {i}",
            )
            experiment_db.log_run(run)

        # Get last 3
        runs = experiment_db.get_recent_runs(limit=3)
        assert len(runs) == 3

        # Most recent should be first (DESC order)
        # Since timestamps are very close, just verify we get 3 runs

    def test_get_recent_runs_default_limit(self, experiment_db):
        """Test getting recent runs with default limit."""
        run = create_run(
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-5-20251101",
            rac_content="# content",
        )
        experiment_db.log_run(run)

        runs = experiment_db.get_recent_runs()
        assert len(runs) == 1


class TestCalibrationData:
    """Tests for getting calibration data."""

    def test_get_calibration_data_empty(self, experiment_db):
        """Test getting calibration data when no data exists."""
        data = experiment_db.get_calibration_data()
        assert data == []

    def test_get_calibration_data_skips_incomplete_runs(
        self, experiment_db, sample_predicted_scores
    ):
        """Test that runs without both predicted and actual scores are skipped."""
        # Run with only predicted scores
        run1 = create_run(
            file_path="/path/to/file1.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-5-20251101",
            rac_content="# content",
        )
        run1.predicted = sample_predicted_scores
        experiment_db.log_run(run1)

        # Run with no scores
        run2 = create_run(
            file_path="/path/to/file2.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-5-20251101",
            rac_content="# content",
        )
        experiment_db.log_run(run2)

        data = experiment_db.get_calibration_data()
        assert len(data) == 0

    def test_get_calibration_data_with_complete_runs(
        self, experiment_db, sample_encoding_run
    ):
        """Test getting calibration data with complete runs."""
        experiment_db.log_run(sample_encoding_run)

        data = experiment_db.get_calibration_data()
        assert len(data) == 1

        pred, actual = data[0]
        assert pred.rac_reviewer == 7.5
        assert actual.rac_reviewer == 8.0


class TestSessionLogging:
    """Tests for session logging (used by SDK orchestrator)."""

    def test_start_session_generates_id(self, experiment_db):
        """Test that start_session generates a unique ID."""
        session = experiment_db.start_session(model="test-model", cwd="/tmp")
        assert session.id is not None
        assert len(session.id) == 8

    def test_start_session_with_custom_id(self, experiment_db):
        """Test that start_session accepts custom session_id."""
        session = experiment_db.start_session(
            model="test-model",
            cwd="/tmp",
            session_id="custom-123"
        )
        assert session.id == "custom-123"

    def test_get_session_retrieves_by_id(self, experiment_db):
        """Test that get_session retrieves session by ID."""
        experiment_db.start_session(
            model="opus-4.5",
            cwd="/workspace",
            session_id="retrieve-test"
        )

        retrieved = experiment_db.get_session("retrieve-test")
        assert retrieved is not None
        assert retrieved.id == "retrieve-test"
        assert retrieved.model == "opus-4.5"
        assert retrieved.cwd == "/workspace"

    def test_get_session_returns_none_for_unknown(self, experiment_db):
        """Test that get_session returns None for unknown ID."""
        retrieved = experiment_db.get_session("nonexistent-id")
        assert retrieved is None

    def test_log_event_to_session(self, experiment_db):
        """Test logging events to a session."""
        experiment_db.start_session(session_id="event-test")

        event = experiment_db.log_event(
            session_id="event-test",
            event_type="agent_start",
            content="Test prompt",
            metadata={"agent_type": "encoder"}
        )

        assert event.sequence == 1
        assert event.event_type == "agent_start"

    def test_get_session_events(self, experiment_db):
        """Test retrieving all events for a session."""
        experiment_db.start_session(session_id="events-test")

        experiment_db.log_event(
            session_id="events-test",
            event_type="agent_start",
            content="Starting"
        )
        experiment_db.log_event(
            session_id="events-test",
            event_type="agent_end",
            content="Done"
        )

        events = experiment_db.get_session_events("events-test")
        assert len(events) == 2
        assert events[0].event_type == "agent_start"
        assert events[1].event_type == "agent_end"

    def test_session_event_count_updates(self, experiment_db):
        """Test that session event_count is tracked."""
        experiment_db.start_session(session_id="count-test")

        for i in range(3):
            experiment_db.log_event(
                session_id="count-test",
                event_type=f"event_{i}"
            )

        session = experiment_db.get_session("count-test")
        assert session.event_count == 3
