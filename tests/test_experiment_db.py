"""
Tests for the experiment database.
"""

import sys
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autorac import (
    AgentSuggestion,
    ExperimentDB,
    create_run,
)


class TestCreateRun:
    """Tests for the create_run factory function."""

    def test_create_run_generates_id(self):
        """Test that create_run generates a unique ID."""
        run = create_run(
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-6",
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
            agent_model="claude-opus-4-6",
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
            agent_model="claude-opus-4-6",
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
            agent_model="claude-opus-4-6",
            rac_content="# content",
            parent_run_id="abc12345",
        )
        assert run.iteration == 2
        assert run.parent_run_id == "abc12345"


class TestExperimentDBInit:
    """Tests for ExperimentDB initialization."""

    def test_creates_database_file(self, temp_db_path):
        """Test that database file is created."""
        ExperimentDB(temp_db_path)
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
            agent_model="claude-opus-4-6",
            rac_content="# content",
        )
        run.predicted = sample_predicted_scores

        experiment_db.log_run(run)
        retrieved = experiment_db.get_run(run.id)

        assert retrieved.predicted is not None
        assert retrieved.predicted.rac_reviewer == sample_predicted_scores.rac_reviewer
        assert retrieved.predicted.ci_pass == sample_predicted_scores.ci_pass
        assert retrieved.predicted.confidence == sample_predicted_scores.confidence

    def test_log_run_with_actual_scores(self, experiment_db, sample_actual_scores):
        """Test logging a run with actual scores."""
        run = create_run(
            file_path="/path/to/file.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-6",
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
            agent_model="claude-opus-4-6",
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
            agent_model="claude-opus-4-6",
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
                agent_model="claude-opus-4-6",
                rac_content=f"# content {i}",
            )
            experiment_db.log_run(run)

        run_other = create_run(
            file_path="/path/to/other.rac",
            citation="26 USC 24",  # Different citation
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-6",
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
                agent_model="claude-opus-4-6",
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
            agent_model="claude-opus-4-6",
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
            agent_model="claude-opus-4-6",
            rac_content="# content",
        )
        run1.predicted = sample_predicted_scores
        experiment_db.log_run(run1)

        # Run with no scores
        run2 = create_run(
            file_path="/path/to/file2.rac",
            citation="26 USC 32",
            agent_type="autorac:encoder",
            agent_model="claude-opus-4-6",
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
            model="test-model", cwd="/tmp", session_id="custom-123"
        )
        assert session.id == "custom-123"

    def test_get_session_retrieves_by_id(self, experiment_db):
        """Test that get_session retrieves session by ID."""
        experiment_db.start_session(
            model="opus-4.5", cwd="/workspace", session_id="retrieve-test"
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
            metadata={"agent_type": "encoder"},
        )

        assert event.sequence == 1
        assert event.event_type == "agent_start"

    def test_get_session_events(self, experiment_db):
        """Test retrieving all events for a session."""
        experiment_db.start_session(session_id="events-test")

        experiment_db.log_event(
            session_id="events-test", event_type="agent_start", content="Starting"
        )
        experiment_db.log_event(
            session_id="events-test", event_type="agent_end", content="Done"
        )

        events = experiment_db.get_session_events("events-test")
        assert len(events) == 2
        assert events[0].event_type == "agent_start"
        assert events[1].event_type == "agent_end"

    def test_session_event_count_updates(self, experiment_db):
        """Test that session event_count is tracked."""
        experiment_db.start_session(session_id="count-test")

        for i in range(3):
            experiment_db.log_event(session_id="count-test", event_type=f"event_{i}")

        session = experiment_db.get_session("count-test")
        assert session.event_count == 3


class TestRowToRunSchemaVersions:
    """Tests for _row_to_run with different schema versions."""

    def test_row_with_11_columns(self, experiment_db):
        """Test _row_to_run with legacy 11-column schema."""
        import sqlite3

        row = (
            "test-id",  # id
            "2024-01-01T00:00:00",  # timestamp
            "26 USC 32",  # citation
            "/path/file.rac",  # file_path
            "{}",  # complexity_json
            "[]",  # iterations_json
            1000,  # total_duration_ms
            None,  # final_scores_json
            "encoder",  # agent_type
            "opus",  # agent_model
            "content",  # rac_content
        )
        run = experiment_db._row_to_run(row)
        assert run.id == "test-id"
        assert run.citation == "26 USC 32"
        assert run.session_id is None
        assert run.iteration == 1

    def test_row_with_13_columns(self, experiment_db):
        """Test _row_to_run with 13-column schema (added predicted_scores, session_id)."""
        import json

        predicted = json.dumps(
            {
                "rac_reviewer": 8.0,
                "formula_reviewer": 7.0,
                "parameter_reviewer": 7.5,
                "integration_reviewer": 7.0,
                "ci_pass": True,
                "confidence": 0.6,
            }
        )
        row = (
            "test-id-13",
            "2024-01-01T00:00:00",
            "26 USC 24",
            "/path/file.rac",
            "{}",
            "[]",
            2000,
            None,
            "encoder",
            "opus",
            "content",
            predicted,  # predicted_scores_json
            "sess-123",  # session_id
        )
        run = experiment_db._row_to_run(row)
        assert run.id == "test-id-13"
        assert run.session_id == "sess-123"
        assert run.predicted is not None
        assert run.predicted.rac_reviewer == 8.0
        assert run.iteration == 1


class TestPredictedScoresSetter:
    """Test the predicted_scores property setter."""

    def test_predicted_scores_setter(self):
        """Test EncodingRun.predicted_scores setter sets .predicted."""
        from autorac import EncodingRun, PredictedScores

        run = EncodingRun(
            file_path="/path.rac",
            citation="26 USC 1",
            agent_type="encoder",
            agent_model="opus",
            rac_content="content",
        )
        scores = PredictedScores(rac_reviewer=8.0, confidence=0.7)
        run.predicted_scores = scores
        assert run.predicted is scores
        assert run.predicted_scores is scores


class TestMigration:
    """Tests for database migration (old 'runs' table to 'encoding_runs')."""

    def test_migrate_runs_table(self, temp_db_path):
        """Test that old 'runs' table is renamed to 'encoding_runs'."""
        import sqlite3

        # Create old-style DB with 'runs' table
        conn = sqlite3.connect(temp_db_path)
        conn.execute("""
            CREATE TABLE runs (
                id TEXT PRIMARY KEY,
                timestamp TEXT,
                citation TEXT,
                file_path TEXT,
                complexity_json TEXT,
                iterations_json TEXT,
                total_duration_ms INTEGER,
                final_scores_json TEXT,
                agent_type TEXT,
                agent_model TEXT,
                rac_content TEXT
            )
        """)
        conn.execute("""
            INSERT INTO runs VALUES (
                'old-run', '2024-01-01T00:00:00', '26 USC 1',
                '/path.rac', '{}', '[]', 1000, NULL, 'encoder', 'opus', 'content'
            )
        """)
        conn.commit()
        conn.close()

        # Now init ExperimentDB — should migrate
        db = ExperimentDB(temp_db_path)

        # Old table should be gone
        conn = sqlite3.connect(temp_db_path)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
        )
        assert cursor.fetchone() is None

        # encoding_runs should have the data
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='encoding_runs'"
        )
        assert cursor.fetchone() is not None

        cursor.execute("SELECT id FROM encoding_runs")
        assert cursor.fetchone()[0] == "old-run"
        conn.close()


class TestArtifactVersioning:
    """Tests for SCD2 artifact versioning."""

    def test_log_artifact_version_new(self, experiment_db):
        """Test creating a new artifact version."""
        version = experiment_db.log_artifact_version(
            artifact_type="plugin",
            content="plugin content v1",
            version_label="1.0.0",
            metadata={"files": ["a.md"]},
        )
        assert version.artifact_type == "plugin"
        assert version.content == "plugin content v1"
        assert version.version_label == "1.0.0"
        assert version.effective_to is None

    def test_log_artifact_version_duplicate_returns_existing(self, experiment_db):
        """Test that duplicate content returns existing version."""
        v1 = experiment_db.log_artifact_version(
            artifact_type="plugin",
            content="same content",
            version_label="1.0.0",
        )
        v2 = experiment_db.log_artifact_version(
            artifact_type="plugin",
            content="same content",
            version_label="1.0.1",
        )
        assert v1.id == v2.id  # Same version returned

    def test_log_artifact_version_new_content_closes_old(self, experiment_db):
        """Test that new content closes old version."""
        v1 = experiment_db.log_artifact_version(
            artifact_type="plugin",
            content="content v1",
            version_label="1.0.0",
        )
        v2 = experiment_db.log_artifact_version(
            artifact_type="plugin",
            content="content v2",
            version_label="2.0.0",
        )

        # v1 and v2 should be different
        assert v1.id != v2.id

        # v1 should be closed (effective_to set)
        history = experiment_db.get_artifact_history("plugin")
        assert len(history) == 2
        # Most recent first
        assert history[0].id == v2.id
        assert history[0].effective_to is None
        assert history[1].id == v1.id
        assert history[1].effective_to is not None

    def test_get_current_artifact_version(self, experiment_db):
        """Test getting the current artifact version."""
        experiment_db.log_artifact_version(
            artifact_type="rac_spec",
            content="spec v1",
        )
        experiment_db.log_artifact_version(
            artifact_type="rac_spec",
            content="spec v2",
        )

        current = experiment_db.get_current_artifact_version("rac_spec")
        assert current is not None
        assert current.content == "spec v2"
        assert current.effective_to is None

    def test_get_current_artifact_version_none(self, experiment_db):
        """Test returns None when no artifact exists."""
        result = experiment_db.get_current_artifact_version("nonexistent")
        assert result is None

    def test_get_artifact_history(self, experiment_db):
        """Test getting full artifact history."""
        experiment_db.log_artifact_version(
            artifact_type="plugin",
            content="v1",
            version_label="1.0",
        )
        experiment_db.log_artifact_version(
            artifact_type="plugin",
            content="v2",
            version_label="2.0",
        )
        experiment_db.log_artifact_version(
            artifact_type="plugin",
            content="v3",
            version_label="3.0",
        )

        history = experiment_db.get_artifact_history("plugin")
        assert len(history) == 3
        # Ordered by effective_from DESC
        assert history[0].version_label == "3.0"
        assert history[2].version_label == "1.0"

    def test_link_run_to_artifacts(self, experiment_db, sample_encoding_run):
        """Test linking a run to artifact versions."""
        experiment_db.log_run(sample_encoding_run)

        v1 = experiment_db.log_artifact_version(
            artifact_type="plugin", content="plugin code"
        )
        v2 = experiment_db.log_artifact_version(
            artifact_type="rac_spec", content="spec code"
        )

        experiment_db.link_run_to_artifacts(
            sample_encoding_run.id, [v1.id, v2.id]
        )

        artifacts = experiment_db.get_run_artifacts(sample_encoding_run.id)
        assert len(artifacts) == 2
        artifact_types = {a.artifact_type for a in artifacts}
        assert "plugin" in artifact_types
        assert "rac_spec" in artifact_types

    def test_link_run_to_current_artifacts(self, experiment_db, sample_encoding_run):
        """Test linking a run to all current artifact versions."""
        experiment_db.log_run(sample_encoding_run)

        experiment_db.log_artifact_version(
            artifact_type="plugin", content="plugin"
        )
        experiment_db.log_artifact_version(
            artifact_type="rac_spec", content="spec"
        )

        ids = experiment_db.link_run_to_current_artifacts(sample_encoding_run.id)
        assert len(ids) == 2

        artifacts = experiment_db.get_run_artifacts(sample_encoding_run.id)
        assert len(artifacts) == 2


class TestUpdateSessionTokens:
    """Tests for updating session tokens."""

    def test_update_session_tokens(self, experiment_db):
        """Test updating token usage for a session."""
        experiment_db.start_session(
            model="test-model", cwd="/tmp", session_id="token-test"
        )

        experiment_db.update_session_tokens(
            session_id="token-test",
            input_tokens=1000,
            output_tokens=500,
            cache_read_tokens=200,
        )

        session = experiment_db.get_session("token-test")
        # Session.total_tokens = input_tokens + output_tokens = 1500
        assert session.total_tokens == 1500


class TestSnapshotPlugin:
    """Tests for snapshot_plugin method."""

    def test_snapshot_plugin(self, experiment_db, tmp_path):
        """Test snapshotting a plugin directory."""
        plugin_dir = tmp_path / "test-plugin"
        plugin_dir.mkdir()

        # Create plugin structure
        (plugin_dir / "plugin.json").write_text('{"name": "test"}')
        agents_dir = plugin_dir / "agents"
        agents_dir.mkdir()
        (agents_dir / "encoder.md").write_text("# Encoder agent")
        skills_dir = plugin_dir / "skills"
        skills_dir.mkdir()
        (skills_dir / "encode.md").write_text("# Encode skill")

        version = experiment_db.snapshot_plugin(plugin_dir, version_label="1.0")
        assert version.artifact_type == "plugin"
        assert "plugin.json" in version.metadata.get("files", [])
        assert "agents/encoder.md" in version.metadata.get("files", [])
        assert "Encoder agent" in version.content

    def test_snapshot_rac_spec(self, experiment_db, tmp_path):
        """Test snapshotting RAC spec."""
        spec_file = tmp_path / "RAC_SPEC.md"
        spec_file.write_text("# RAC Specification\nVersion 2.0")

        version = experiment_db.snapshot_rac_spec(spec_file, version_label="2.0")
        assert version.artifact_type == "rac_spec"
        assert "RAC Specification" in version.content
        assert version.metadata["path"] == str(spec_file)

    def test_snapshot_rac_spec_missing_file(self, experiment_db, tmp_path):
        """Test snapshotting RAC spec when file doesn't exist."""
        spec_file = tmp_path / "MISSING_SPEC.md"
        version = experiment_db.snapshot_rac_spec(spec_file)
        assert version.content == ""
