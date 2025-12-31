"""
Experiment Database for encoding runs.

Logs every encoding attempt with:
- Predicted scores (agent's self-assessment)
- Actual scores (from validators)
- Agent suggestions for framework improvements
- Calibration tracking over time
"""

import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid


@dataclass
class PredictedScores:
    """Agent's predicted scores before validation."""
    # Iteration predictions (key calibration metrics)
    iterations_to_pass: int = 1  # How many attempts until CI passes
    expected_errors: list[str] = field(default_factory=list)  # ["parse", "test", "import"]
    time_minutes: float = 5.0  # Estimated time to complete

    # Reviewer score predictions (after CI passes)
    rac_reviewer: float = 7.0  # 0-10
    formula_reviewer: float = 7.0
    parameter_reviewer: float = 7.0
    integration_reviewer: float = 7.0

    # Oracle predictions
    policyengine_match: Optional[float] = None  # 0-1, None if no oracle
    taxsim_match: Optional[float] = None

    # Meta
    confidence: float = 0.5  # Agent's confidence in predictions


@dataclass
class ActualScores:
    """Actual scores from validators."""
    # Iteration actuals
    iterations_needed: int = 1
    errors_encountered: list[str] = field(default_factory=list)
    time_minutes: float = 0.0

    # Reviewer scores
    rac_reviewer: Optional[float] = None
    formula_reviewer: Optional[float] = None
    parameter_reviewer: Optional[float] = None
    integration_reviewer: Optional[float] = None

    # CI details
    ci_pass: Optional[bool] = None
    ci_error: Optional[str] = None

    # Oracle results
    policyengine_match: Optional[float] = None
    taxsim_match: Optional[float] = None

    # Issues found
    reviewer_issues: list[str] = field(default_factory=list)


@dataclass
class AgentSuggestion:
    """Agent's suggestion for framework improvement."""
    category: str  # "documentation", "agent_prompt", "validator", "dsl"
    description: str
    predicted_impact: str  # "high", "medium", "low"
    specific_change: Optional[str] = None


@dataclass
class EncodingRun:
    """A single encoding attempt."""
    id: str
    timestamp: datetime
    file_path: str
    citation: str  # e.g., "26 USC 1(h)(1)(E)"

    # Agent info
    agent_type: str  # "encoder", "formula_writer", etc.
    agent_model: str  # "claude-opus-4-5-20251101"

    # Content
    rac_content: str
    statute_text: Optional[str] = None

    # Scores
    predicted: Optional[PredictedScores] = None
    actual: Optional[ActualScores] = None

    # Suggestions
    suggestions: list[AgentSuggestion] = field(default_factory=list)

    # Iteration info
    iteration: int = 1
    parent_run_id: Optional[str] = None  # If this is a revision

    # Timing
    encoding_duration_ms: Optional[int] = None
    validation_duration_ms: Optional[int] = None


class ExperimentDB:
    """SQLite-based experiment database."""

    def __init__(self, db_path: str | Path = "experiments.db"):
        self.db_path = Path(db_path)
        self._init_db()

    def _init_db(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS encoding_runs (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                file_path TEXT NOT NULL,
                citation TEXT NOT NULL,
                agent_type TEXT NOT NULL,
                agent_model TEXT NOT NULL,
                rac_content TEXT NOT NULL,
                statute_text TEXT,
                predicted_scores TEXT,  -- JSON
                actual_scores TEXT,     -- JSON
                suggestions TEXT,       -- JSON
                iteration INTEGER DEFAULT 1,
                parent_run_id TEXT,
                encoding_duration_ms INTEGER,
                validation_duration_ms INTEGER,
                FOREIGN KEY (parent_run_id) REFERENCES encoding_runs(id)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_snapshots (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                metric_name TEXT NOT NULL,
                predicted_mean REAL,
                actual_mean REAL,
                mse REAL,
                n_samples INTEGER
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_citation ON encoding_runs(citation)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_runs_timestamp ON encoding_runs(timestamp)
        """)

        conn.commit()
        conn.close()

    def log_run(self, run: EncodingRun) -> str:
        """Log an encoding run to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO encoding_runs (
                id, timestamp, file_path, citation, agent_type, agent_model,
                rac_content, statute_text, predicted_scores, actual_scores,
                suggestions, iteration, parent_run_id, encoding_duration_ms,
                validation_duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run.id,
            run.timestamp.isoformat(),
            run.file_path,
            run.citation,
            run.agent_type,
            run.agent_model,
            run.rac_content,
            run.statute_text,
            json.dumps(asdict(run.predicted)) if run.predicted else None,
            json.dumps(asdict(run.actual)) if run.actual else None,
            json.dumps([asdict(s) for s in run.suggestions]),
            run.iteration,
            run.parent_run_id,
            run.encoding_duration_ms,
            run.validation_duration_ms,
        ))

        conn.commit()
        conn.close()
        return run.id

    def update_actual_scores(self, run_id: str, actual: ActualScores):
        """Update a run with actual validation scores."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE encoding_runs
            SET actual_scores = ?
            WHERE id = ?
        """, (json.dumps(asdict(actual)), run_id))

        conn.commit()
        conn.close()

    def get_run(self, run_id: str) -> Optional[EncodingRun]:
        """Get a specific run by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM encoding_runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_run(row)

    def get_runs_for_citation(self, citation: str) -> list[EncodingRun]:
        """Get all runs for a specific citation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM encoding_runs WHERE citation = ? ORDER BY timestamp",
            (citation,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_run(row) for row in rows]

    def get_recent_runs(self, limit: int = 100) -> list[EncodingRun]:
        """Get most recent runs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM encoding_runs ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_run(row) for row in rows]

    def get_calibration_data(self) -> list[tuple[PredictedScores, ActualScores]]:
        """Get all runs with both predicted and actual scores for calibration."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT predicted_scores, actual_scores
            FROM encoding_runs
            WHERE predicted_scores IS NOT NULL AND actual_scores IS NOT NULL
        """)
        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            pred = PredictedScores(**json.loads(row[0]))
            actual = ActualScores(**json.loads(row[1]))
            results.append((pred, actual))

        return results

    def _row_to_run(self, row: tuple) -> EncodingRun:
        """Convert a database row to an EncodingRun."""
        (id_, timestamp, file_path, citation, agent_type, agent_model,
         rac_content, statute_text, predicted_scores, actual_scores,
         suggestions, iteration, parent_run_id, encoding_duration_ms,
         validation_duration_ms) = row

        return EncodingRun(
            id=id_,
            timestamp=datetime.fromisoformat(timestamp),
            file_path=file_path,
            citation=citation,
            agent_type=agent_type,
            agent_model=agent_model,
            rac_content=rac_content,
            statute_text=statute_text,
            predicted=PredictedScores(**json.loads(predicted_scores)) if predicted_scores else None,
            actual=ActualScores(**json.loads(actual_scores)) if actual_scores else None,
            suggestions=[AgentSuggestion(**s) for s in json.loads(suggestions)] if suggestions else [],
            iteration=iteration,
            parent_run_id=parent_run_id,
            encoding_duration_ms=encoding_duration_ms,
            validation_duration_ms=validation_duration_ms,
        )


def create_run(
    file_path: str,
    citation: str,
    agent_type: str,
    agent_model: str,
    rac_content: str,
    statute_text: Optional[str] = None,
    parent_run_id: Optional[str] = None,
) -> EncodingRun:
    """Create a new encoding run."""
    return EncodingRun(
        id=str(uuid.uuid4())[:8],
        timestamp=datetime.now(),
        file_path=file_path,
        citation=citation,
        agent_type=agent_type,
        agent_model=agent_model,
        rac_content=rac_content,
        statute_text=statute_text,
        parent_run_id=parent_run_id,
        iteration=1 if parent_run_id is None else 2,  # Will be updated properly
    )
