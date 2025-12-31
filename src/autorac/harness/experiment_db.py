"""
Experiment Database - tracks encoding runs for continuous improvement.

Key insight: We learn from the JOURNEY (errors, fixes, iterations),
not from comparing predictions to actuals.

Now also tracks full session transcripts for replay and analysis.
"""

import sqlite3
import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, Literal
import uuid


# Session event types
EventType = Literal[
    "session_start",
    "session_end",
    "user_prompt",
    "assistant_response",
    "tool_call",
    "tool_result",
    "subagent_start",
    "subagent_end",
]


@dataclass
class SessionEvent:
    """A single event in a session transcript."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    session_id: str = ""
    sequence: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    event_type: str = ""  # EventType
    tool_name: Optional[str] = None
    content: str = ""  # Main content (prompt, response, tool input/output)
    metadata: dict = field(default_factory=dict)  # Extra data (tokens, duration, etc.)


@dataclass
class Session:
    """A full Claude Code session transcript."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    run_id: Optional[str] = None  # FK to EncodingRun if this is an encoding session
    started_at: datetime = field(default_factory=datetime.now)
    ended_at: Optional[datetime] = None
    model: str = ""
    cwd: str = ""
    event_count: int = 0
    total_tokens: int = 0


@dataclass
class ComplexityFactors:
    """Upfront analysis of statute complexity."""
    cross_references: list[str] = field(default_factory=list)  # ["1402(a)", "164(f)"]
    has_nested_structure: bool = False
    has_numeric_thresholds: bool = False
    has_phase_in_out: bool = False
    estimated_variables: int = 1
    estimated_parameters: int = 0


@dataclass
class IterationError:
    """An error encountered during encoding."""
    error_type: str  # "parse", "test", "import", "style", "other"
    message: str
    variable: Optional[str] = None  # Which variable failed, if applicable
    fix_applied: Optional[str] = None  # What fix was attempted


@dataclass
class Iteration:
    """A single encoding attempt."""
    attempt: int
    duration_ms: int
    errors: list[IterationError] = field(default_factory=list)
    success: bool = False


@dataclass
class FinalScores:
    """Scores from validators after CI passes."""
    rac_reviewer: float = 0.0
    formula_reviewer: float = 0.0
    parameter_reviewer: float = 0.0
    integration_reviewer: float = 0.0
    policyengine_match: Optional[float] = None
    taxsim_match: Optional[float] = None


@dataclass
class PredictedScores:
    """Upfront predictions for calibration tracking."""
    # Dimension scores (0-10)
    rac: float = 0.0
    formula: float = 0.0
    param: float = 0.0
    integration: float = 0.0
    # Effort predictions
    iterations: int = 1
    time_minutes: float = 0.0
    confidence: float = 0.5  # 0-1 confidence in predictions


@dataclass
class EncodingRun:
    """A complete encoding run from start to finish."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)

    # What we're encoding
    citation: str = ""
    file_path: str = ""
    statute_text: Optional[str] = None

    # Upfront analysis
    complexity: ComplexityFactors = field(default_factory=ComplexityFactors)

    # Predictions (for calibration)
    predicted_scores: Optional[PredictedScores] = None

    # The journey
    iterations: list[Iteration] = field(default_factory=list)

    # Final result
    total_duration_ms: int = 0
    final_scores: Optional[FinalScores] = None
    rac_content: str = ""

    # Agent info
    agent_type: str = "encoder"
    agent_model: str = "claude-opus-4-5-20251101"

    # Session linkage
    session_id: Optional[str] = None

    @property
    def iterations_needed(self) -> int:
        return len(self.iterations)

    @property
    def success(self) -> bool:
        return self.iterations and self.iterations[-1].success

    @property
    def all_errors(self) -> list[IterationError]:
        errors = []
        for it in self.iterations:
            errors.extend(it.errors)
        return errors


class ExperimentDB:
    """SQLite-based experiment database."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Encoding runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS runs (
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
                rac_content TEXT,
                predicted_scores_json TEXT,
                session_id TEXT
            )
        """)

        # Add columns if they don't exist (for migration)
        try:
            cursor.execute("ALTER TABLE runs ADD COLUMN predicted_scores_json TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists
        try:
            cursor.execute("ALTER TABLE runs ADD COLUMN session_id TEXT")
        except sqlite3.OperationalError:
            pass  # Column already exists

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_citation ON runs(citation)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON runs(timestamp)
        """)

        # Sessions table - full Claude Code session transcripts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                run_id TEXT,
                started_at TEXT,
                ended_at TEXT,
                model TEXT,
                cwd TEXT,
                event_count INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                FOREIGN KEY (run_id) REFERENCES runs(id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_run ON sessions(run_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_session_started ON sessions(started_at)
        """)

        # Session events table - individual events within a session
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS session_events (
                id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                sequence INTEGER,
                timestamp TEXT,
                event_type TEXT,
                tool_name TEXT,
                content TEXT,
                metadata_json TEXT,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_session ON session_events(session_id)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_event_type ON session_events(event_type)
        """)

        conn.commit()
        conn.close()

    def log_run(self, run: EncodingRun):
        """Log a completed encoding run."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Convert dataclasses to JSON
        complexity_json = json.dumps({
            "cross_references": run.complexity.cross_references,
            "has_nested_structure": run.complexity.has_nested_structure,
            "has_numeric_thresholds": run.complexity.has_numeric_thresholds,
            "has_phase_in_out": run.complexity.has_phase_in_out,
            "estimated_variables": run.complexity.estimated_variables,
            "estimated_parameters": run.complexity.estimated_parameters,
        })

        iterations_json = json.dumps([
            {
                "attempt": it.attempt,
                "duration_ms": it.duration_ms,
                "success": it.success,
                "errors": [
                    {
                        "error_type": e.error_type,
                        "message": e.message,
                        "variable": e.variable,
                        "fix_applied": e.fix_applied,
                    }
                    for e in it.errors
                ]
            }
            for it in run.iterations
        ])

        final_scores_json = None
        if run.final_scores:
            final_scores_json = json.dumps({
                "rac_reviewer": run.final_scores.rac_reviewer,
                "formula_reviewer": run.final_scores.formula_reviewer,
                "parameter_reviewer": run.final_scores.parameter_reviewer,
                "integration_reviewer": run.final_scores.integration_reviewer,
                "policyengine_match": run.final_scores.policyengine_match,
                "taxsim_match": run.final_scores.taxsim_match,
            })

        predicted_scores_json = None
        if run.predicted_scores:
            predicted_scores_json = json.dumps({
                "rac": run.predicted_scores.rac,
                "formula": run.predicted_scores.formula,
                "param": run.predicted_scores.param,
                "integration": run.predicted_scores.integration,
                "iterations": run.predicted_scores.iterations,
                "time_minutes": run.predicted_scores.time_minutes,
                "confidence": run.predicted_scores.confidence,
            })

        cursor.execute("""
            INSERT OR REPLACE INTO runs
            (id, timestamp, citation, file_path, complexity_json, iterations_json,
             total_duration_ms, final_scores_json, agent_type, agent_model, rac_content,
             predicted_scores_json, session_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run.id,
            run.timestamp.isoformat(),
            run.citation,
            run.file_path,
            complexity_json,
            iterations_json,
            run.total_duration_ms,
            final_scores_json,
            run.agent_type,
            run.agent_model,
            run.rac_content,
            predicted_scores_json,
            run.session_id,
        ))

        conn.commit()
        conn.close()

    def get_run(self, run_id: str) -> Optional[EncodingRun]:
        """Get a specific run by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM runs WHERE id = ?", (run_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return self._row_to_run(row)

    def get_recent_runs(self, limit: int = 20) -> list[EncodingRun]:
        """Get recent runs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM runs ORDER BY timestamp DESC LIMIT ?",
            (limit,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_run(row) for row in rows]

    def get_error_stats(self) -> dict:
        """Get error type distribution."""
        runs = self.get_recent_runs(limit=100)

        error_counts = {}
        for run in runs:
            for err in run.all_errors:
                error_counts[err.error_type] = error_counts.get(err.error_type, 0) + 1

        total = sum(error_counts.values())
        return {
            "counts": error_counts,
            "percentages": {k: v/total*100 if total > 0 else 0 for k, v in error_counts.items()},
            "total_runs": len(runs),
            "total_errors": total,
        }

    def get_iteration_stats(self) -> dict:
        """Get iteration distribution."""
        runs = self.get_recent_runs(limit=100)

        iteration_counts = {}
        for run in runs:
            n = run.iterations_needed
            iteration_counts[n] = iteration_counts.get(n, 0) + 1

        total = len(runs)
        avg = sum(n * c for n, c in iteration_counts.items()) / total if total > 0 else 0

        return {
            "distribution": iteration_counts,
            "average": avg,
            "first_try_rate": iteration_counts.get(1, 0) / total * 100 if total > 0 else 0,
            "total_runs": total,
        }

    def _row_to_run(self, row) -> EncodingRun:
        """Convert database row to EncodingRun."""
        # Handle both old (11 columns) and new (13 columns) schema
        if len(row) == 11:
            (id, timestamp, citation, file_path, complexity_json, iterations_json,
             total_duration_ms, final_scores_json, agent_type, agent_model, rac_content) = row
            predicted_scores_json = None
            session_id = None
        else:
            (id, timestamp, citation, file_path, complexity_json, iterations_json,
             total_duration_ms, final_scores_json, agent_type, agent_model, rac_content,
             predicted_scores_json, session_id) = row

        # Parse complexity
        c = json.loads(complexity_json) if complexity_json else {}
        complexity = ComplexityFactors(
            cross_references=c.get("cross_references", []),
            has_nested_structure=c.get("has_nested_structure", False),
            has_numeric_thresholds=c.get("has_numeric_thresholds", False),
            has_phase_in_out=c.get("has_phase_in_out", False),
            estimated_variables=c.get("estimated_variables", 1),
            estimated_parameters=c.get("estimated_parameters", 0),
        )

        # Parse iterations
        iterations = []
        if iterations_json:
            for it_data in json.loads(iterations_json):
                errors = [
                    IterationError(
                        error_type=e["error_type"],
                        message=e["message"],
                        variable=e.get("variable"),
                        fix_applied=e.get("fix_applied"),
                    )
                    for e in it_data.get("errors", [])
                ]
                iterations.append(Iteration(
                    attempt=it_data["attempt"],
                    duration_ms=it_data["duration_ms"],
                    errors=errors,
                    success=it_data.get("success", False),
                ))

        # Parse final scores
        final_scores = None
        if final_scores_json:
            f = json.loads(final_scores_json)
            final_scores = FinalScores(
                rac_reviewer=f.get("rac_reviewer", 0),
                formula_reviewer=f.get("formula_reviewer", 0),
                parameter_reviewer=f.get("parameter_reviewer", 0),
                integration_reviewer=f.get("integration_reviewer", 0),
                policyengine_match=f.get("policyengine_match"),
                taxsim_match=f.get("taxsim_match"),
            )

        # Parse predicted scores
        predicted_scores = None
        if predicted_scores_json:
            p = json.loads(predicted_scores_json)
            predicted_scores = PredictedScores(
                rac=p.get("rac", 0),
                formula=p.get("formula", 0),
                param=p.get("param", 0),
                integration=p.get("integration", 0),
                iterations=p.get("iterations", 1),
                time_minutes=p.get("time_minutes", 0),
                confidence=p.get("confidence", 0.5),
            )

        return EncodingRun(
            id=id,
            timestamp=datetime.fromisoformat(timestamp),
            citation=citation,
            file_path=file_path,
            complexity=complexity,
            predicted_scores=predicted_scores,
            iterations=iterations,
            total_duration_ms=total_duration_ms or 0,
            final_scores=final_scores,
            agent_type=agent_type or "encoder",
            agent_model=agent_model or "",
            rac_content=rac_content or "",
            session_id=session_id,
        )

    # =========================================================================
    # Session Logging Methods
    # =========================================================================

    def start_session(self, model: str = "", cwd: str = "") -> Session:
        """Start a new session and return it."""
        session = Session(
            model=model,
            cwd=cwd or os.getcwd(),
        )

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO sessions (id, started_at, model, cwd, event_count, total_tokens)
            VALUES (?, ?, ?, ?, 0, 0)
        """, (session.id, session.started_at.isoformat(), session.model, session.cwd))

        conn.commit()
        conn.close()

        return session

    def end_session(self, session_id: str) -> None:
        """Mark a session as ended."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE sessions SET ended_at = ? WHERE id = ?
        """, (datetime.now().isoformat(), session_id))

        conn.commit()
        conn.close()

    def log_event(
        self,
        session_id: str,
        event_type: str,
        content: str = "",
        tool_name: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> SessionEvent:
        """Log an event to a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get next sequence number
        cursor.execute(
            "SELECT COALESCE(MAX(sequence), 0) + 1 FROM session_events WHERE session_id = ?",
            (session_id,)
        )
        sequence = cursor.fetchone()[0]

        event = SessionEvent(
            session_id=session_id,
            sequence=sequence,
            event_type=event_type,
            tool_name=tool_name,
            content=content,
            metadata=metadata or {},
        )

        cursor.execute("""
            INSERT INTO session_events (id, session_id, sequence, timestamp, event_type, tool_name, content, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            event.id,
            event.session_id,
            event.sequence,
            event.timestamp.isoformat(),
            event.event_type,
            event.tool_name,
            event.content,
            json.dumps(event.metadata),
        ))

        # Update event count
        cursor.execute("""
            UPDATE sessions SET event_count = event_count + 1 WHERE id = ?
        """, (session_id,))

        conn.commit()
        conn.close()

        return event

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get a session by ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM sessions WHERE id = ?", (session_id,))
        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return Session(
            id=row[0],
            run_id=row[1],
            started_at=datetime.fromisoformat(row[2]) if row[2] else datetime.now(),
            ended_at=datetime.fromisoformat(row[3]) if row[3] else None,
            model=row[4] or "",
            cwd=row[5] or "",
            event_count=row[6] or 0,
            total_tokens=row[7] or 0,
        )

    def get_session_events(self, session_id: str) -> list[SessionEvent]:
        """Get all events for a session, ordered by sequence."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT id, session_id, sequence, timestamp, event_type, tool_name, content, metadata_json
            FROM session_events
            WHERE session_id = ?
            ORDER BY sequence
        """, (session_id,))

        rows = cursor.fetchall()
        conn.close()

        events = []
        for row in rows:
            events.append(SessionEvent(
                id=row[0],
                session_id=row[1],
                sequence=row[2],
                timestamp=datetime.fromisoformat(row[3]) if row[3] else datetime.now(),
                event_type=row[4] or "",
                tool_name=row[5],
                content=row[6] or "",
                metadata=json.loads(row[7]) if row[7] else {},
            ))

        return events

    def get_recent_sessions(self, limit: int = 20) -> list[Session]:
        """Get recent sessions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?
        """, (limit,))

        rows = cursor.fetchall()
        conn.close()

        sessions = []
        for row in rows:
            sessions.append(Session(
                id=row[0],
                run_id=row[1],
                started_at=datetime.fromisoformat(row[2]) if row[2] else datetime.now(),
                ended_at=datetime.fromisoformat(row[3]) if row[3] else None,
                model=row[4] or "",
                cwd=row[5] or "",
                event_count=row[6] or 0,
                total_tokens=row[7] or 0,
            ))

        return sessions

    def get_session_stats(self) -> dict:
        """Get session statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Total sessions
        cursor.execute("SELECT COUNT(*) FROM sessions")
        total = cursor.fetchone()[0]

        # Event type distribution
        cursor.execute("""
            SELECT event_type, COUNT(*) as count
            FROM session_events
            GROUP BY event_type
            ORDER BY count DESC
        """)
        event_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Tool usage
        cursor.execute("""
            SELECT tool_name, COUNT(*) as count
            FROM session_events
            WHERE tool_name IS NOT NULL
            GROUP BY tool_name
            ORDER BY count DESC
            LIMIT 20
        """)
        tool_counts = {row[0]: row[1] for row in cursor.fetchall()}

        # Average events per session
        cursor.execute("SELECT AVG(event_count) FROM sessions")
        avg_events = cursor.fetchone()[0] or 0

        conn.close()

        return {
            "total_sessions": total,
            "event_type_counts": event_counts,
            "tool_usage": tool_counts,
            "avg_events_per_session": round(avg_events, 1),
        }
