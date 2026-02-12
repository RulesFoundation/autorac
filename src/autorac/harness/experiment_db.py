"""
Experiment Database - tracks encoding runs for continuous improvement.

Key insight: We learn from the JOURNEY (errors, fixes, iterations),
not from comparing predictions to actuals.

Now also tracks full session transcripts for replay and analysis.

Schema includes:
- SCD2 artifact versioning (plugin, RAC spec, etc.)
- Token usage tracking (via OpenTelemetry or manual entry)
- Full session transcripts
"""

import hashlib
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, Optional

# Artifact types for SCD2 tracking
ArtifactType = Literal[
    "plugin",  # cosilico Claude Code plugin
    "rac_spec",  # RAC_SPEC.md
    "encoder_prompt",  # encoder agent system prompt
    "reviewer_prompt",  # reviewer agent prompts
    "test_runner",  # test runner version
]


@dataclass
class ArtifactVersion:
    """A versioned artifact (SCD2 pattern)."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    artifact_type: str = ""  # ArtifactType
    content_hash: str = ""  # SHA-256 of content for quick comparison
    version_label: str = ""  # e.g., "0.2.1" or git commit hash
    content: str = ""  # Full content (may be large)
    effective_from: datetime = field(default_factory=datetime.now)
    effective_to: Optional[datetime] = None  # None = current version
    metadata: dict = field(default_factory=dict)  # Extra info (file paths, etc.)

    @staticmethod
    def compute_hash(content: str) -> str:
        """Compute SHA-256 hash of content."""
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class TokenUsage:
    """Token usage for a session or run."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Cost estimate using Opus 4.5 pricing (updated Feb 2026).

        Rates: $5/M input, $25/M output, $0.50/M cache read, $6.25/M cache create.
        Source: https://platform.claude.com/docs/en/about-claude/pricing
        """
        return (
            self.input_tokens * 5 / 1_000_000
            + self.output_tokens * 25 / 1_000_000
            + self.cache_read_tokens * 0.50 / 1_000_000
            + self.cache_creation_tokens * 6.25 / 1_000_000
        )


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
    # Validation events (3-tier pipeline)
    "validation_ci_start",
    "validation_ci_end",
    "validation_oracle_start",
    "validation_oracle_end",
    "validation_llm_start",
    "validation_llm_end",
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
class OracleResult:
    """Detailed result from an oracle validator."""

    name: str  # "policyengine" or "taxsim"
    score: Optional[float] = None  # Match rate 0-1
    passed: bool = False
    issues: list[str] = field(default_factory=list)
    duration_ms: int = 0
    test_cases_run: int = 0
    test_cases_passed: int = 0


@dataclass
class FinalScores:
    """Scores from validators after CI passes."""

    rac_reviewer: float = 0.0
    formula_reviewer: float = 0.0
    parameter_reviewer: float = 0.0
    integration_reviewer: float = 0.0
    policyengine_match: Optional[float] = None
    taxsim_match: Optional[float] = None
    # Detailed oracle context (passed to LLM reviewers)
    oracle_context: dict = field(
        default_factory=dict
    )  # {oracle_name: OracleResult as dict}


@dataclass
class PredictedScores:
    """Upfront predictions for calibration tracking."""

    # Dimension scores (0-10)
    rac_reviewer: float = 0.0
    formula_reviewer: float = 0.0
    parameter_reviewer: float = 0.0
    integration_reviewer: float = 0.0
    # Pass/fail predictions
    ci_pass: bool = True
    policyengine_match: Optional[float] = None
    taxsim_match: Optional[float] = None
    # Confidence
    confidence: float = 0.5  # 0-1 confidence in predictions


@dataclass
class ActualScores:
    """Actual scores from validation after encoding."""

    # Dimension scores (0-10)
    rac_reviewer: float = 0.0
    formula_reviewer: float = 0.0
    parameter_reviewer: float = 0.0
    integration_reviewer: float = 0.0
    # Pass/fail results
    ci_pass: bool = False
    ci_error: Optional[str] = None
    policyengine_match: Optional[float] = None
    taxsim_match: Optional[float] = None
    reviewer_issues: list[str] = field(default_factory=list)


@dataclass
class AgentSuggestion:
    """Suggestion from agent for framework improvement."""

    category: str = "documentation"  # documentation, validation, tooling, etc.
    description: str = ""
    predicted_impact: str = "medium"  # low, medium, high
    specific_change: Optional[str] = None


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

    # Predictions and actuals (for calibration)
    predicted: Optional[PredictedScores] = None
    actual: Optional[ActualScores] = None

    # Iteration tracking
    iteration: int = 1
    parent_run_id: Optional[str] = None

    # The journey
    iterations: list[Iteration] = field(default_factory=list)

    # Final result
    total_duration_ms: int = 0
    final_scores: Optional[FinalScores] = None
    rac_content: str = ""

    # Agent info
    agent_type: str = "encoder"
    agent_model: str = ""

    # Suggestions for framework improvement
    suggestions: list[AgentSuggestion] = field(default_factory=list)

    # Session linkage
    session_id: Optional[str] = None

    @property
    def predicted_scores(self) -> Optional[PredictedScores]:
        """Alias for backwards compatibility."""
        return self.predicted

    @predicted_scores.setter
    def predicted_scores(self, value: Optional[PredictedScores]):
        self.predicted = value

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


def create_run(
    file_path: str,
    citation: str,
    agent_type: str,
    agent_model: str,
    rac_content: str,
    statute_text: Optional[str] = None,
    parent_run_id: Optional[str] = None,
) -> EncodingRun:
    """Factory function to create an EncodingRun with defaults."""
    return EncodingRun(
        file_path=file_path,
        citation=citation,
        agent_type=agent_type,
        agent_model=agent_model,
        rac_content=rac_content,
        statute_text=statute_text,
        iteration=2 if parent_run_id else 1,
        parent_run_id=parent_run_id,
    )


class ExperimentDB:
    """SQLite-based experiment database."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Migrate: rename 'runs' to 'encoding_runs' if old table exists
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='runs'"
        )
        if cursor.fetchone():
            cursor.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='encoding_runs'"
            )
            if not cursor.fetchone():
                cursor.execute("ALTER TABLE runs RENAME TO encoding_runs")

        # Encoding runs table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS encoding_runs (
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
                session_id TEXT,
                iteration INTEGER DEFAULT 1,
                parent_run_id TEXT,
                actual_scores_json TEXT,
                suggestions_json TEXT
            )
        """)

        # Add columns if they don't exist (for migration)
        for col, col_type, default in [
            ("predicted_scores_json", "TEXT", None),
            ("session_id", "TEXT", None),
            ("iteration", "INTEGER", "1"),
            ("parent_run_id", "TEXT", None),
            ("actual_scores_json", "TEXT", None),
            ("suggestions_json", "TEXT", None),
        ]:
            try:
                stmt = f"ALTER TABLE encoding_runs ADD COLUMN {col} {col_type}"
                if default is not None:
                    stmt += f" DEFAULT {default}"
                cursor.execute(stmt)
            except sqlite3.OperationalError:
                pass  # Column already exists

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_citation ON encoding_runs(citation)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON encoding_runs(timestamp)
        """)

        # Calibration snapshots table (per-metric rows for trend analysis)
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

        # =====================================================================
        # Artifact Versions table (SCD2 pattern)
        # Tracks plugin, RAC spec, prompts, etc. with version history
        # =====================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS artifact_versions (
                id TEXT PRIMARY KEY,
                artifact_type TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                version_label TEXT,
                content TEXT,
                effective_from TEXT NOT NULL,
                effective_to TEXT,
                metadata_json TEXT
            )
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_artifact_type ON artifact_versions(artifact_type)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_artifact_current ON artifact_versions(artifact_type, effective_to)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_artifact_hash ON artifact_versions(content_hash)
        """)

        # =====================================================================
        # Run-Artifact junction table
        # Links runs to the artifact versions that were active at encoding time
        # =====================================================================
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS run_artifacts (
                run_id TEXT NOT NULL,
                artifact_version_id TEXT NOT NULL,
                PRIMARY KEY (run_id, artifact_version_id),
                FOREIGN KEY (run_id) REFERENCES runs(id),
                FOREIGN KEY (artifact_version_id) REFERENCES artifact_versions(id)
            )
        """)

        # =====================================================================
        # Add token usage columns to sessions (migration)
        # =====================================================================
        for col in [
            "input_tokens",
            "output_tokens",
            "cache_read_tokens",
            "cache_creation_tokens",
            "estimated_cost_usd",
        ]:
            try:
                cursor.execute(
                    f"ALTER TABLE sessions ADD COLUMN {col} INTEGER DEFAULT 0"
                )
            except sqlite3.OperationalError:
                pass  # Column already exists

        conn.commit()
        conn.close()

    def log_run(self, run: EncodingRun) -> str:
        """Log a completed encoding run. Returns the run ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Convert dataclasses to JSON
        complexity_json = json.dumps(
            {
                "cross_references": run.complexity.cross_references,
                "has_nested_structure": run.complexity.has_nested_structure,
                "has_numeric_thresholds": run.complexity.has_numeric_thresholds,
                "has_phase_in_out": run.complexity.has_phase_in_out,
                "estimated_variables": run.complexity.estimated_variables,
                "estimated_parameters": run.complexity.estimated_parameters,
            }
        )

        iterations_json = json.dumps(
            [
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
                    ],
                }
                for it in run.iterations
            ]
        )

        final_scores_json = None
        if run.final_scores:
            final_scores_json = json.dumps(
                {
                    "rac_reviewer": run.final_scores.rac_reviewer,
                    "formula_reviewer": run.final_scores.formula_reviewer,
                    "parameter_reviewer": run.final_scores.parameter_reviewer,
                    "integration_reviewer": run.final_scores.integration_reviewer,
                    "policyengine_match": run.final_scores.policyengine_match,
                    "taxsim_match": run.final_scores.taxsim_match,
                    "oracle_context": run.final_scores.oracle_context,
                }
            )

        predicted_scores_json = None
        if run.predicted:
            predicted_scores_json = json.dumps(
                {
                    "rac_reviewer": run.predicted.rac_reviewer,
                    "formula_reviewer": run.predicted.formula_reviewer,
                    "parameter_reviewer": run.predicted.parameter_reviewer,
                    "integration_reviewer": run.predicted.integration_reviewer,
                    "ci_pass": run.predicted.ci_pass,
                    "policyengine_match": run.predicted.policyengine_match,
                    "taxsim_match": run.predicted.taxsim_match,
                    "confidence": run.predicted.confidence,
                }
            )

        actual_scores_json = None
        if run.actual:
            actual_scores_json = json.dumps(
                {
                    "rac_reviewer": run.actual.rac_reviewer,
                    "formula_reviewer": run.actual.formula_reviewer,
                    "parameter_reviewer": run.actual.parameter_reviewer,
                    "integration_reviewer": run.actual.integration_reviewer,
                    "ci_pass": run.actual.ci_pass,
                    "ci_error": run.actual.ci_error,
                    "policyengine_match": run.actual.policyengine_match,
                    "taxsim_match": run.actual.taxsim_match,
                    "reviewer_issues": run.actual.reviewer_issues,
                }
            )

        suggestions_json = None
        if run.suggestions:
            suggestions_json = json.dumps(
                [
                    {
                        "category": s.category,
                        "description": s.description,
                        "predicted_impact": s.predicted_impact,
                        "specific_change": s.specific_change,
                    }
                    for s in run.suggestions
                ]
            )

        cursor.execute(
            """
            INSERT OR REPLACE INTO encoding_runs
            (id, timestamp, citation, file_path, complexity_json, iterations_json,
             total_duration_ms, final_scores_json, agent_type, agent_model, rac_content,
             predicted_scores_json, session_id, iteration, parent_run_id,
             actual_scores_json, suggestions_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
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
                run.iteration,
                run.parent_run_id,
                actual_scores_json,
                suggestions_json,
            ),
        )

        conn.commit()
        conn.close()

        return run.id

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

    def get_recent_runs(self, limit: int = 20) -> list[EncodingRun]:
        """Get recent runs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM encoding_runs ORDER BY timestamp DESC LIMIT ?", (limit,)
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_run(row) for row in rows]

    def get_runs_for_citation(self, citation: str) -> list[EncodingRun]:
        """Get all runs for a specific citation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM encoding_runs WHERE citation = ? ORDER BY timestamp DESC",
            (citation,),
        )
        rows = cursor.fetchall()
        conn.close()

        return [self._row_to_run(row) for row in rows]

    def update_actual_scores(self, run_id: str, actual: ActualScores) -> None:
        """Update a run with actual scores after validation."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        actual_scores_json = json.dumps(
            {
                "rac_reviewer": actual.rac_reviewer,
                "formula_reviewer": actual.formula_reviewer,
                "parameter_reviewer": actual.parameter_reviewer,
                "integration_reviewer": actual.integration_reviewer,
                "ci_pass": actual.ci_pass,
                "ci_error": actual.ci_error,
                "policyengine_match": actual.policyengine_match,
                "taxsim_match": actual.taxsim_match,
                "reviewer_issues": actual.reviewer_issues,
            }
        )

        cursor.execute(
            "UPDATE encoding_runs SET actual_scores_json = ? WHERE id = ?",
            (actual_scores_json, run_id),
        )

        conn.commit()
        conn.close()

    def get_calibration_data(self) -> list[tuple[PredictedScores, ActualScores]]:
        """Get pairs of predicted and actual scores for calibration analysis.

        Only returns runs that have BOTH predicted and actual scores.
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT * FROM encoding_runs WHERE predicted_scores_json IS NOT NULL AND actual_scores_json IS NOT NULL"
        )
        rows = cursor.fetchall()
        conn.close()

        pairs = []
        for row in rows:
            run = self._row_to_run(row)
            if run.predicted and run.actual:
                pairs.append((run.predicted, run.actual))

        return pairs

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
            "percentages": {
                k: v / total * 100 if total > 0 else 0 for k, v in error_counts.items()
            },
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
        avg = (
            sum(n * c for n, c in iteration_counts.items()) / total if total > 0 else 0
        )

        return {
            "distribution": iteration_counts,
            "average": avg,
            "first_try_rate": iteration_counts.get(1, 0) / total * 100
            if total > 0
            else 0,
            "total_runs": total,
        }

    def _row_to_run(self, row) -> EncodingRun:
        """Convert database row to EncodingRun."""
        # Handle various schema versions (11, 13, or 17 columns)
        if len(row) == 11:
            (
                id,
                timestamp,
                citation,
                file_path,
                complexity_json,
                iterations_json,
                total_duration_ms,
                final_scores_json,
                agent_type,
                agent_model,
                rac_content,
            ) = row
            predicted_scores_json = None
            session_id = None
            iteration = 1
            parent_run_id = None
            actual_scores_json = None
            suggestions_json = None
        elif len(row) == 13:
            (
                id,
                timestamp,
                citation,
                file_path,
                complexity_json,
                iterations_json,
                total_duration_ms,
                final_scores_json,
                agent_type,
                agent_model,
                rac_content,
                predicted_scores_json,
                session_id,
            ) = row
            iteration = 1
            parent_run_id = None
            actual_scores_json = None
            suggestions_json = None
        else:
            (
                id,
                timestamp,
                citation,
                file_path,
                complexity_json,
                iterations_json,
                total_duration_ms,
                final_scores_json,
                agent_type,
                agent_model,
                rac_content,
                predicted_scores_json,
                session_id,
                iteration,
                parent_run_id,
                actual_scores_json,
                suggestions_json,
            ) = row

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
                iterations.append(
                    Iteration(
                        attempt=it_data["attempt"],
                        duration_ms=it_data["duration_ms"],
                        errors=errors,
                        success=it_data.get("success", False),
                    )
                )

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
                oracle_context=f.get("oracle_context", {}),
            )

        # Parse predicted scores
        predicted = None
        if predicted_scores_json:
            p = json.loads(predicted_scores_json)
            predicted = PredictedScores(
                rac_reviewer=p.get("rac_reviewer", p.get("rac", 0)),
                formula_reviewer=p.get("formula_reviewer", p.get("formula", 0)),
                parameter_reviewer=p.get("parameter_reviewer", p.get("param", 0)),
                integration_reviewer=p.get(
                    "integration_reviewer", p.get("integration", 0)
                ),
                ci_pass=p.get("ci_pass", True),
                policyengine_match=p.get("policyengine_match"),
                taxsim_match=p.get("taxsim_match"),
                confidence=p.get("confidence", 0.5),
            )

        # Parse actual scores
        actual = None
        if actual_scores_json:
            a = json.loads(actual_scores_json)
            actual = ActualScores(
                rac_reviewer=a.get("rac_reviewer", 0),
                formula_reviewer=a.get("formula_reviewer", 0),
                parameter_reviewer=a.get("parameter_reviewer", 0),
                integration_reviewer=a.get("integration_reviewer", 0),
                ci_pass=a.get("ci_pass", False),
                ci_error=a.get("ci_error"),
                policyengine_match=a.get("policyengine_match"),
                taxsim_match=a.get("taxsim_match"),
                reviewer_issues=a.get("reviewer_issues", []),
            )

        # Parse suggestions
        suggestions = [
            AgentSuggestion(
                category=s_data.get("category", "documentation"),
                description=s_data.get("description", ""),
                predicted_impact=s_data.get("predicted_impact", "medium"),
                specific_change=s_data.get("specific_change"),
            )
            for s_data in json.loads(suggestions_json)
        ] if suggestions_json else []

        return EncodingRun(
            id=id,
            timestamp=datetime.fromisoformat(timestamp),
            citation=citation,
            file_path=file_path,
            complexity=complexity,
            predicted=predicted,
            actual=actual,
            iteration=iteration or 1,
            parent_run_id=parent_run_id,
            iterations=iterations,
            total_duration_ms=total_duration_ms or 0,
            final_scores=final_scores,
            agent_type=agent_type or "encoder",
            agent_model=agent_model or "",
            rac_content=rac_content or "",
            suggestions=suggestions,
            session_id=session_id,
        )

    # =========================================================================
    # Session Logging Methods
    # =========================================================================

    def start_session(
        self, model: str = "", cwd: str = "", session_id: str = None
    ) -> Session:
        """Start a new session and return it."""
        session = Session(
            model=model,
            cwd=cwd or os.getcwd(),
        )
        # Allow custom session_id for SDK orchestrator
        if session_id:
            session.id = session_id

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            INSERT INTO sessions (id, started_at, model, cwd, event_count, total_tokens)
            VALUES (?, ?, ?, ?, 0, 0)
        """,
            (session.id, session.started_at.isoformat(), session.model, session.cwd),
        )

        conn.commit()
        conn.close()

        return session

    def end_session(self, session_id: str) -> None:
        """Mark a session as ended."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            UPDATE sessions SET ended_at = ? WHERE id = ?
        """,
            (datetime.now().isoformat(), session_id),
        )

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
            (session_id,),
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

        cursor.execute(
            """
            INSERT INTO session_events (id, session_id, sequence, timestamp, event_type, tool_name, content, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                event.id,
                event.session_id,
                event.sequence,
                event.timestamp.isoformat(),
                event.event_type,
                event.tool_name,
                event.content,
                json.dumps(event.metadata),
            ),
        )

        # Update event count
        cursor.execute(
            """
            UPDATE sessions SET event_count = event_count + 1 WHERE id = ?
        """,
            (session_id,),
        )

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

        cursor.execute(
            """
            SELECT id, session_id, sequence, timestamp, event_type, tool_name, content, metadata_json
            FROM session_events
            WHERE session_id = ?
            ORDER BY sequence
        """,
            (session_id,),
        )

        rows = cursor.fetchall()
        conn.close()

        events = []
        for row in rows:
            events.append(
                SessionEvent(
                    id=row[0],
                    session_id=row[1],
                    sequence=row[2],
                    timestamp=datetime.fromisoformat(row[3])
                    if row[3]
                    else datetime.now(),
                    event_type=row[4] or "",
                    tool_name=row[5],
                    content=row[6] or "",
                    metadata=json.loads(row[7]) if row[7] else {},
                )
            )

        return events

    def get_recent_sessions(self, limit: int = 20) -> list[Session]:
        """Get recent sessions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT * FROM sessions ORDER BY started_at DESC LIMIT ?
        """,
            (limit,),
        )

        rows = cursor.fetchall()
        conn.close()

        sessions = []
        for row in rows:
            sessions.append(
                Session(
                    id=row[0],
                    run_id=row[1],
                    started_at=datetime.fromisoformat(row[2])
                    if row[2]
                    else datetime.now(),
                    ended_at=datetime.fromisoformat(row[3]) if row[3] else None,
                    model=row[4] or "",
                    cwd=row[5] or "",
                    event_count=row[6] or 0,
                    total_tokens=row[7] or 0,
                )
            )

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

    # =========================================================================
    # Artifact Version Methods (SCD2)
    # =========================================================================

    def log_artifact_version(
        self,
        artifact_type: str,
        content: str,
        version_label: str = "",
        metadata: Optional[dict] = None,
    ) -> ArtifactVersion:
        """
        Log a new artifact version. Uses SCD2 pattern:
        - If content hash matches current version, returns existing
        - Otherwise, closes current version and creates new one
        """
        content_hash = ArtifactVersion.compute_hash(content)

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Check if this exact version already exists
        cursor.execute(
            """
            SELECT id, version_label, effective_from FROM artifact_versions
            WHERE artifact_type = ? AND content_hash = ? AND effective_to IS NULL
        """,
            (artifact_type, content_hash),
        )
        existing = cursor.fetchone()

        if existing:
            conn.close()
            return ArtifactVersion(
                id=existing[0],
                artifact_type=artifact_type,
                content_hash=content_hash,
                version_label=existing[1] or version_label,
                content=content,
                effective_from=datetime.fromisoformat(existing[2]),
                effective_to=None,
                metadata=metadata or {},
            )

        # Close any existing current version
        now = datetime.now()
        cursor.execute(
            """
            UPDATE artifact_versions
            SET effective_to = ?
            WHERE artifact_type = ? AND effective_to IS NULL
        """,
            (now.isoformat(), artifact_type),
        )

        # Create new version
        version = ArtifactVersion(
            artifact_type=artifact_type,
            content_hash=content_hash,
            version_label=version_label,
            content=content,
            effective_from=now,
            effective_to=None,
            metadata=metadata or {},
        )

        cursor.execute(
            """
            INSERT INTO artifact_versions
            (id, artifact_type, content_hash, version_label, content, effective_from, effective_to, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                version.id,
                version.artifact_type,
                version.content_hash,
                version.version_label,
                version.content,
                version.effective_from.isoformat(),
                None,
                json.dumps(version.metadata),
            ),
        )

        conn.commit()
        conn.close()

        return version

    def get_current_artifact_version(
        self, artifact_type: str
    ) -> Optional[ArtifactVersion]:
        """Get the current (effective_to IS NULL) version of an artifact."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, artifact_type, content_hash, version_label, content, effective_from, effective_to, metadata_json
            FROM artifact_versions
            WHERE artifact_type = ? AND effective_to IS NULL
        """,
            (artifact_type,),
        )

        row = cursor.fetchone()
        conn.close()

        if not row:
            return None

        return ArtifactVersion(
            id=row[0],
            artifact_type=row[1],
            content_hash=row[2],
            version_label=row[3] or "",
            content=row[4] or "",
            effective_from=datetime.fromisoformat(row[5]),
            effective_to=datetime.fromisoformat(row[6]) if row[6] else None,
            metadata=json.loads(row[7]) if row[7] else {},
        )

    def get_artifact_history(self, artifact_type: str) -> list[ArtifactVersion]:
        """Get version history for an artifact type, ordered by effective_from DESC."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT id, artifact_type, content_hash, version_label, content, effective_from, effective_to, metadata_json
            FROM artifact_versions
            WHERE artifact_type = ?
            ORDER BY effective_from DESC
        """,
            (artifact_type,),
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            ArtifactVersion(
                id=row[0],
                artifact_type=row[1],
                content_hash=row[2],
                version_label=row[3] or "",
                content=row[4] or "",
                effective_from=datetime.fromisoformat(row[5]),
                effective_to=datetime.fromisoformat(row[6]) if row[6] else None,
                metadata=json.loads(row[7]) if row[7] else {},
            )
            for row in rows
        ]

    def link_run_to_artifacts(
        self, run_id: str, artifact_version_ids: list[str]
    ) -> None:
        """Link a run to the artifact versions that were active during encoding."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for av_id in artifact_version_ids:
            cursor.execute(
                """
                INSERT OR IGNORE INTO run_artifacts (run_id, artifact_version_id)
                VALUES (?, ?)
            """,
                (run_id, av_id),
            )

        conn.commit()
        conn.close()

    def link_run_to_current_artifacts(self, run_id: str) -> list[str]:
        """Link a run to all current artifact versions. Returns list of artifact version IDs."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get all current artifact versions
        cursor.execute("""
            SELECT id FROM artifact_versions WHERE effective_to IS NULL
        """)
        current_ids = [row[0] for row in cursor.fetchall()]

        # Link to run
        for av_id in current_ids:
            cursor.execute(
                """
                INSERT OR IGNORE INTO run_artifacts (run_id, artifact_version_id)
                VALUES (?, ?)
            """,
                (run_id, av_id),
            )

        conn.commit()
        conn.close()

        return current_ids

    def get_run_artifacts(self, run_id: str) -> list[ArtifactVersion]:
        """Get all artifact versions linked to a run."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT av.id, av.artifact_type, av.content_hash, av.version_label, av.content,
                   av.effective_from, av.effective_to, av.metadata_json
            FROM artifact_versions av
            JOIN run_artifacts ra ON av.id = ra.artifact_version_id
            WHERE ra.run_id = ?
        """,
            (run_id,),
        )

        rows = cursor.fetchall()
        conn.close()

        return [
            ArtifactVersion(
                id=row[0],
                artifact_type=row[1],
                content_hash=row[2],
                version_label=row[3] or "",
                content=row[4] or "",
                effective_from=datetime.fromisoformat(row[5]),
                effective_to=datetime.fromisoformat(row[6]) if row[6] else None,
                metadata=json.loads(row[7]) if row[7] else {},
            )
            for row in rows
        ]

    def update_session_tokens(
        self,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_creation_tokens: int = 0,
    ) -> None:
        """Update token usage for a session."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Estimate cost (Opus 4.5 pricing, updated Feb 2026)
        estimated_cost = (
            input_tokens * 5 / 1_000_000
            + output_tokens * 25 / 1_000_000
            + cache_read_tokens * 0.50 / 1_000_000
        )

        cursor.execute(
            """
            UPDATE sessions
            SET input_tokens = ?,
                output_tokens = ?,
                cache_read_tokens = ?,
                cache_creation_tokens = ?,
                total_tokens = ?,
                estimated_cost_usd = ?
            WHERE id = ?
        """,
            (
                input_tokens,
                output_tokens,
                cache_read_tokens,
                cache_creation_tokens,
                input_tokens + output_tokens,
                estimated_cost,
                session_id,
            ),
        )

        conn.commit()
        conn.close()

    def snapshot_plugin(
        self, plugin_path: Path, version_label: str = ""
    ) -> ArtifactVersion:
        """
        Snapshot the entire cosilico plugin as a single artifact.
        Concatenates all agent, skill, command, and hook files.
        """
        content_parts = []
        file_list = []

        # Collect all plugin files
        for subdir in ["agents", "skills", "commands", "hooks"]:
            subdir_path = plugin_path / subdir
            if subdir_path.exists():
                for file in sorted(subdir_path.rglob("*")):
                    if file.is_file() and file.suffix in [".md", ".sh", ".py"]:
                        rel_path = file.relative_to(plugin_path)
                        file_list.append(str(rel_path))
                        content_parts.append(f"=== {rel_path} ===\n")
                        content_parts.append(file.read_text())
                        content_parts.append("\n\n")

        # Also include plugin.json
        plugin_json = plugin_path / "plugin.json"
        if plugin_json.exists():
            file_list.insert(0, "plugin.json")
            content_parts.insert(
                0, f"=== plugin.json ===\n{plugin_json.read_text()}\n\n"
            )

        full_content = "".join(content_parts)

        return self.log_artifact_version(
            artifact_type="plugin",
            content=full_content,
            version_label=version_label,
            metadata={"files": file_list, "path": str(plugin_path)},
        )

    def snapshot_rac_spec(
        self, spec_path: Path, version_label: str = ""
    ) -> ArtifactVersion:
        """Snapshot the RAC_SPEC.md file."""
        content = spec_path.read_text() if spec_path.exists() else ""

        return self.log_artifact_version(
            artifact_type="rac_spec",
            content=content,
            version_label=version_label,
            metadata={"path": str(spec_path)},
        )
