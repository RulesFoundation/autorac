"""
Experiment Database - tracks encoding runs for continuous improvement.

Key insight: We learn from the JOURNEY (errors, fixes, iterations),
not from comparing predictions to actuals.
"""

import sqlite3
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
import uuid


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

    # The journey
    iterations: list[Iteration] = field(default_factory=list)

    # Final result
    total_duration_ms: int = 0
    final_scores: Optional[FinalScores] = None
    rac_content: str = ""

    # Agent info
    agent_type: str = "encoder"
    agent_model: str = "claude-opus-4-5-20251101"

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
                rac_content TEXT
            )
        """)

        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_citation ON runs(citation)
        """)
        cursor.execute("""
            CREATE INDEX IF NOT EXISTS idx_timestamp ON runs(timestamp)
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

        cursor.execute("""
            INSERT OR REPLACE INTO runs
            (id, timestamp, citation, file_path, complexity_json, iterations_json,
             total_duration_ms, final_scores_json, agent_type, agent_model, rac_content)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        (id, timestamp, citation, file_path, complexity_json, iterations_json,
         total_duration_ms, final_scores_json, agent_type, agent_model, rac_content) = row

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

        return EncodingRun(
            id=id,
            timestamp=datetime.fromisoformat(timestamp),
            citation=citation,
            file_path=file_path,
            complexity=complexity,
            iterations=iterations,
            total_duration_ms=total_duration_ms or 0,
            final_scores=final_scores,
            agent_type=agent_type or "encoder",
            agent_model=agent_model or "",
            rac_content=rac_content or "",
        )
