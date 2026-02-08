"""
Calibration Metrics - track prediction accuracy over time.

Measures how well agents predict their own performance:
- Mean squared error between predicted and actual scores
- Calibration curves (predicted vs actual)
- Confidence calibration
- Trend analysis over time
"""

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import sqlite3
from pathlib import Path

from .experiment_db import ExperimentDB, PredictedScores, ActualScores


@dataclass
class CalibrationMetrics:
    """Calibration metrics for a specific score type."""
    metric_name: str = ""
    n_samples: int = 0
    predicted_mean: float = 0.0
    actual_mean: float = 0.0
    mse: float = 0.0  # Mean squared error
    mae: float = 0.0  # Mean absolute error
    bias: float = 0.0  # Systematic over/under prediction
    correlation: Optional[float] = None  # Pearson correlation


@dataclass
class CalibrationSnapshot:
    """Point-in-time calibration across all metrics."""
    timestamp: datetime = field(default_factory=datetime.now)
    metrics: dict[str, CalibrationMetrics] = field(default_factory=dict)
    total_runs: int = 0
    pass_rate: float = 0.0  # % of runs where all validators passed


def _compute_metric(name: str, pairs: list[tuple[float, float]]) -> CalibrationMetrics:
    """Compute calibration metrics for a list of (predicted, actual) pairs."""
    n = len(pairs)
    if n == 0:
        return CalibrationMetrics(
            metric_name=name,
            n_samples=0,
            predicted_mean=0,
            actual_mean=0,
            mse=0,
            mae=0,
            bias=0,
        )

    preds = [p for p, _ in pairs]
    actuals = [a for _, a in pairs]

    pred_mean = sum(preds) / n
    actual_mean = sum(actuals) / n

    # MSE
    mse = sum((p - a) ** 2 for p, a in pairs) / n

    # MAE
    mae = sum(abs(p - a) for p, a in pairs) / n

    # Bias (positive = overconfident)
    bias = pred_mean - actual_mean

    # Correlation (requires >= 3 non-constant samples)
    correlation = None
    if n >= 3:
        pred_std = (sum((p - pred_mean) ** 2 for p in preds) / n) ** 0.5
        actual_std = (sum((a - actual_mean) ** 2 for a in actuals) / n) ** 0.5

        if pred_std > 0 and actual_std > 0:
            cov = sum((p - pred_mean) * (a - actual_mean) for p, a in pairs) / n
            correlation = cov / (pred_std * actual_std)

    return CalibrationMetrics(
        metric_name=name,
        n_samples=n,
        predicted_mean=pred_mean,
        actual_mean=actual_mean,
        mse=mse,
        mae=mae,
        bias=bias,
        correlation=correlation,
    )


def compute_calibration(
    db: ExperimentDB,
    min_samples: int = 10,
) -> CalibrationSnapshot:
    """
    Compute calibration metrics from experiment database.

    Args:
        db: Experiment database
        min_samples: Minimum samples required per metric

    Returns:
        CalibrationSnapshot with all metrics
    """
    data = db.get_calibration_data()

    if not data:
        return CalibrationSnapshot(
            timestamp=datetime.now(),
            metrics={},
            total_runs=0,
            pass_rate=0.0,
        )

    # Extract score pairs for each metric
    metric_pairs: dict[str, list[tuple[float, float]]] = {}

    ci_actuals = []

    for pred, actual in data:
        # Numeric dimensions
        for dim in ["rac_reviewer", "formula_reviewer", "parameter_reviewer", "integration_reviewer"]:
            pred_val = getattr(pred, dim, None)
            act_val = getattr(actual, dim, None)
            if pred_val is not None and act_val is not None:
                metric_pairs.setdefault(dim, []).append((pred_val, act_val))

        # Optional oracle dimensions
        for dim in ["policyengine_match", "taxsim_match"]:
            pred_val = getattr(pred, dim, None)
            act_val = getattr(actual, dim, None)
            if pred_val is not None and act_val is not None:
                metric_pairs.setdefault(dim, []).append((pred_val, act_val))

        # Track CI pass rate
        if actual.ci_pass is not None:
            ci_actuals.append(actual.ci_pass)

    # Compute metrics for each score type (only if enough samples)
    metrics = {}
    for name, pairs in metric_pairs.items():
        if len(pairs) >= min_samples:
            metrics[name] = _compute_metric(name, pairs)

    # Overall pass rate
    pass_rate = 0.0
    if ci_actuals:
        pass_rate = sum(1 for p in ci_actuals if p) / len(ci_actuals)

    return CalibrationSnapshot(
        timestamp=datetime.now(),
        metrics=metrics,
        total_runs=len(data),
        pass_rate=pass_rate,
    )


def print_calibration_report(snapshot: CalibrationSnapshot) -> str:
    """Generate human-readable calibration report."""
    lines = [
        "=" * 60,
        "CALIBRATION REPORT",
        f"Generated: {snapshot.timestamp.isoformat()}",
        f"Total Runs: {snapshot.total_runs}",
        f"Pass Rate: {snapshot.pass_rate * 100:.1f}%",
        "=" * 60,
        "",
    ]

    if not snapshot.metrics:
        lines.append("No calibration data available yet.")
        return "\n".join(lines)

    # Header
    lines.append(f"{'Metric':<25} {'N':>5} {'Pred':>7} {'Actual':>7} {'Bias':>7} {'MSE':>7}")
    lines.append("-" * 60)

    for name, m in sorted(snapshot.metrics.items()):
        bias_str = f"+{m.bias:.3f}" if m.bias > 0 else f"{m.bias:.3f}"
        lines.append(
            f"{name:<25} {m.n_samples:>5} {m.predicted_mean:>7.2f} "
            f"{m.actual_mean:>7.2f} {bias_str:>7} {m.mse:>7.4f}"
        )

    lines.append("")
    lines.append("Interpretation:")
    lines.append("  Bias > 0: Agent overconfident (predicts higher than actual)")
    lines.append("  Bias < 0: Agent underconfident")
    lines.append("  Lower MSE = better calibration")

    return "\n".join(lines)


def save_calibration_snapshot(
    db_path: Path,
    snapshot: CalibrationSnapshot,
) -> None:
    """Save calibration snapshot to database for trend analysis."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Ensure table exists
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

    for name, m in snapshot.metrics.items():
        cursor.execute("""
            INSERT INTO calibration_snapshots (
                id, timestamp, metric_name, predicted_mean, actual_mean, mse, n_samples
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            f"{snapshot.timestamp.isoformat()}_{name}",
            snapshot.timestamp.isoformat(),
            name,
            m.predicted_mean,
            m.actual_mean,
            m.mse,
            m.n_samples,
        ))

    conn.commit()
    conn.close()


def get_calibration_trend(
    db_path: Path,
    metric_name: str,
    limit: int = 30,
) -> list[tuple[datetime, float, float]]:
    """
    Get calibration trend over time for a specific metric.

    Returns list of (timestamp, predicted_mean, actual_mean).
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Ensure table exists
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
    conn.commit()

    cursor.execute("""
        SELECT timestamp, predicted_mean, actual_mean
        FROM calibration_snapshots
        WHERE metric_name = ?
        ORDER BY timestamp DESC
        LIMIT ?
    """, (metric_name, limit))

    rows = cursor.fetchall()
    conn.close()

    return [
        (datetime.fromisoformat(ts), pred, actual)
        for ts, pred, actual in rows
    ]
