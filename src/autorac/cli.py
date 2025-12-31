"""
AutoRAC CLI - Command line interface for encoding experiments.

Primary workflow:
  1. /encode (slash command) invokes RAC Encoder agent
  2. Agent iterates until CI passes, tracking errors and fixes
  3. autorac validate <file.rac> runs final validation
  4. autorac log records the full journey
  5. autorac stats shows patterns for improvement
"""

import argparse
import json
import sys
from pathlib import Path

from .harness.experiment_db import (
    ExperimentDB,
    EncodingRun,
    ComplexityFactors,
    Iteration,
    IterationError,
    FinalScores,
    PredictedScores,
    Session,
    SessionEvent,
)
from .harness.validator_pipeline import ValidatorPipeline

# Default DB path - can be overridden with --db
DEFAULT_DB = Path.home() / "CosilicoAI" / "autorac" / "experiments.db"


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AutoRAC - AI-assisted RAC encoding infrastructure"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a .rac file (CI + reviewer agents)"
    )
    validate_parser.add_argument("file", type=Path, help="Path to .rac file")
    validate_parser.add_argument("--json", action="store_true", help="Output as JSON")
    validate_parser.add_argument("--skip-reviewers", action="store_true")

    # log command
    log_parser = subparsers.add_parser(
        "log",
        help="Log an encoding run to experiment DB"
    )
    log_parser.add_argument("--citation", required=True, help="Legal citation")
    log_parser.add_argument("--file", type=Path, required=True, help="Path to .rac file")
    log_parser.add_argument("--iterations", type=int, default=1, help="Number of iterations")
    log_parser.add_argument("--errors", type=str, default="[]", help="Errors as JSON array")
    log_parser.add_argument("--duration", type=int, default=0, help="Total duration in ms")
    log_parser.add_argument("--scores", type=str, help="Actual scores as JSON {rac,formula,param,integration}")
    log_parser.add_argument("--predicted", type=str, help="Predicted scores as JSON {rac,formula,param,integration,iterations,time}")
    log_parser.add_argument("--session", type=str, help="Session ID to link this run to")
    log_parser.add_argument("--db", type=Path, default=Path("experiments.db"))

    # stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show encoding statistics"
    )
    stats_parser.add_argument("--db", type=Path, default=Path("experiments.db"))

    # calibration command
    calibration_parser = subparsers.add_parser(
        "calibration",
        help="Show calibration metrics (predicted vs actual)"
    )
    calibration_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    calibration_parser.add_argument("--limit", type=int, default=50)

    # runs command
    runs_parser = subparsers.add_parser(
        "runs",
        help="List recent encoding runs"
    )
    runs_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    runs_parser.add_argument("--limit", type=int, default=20)

    # =========================================================================
    # Session logging commands (for hooks)
    # =========================================================================

    # session-start command
    session_start_parser = subparsers.add_parser(
        "session-start",
        help="Start a new session (called by SessionStart hook)"
    )
    session_start_parser.add_argument("--model", default="", help="Model name")
    session_start_parser.add_argument("--cwd", default="", help="Working directory")
    session_start_parser.add_argument("--db", type=Path, default=DEFAULT_DB)

    # session-end command
    session_end_parser = subparsers.add_parser(
        "session-end",
        help="End a session (called by SessionEnd hook)"
    )
    session_end_parser.add_argument("--session", required=True, help="Session ID")
    session_end_parser.add_argument("--db", type=Path, default=DEFAULT_DB)

    # log-event command
    log_event_parser = subparsers.add_parser(
        "log-event",
        help="Log an event to a session (called by hooks)"
    )
    log_event_parser.add_argument("--session", required=True, help="Session ID")
    log_event_parser.add_argument("--type", required=True, help="Event type")
    log_event_parser.add_argument("--tool", default=None, help="Tool name (for tool events)")
    log_event_parser.add_argument("--content", default="", help="Event content")
    log_event_parser.add_argument("--metadata", default="{}", help="Metadata as JSON")
    log_event_parser.add_argument("--db", type=Path, default=DEFAULT_DB)

    # sessions command
    sessions_parser = subparsers.add_parser(
        "sessions",
        help="List recent sessions"
    )
    sessions_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    sessions_parser.add_argument("--limit", type=int, default=20)

    # session-show command
    session_show_parser = subparsers.add_parser(
        "session-show",
        help="Show a session transcript"
    )
    session_show_parser.add_argument("session_id", help="Session ID")
    session_show_parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    session_show_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # session-stats command
    session_stats_parser = subparsers.add_parser(
        "session-stats",
        help="Show session statistics"
    )
    session_stats_parser.add_argument("--db", type=Path, default=DEFAULT_DB)

    args = parser.parse_args()

    if args.command == "validate":
        cmd_validate(args)
    elif args.command == "log":
        cmd_log(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "calibration":
        cmd_calibration(args)
    elif args.command == "runs":
        cmd_runs(args)
    elif args.command == "session-start":
        cmd_session_start(args)
    elif args.command == "session-end":
        cmd_session_end(args)
    elif args.command == "log-event":
        cmd_log_event(args)
    elif args.command == "sessions":
        cmd_sessions(args)
    elif args.command == "session-show":
        cmd_session_show(args)
    elif args.command == "session-stats":
        cmd_session_stats(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_validate(args):
    """Validate a .rac file."""
    if not args.file.exists():
        print(f"File not found: {args.file}")
        sys.exit(1)

    rac_file = args.file.resolve()

    # Find rac paths
    rac_us = rac_file.parent
    while rac_us.name != "rac-us" and rac_us.parent != rac_us:
        rac_us = rac_us.parent

    if rac_us.name != "rac-us":
        rac_us = Path.home() / "CosilicoAI" / "rac-us"
        rac_path = Path.home() / "CosilicoAI" / "rac"
    else:
        rac_path = rac_us.parent / "rac"

    pipeline = ValidatorPipeline(
        rac_us_path=rac_us,
        rac_path=rac_path,
        enable_oracles=False,
    )

    result = pipeline.validate(rac_file)
    scores = result.to_actual_scores()

    errors = []
    for name, vr in result.results.items():
        if vr.error:
            errors.append(f"{name}: {vr.error}")

    if args.json:
        output = {
            "file": str(rac_file),
            "ci_pass": result.ci_pass,
            "scores": {
                "rac_reviewer": scores.rac_reviewer,
                "formula_reviewer": scores.formula_reviewer,
                "parameter_reviewer": scores.parameter_reviewer,
                "integration_reviewer": scores.integration_reviewer,
            },
            "all_passed": result.all_passed,
            "errors": errors,
            "duration_ms": result.total_duration_ms,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"File: {rac_file}")
        print(f"CI: {'✓' if result.ci_pass else '✗'}")
        if not args.skip_reviewers:
            print(f"Scores: RAC {scores.rac_reviewer}/10 | Formula {scores.formula_reviewer}/10 | Param {scores.parameter_reviewer}/10 | Integration {scores.integration_reviewer}/10")
        print(f"Result: {'✓ PASSED' if result.all_passed else '✗ FAILED'}")
        if errors:
            for err in errors:
                print(f"  - {err}")

    sys.exit(0 if result.all_passed else 1)


def cmd_log(args):
    """Log an encoding run."""
    db = ExperimentDB(args.db)

    # Parse errors
    errors_data = json.loads(args.errors) if args.errors else []
    iteration_errors = [
        IterationError(
            error_type=e.get("type", "other"),
            message=e.get("message", ""),
            variable=e.get("variable"),
            fix_applied=e.get("fix"),
        )
        for e in errors_data
    ]

    # Build iterations (simplified: all errors in iteration 1, success in last)
    iterations = []
    for i in range(1, args.iterations + 1):
        is_last = i == args.iterations
        iterations.append(Iteration(
            attempt=i,
            duration_ms=args.duration // args.iterations,
            errors=iteration_errors if i == 1 else [],
            success=is_last,
        ))

    # Parse actual scores
    final_scores = None
    if args.scores:
        s = json.loads(args.scores)
        final_scores = FinalScores(
            rac_reviewer=s.get("rac", 0),
            formula_reviewer=s.get("formula", 0),
            parameter_reviewer=s.get("param", 0),
            integration_reviewer=s.get("integration", 0),
        )

    # Parse predicted scores (for calibration)
    predicted_scores = None
    if args.predicted:
        p = json.loads(args.predicted)
        predicted_scores = PredictedScores(
            rac=p.get("rac", 0),
            formula=p.get("formula", 0),
            param=p.get("param", 0),
            integration=p.get("integration", 0),
            iterations=p.get("iterations", 1),
            time_minutes=p.get("time", 0),
            confidence=p.get("confidence", 0.5),
        )

    # Read RAC content
    rac_content = ""
    if args.file.exists():
        rac_content = args.file.read_text()

    run = EncodingRun(
        citation=args.citation,
        file_path=str(args.file),
        predicted_scores=predicted_scores,
        iterations=iterations,
        total_duration_ms=args.duration,
        final_scores=final_scores,
        rac_content=rac_content,
        session_id=args.session,
    )

    db.log_run(run)

    print(f"Logged: {run.id}")
    print(f"  Citation: {args.citation}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Duration: {args.duration}ms")
    if args.session:
        print(f"  Session: {args.session}")
    if predicted_scores:
        print(f"  Predicted: RAC {predicted_scores.rac}/10 | Formula {predicted_scores.formula}/10 | Param {predicted_scores.param}/10 | Iter {predicted_scores.iterations}")
    if final_scores:
        print(f"  Actual: RAC {final_scores.rac_reviewer}/10 | Formula {final_scores.formula_reviewer}/10 | Param {final_scores.parameter_reviewer}/10")


def cmd_stats(args):
    """Show encoding statistics."""
    if not args.db.exists():
        print(f"Database not found: {args.db}")
        print("Run some encodings first to collect data.")
        sys.exit(1)

    db = ExperimentDB(args.db)

    # Iteration stats
    iter_stats = db.get_iteration_stats()
    print("=== Iteration Statistics ===")
    print(f"Total runs: {iter_stats['total_runs']}")
    print(f"Average iterations: {iter_stats['average']:.1f}")
    print(f"First-try success rate: {iter_stats['first_try_rate']:.0f}%")
    print(f"Distribution: {iter_stats['distribution']}")
    print()

    # Error stats
    error_stats = db.get_error_stats()
    print("=== Error Statistics ===")
    print(f"Total errors: {error_stats['total_errors']}")
    if error_stats['counts']:
        print("By type:")
        for error_type, count in sorted(error_stats['counts'].items(), key=lambda x: -x[1]):
            pct = error_stats['percentages'][error_type]
            print(f"  {error_type}: {count} ({pct:.0f}%)")
    print()

    # Improvement suggestions
    print("=== Improvement Suggestions ===")
    if error_stats['counts']:
        top_error = max(error_stats['counts'].items(), key=lambda x: x[1])
        print(f"Focus on: {top_error[0]} errors ({top_error[1]} occurrences)")
        if top_error[0] == "test":
            print("  → Add more test examples to RAC_SPEC.md")
        elif top_error[0] == "parse":
            print("  → Clarify syntax in RAC_SPEC.md")
        elif top_error[0] == "import":
            print("  → Document import patterns better")
    else:
        print("Not enough data yet. Run more encodings.")


def cmd_calibration(args):
    """Show calibration metrics - predicted vs actual scores."""
    if not args.db.exists():
        print(f"Database not found: {args.db}")
        sys.exit(1)

    db = ExperimentDB(args.db)
    runs = db.get_recent_runs(limit=args.limit)

    # Filter to runs with predictions
    runs_with_pred = [r for r in runs if r.predicted_scores and r.final_scores]

    if not runs_with_pred:
        print("No runs with both predictions and actual scores yet.")
        print("Use --predicted flag when logging runs to enable calibration.")
        return

    print("=== Calibration Report ===\n")
    print(f"Runs with predictions: {len(runs_with_pred)}")
    print()

    # Calculate per-dimension errors
    errors = {"rac": [], "formula": [], "param": [], "integration": []}
    iter_errors = []
    time_errors = []

    for run in runs_with_pred:
        p = run.predicted_scores
        a = run.final_scores

        errors["rac"].append(p.rac - a.rac_reviewer)
        errors["formula"].append(p.formula - a.formula_reviewer)
        errors["param"].append(p.param - a.parameter_reviewer)
        errors["integration"].append(p.integration - a.integration_reviewer)

        # Iteration prediction error
        iter_errors.append(p.iterations - run.iterations_needed)

        # Time prediction error (in minutes)
        actual_time = run.total_duration_ms / 60000
        if p.time_minutes > 0:
            time_errors.append(p.time_minutes - actual_time)

    # Print dimension calibration
    print("Dimension Calibration (predicted - actual):")
    print("-" * 50)
    print(f"{'Dimension':<15} {'Mean Err':>10} {'Bias':>10} {'MAE':>10}")
    print("-" * 50)

    for dim, errs in errors.items():
        if errs:
            mean_err = sum(errs) / len(errs)
            bias = "over" if mean_err > 0.5 else "under" if mean_err < -0.5 else "good"
            mae = sum(abs(e) for e in errs) / len(errs)
            print(f"{dim:<15} {mean_err:>+10.1f} {bias:>10} {mae:>10.1f}")

    print()

    # Print iteration calibration
    if iter_errors:
        mean_iter_err = sum(iter_errors) / len(iter_errors)
        iter_bias = "over" if mean_iter_err > 0.3 else "under" if mean_iter_err < -0.3 else "good"
        print(f"Iteration prediction: mean error {mean_iter_err:+.1f} ({iter_bias})")

    # Print time calibration
    if time_errors:
        mean_time_err = sum(time_errors) / len(time_errors)
        time_bias = "over" if mean_time_err > 2 else "under" if mean_time_err < -2 else "good"
        print(f"Time prediction: mean error {mean_time_err:+.1f} min ({time_bias})")

    print()

    # Per-run breakdown
    print("Per-Run Breakdown:")
    print("-" * 70)
    print(f"{'Citation':<25} {'Pred':>8} {'Act':>8} {'Err':>8} {'Iter':>6}")
    print("-" * 70)

    for run in runs_with_pred[-10:]:  # Last 10
        p = run.predicted_scores
        a = run.final_scores
        pred_avg = (p.rac + p.formula + p.param + p.integration) / 4
        act_avg = (a.rac_reviewer + a.formula_reviewer + a.parameter_reviewer + a.integration_reviewer) / 4
        err = pred_avg - act_avg
        citation = run.citation[:25]
        print(f"{citation:<25} {pred_avg:>8.1f} {act_avg:>8.1f} {err:>+8.1f} {run.iterations_needed:>6}")


def cmd_runs(args):
    """List recent runs."""
    if not args.db.exists():
        print(f"Database not found: {args.db}")
        sys.exit(1)

    db = ExperimentDB(args.db)
    runs = db.get_recent_runs(limit=args.limit)

    if not runs:
        print("No encoding runs found.")
        return

    print(f"{'ID':<10} {'Citation':<30} {'Iter':<5} {'Time':<8} {'Result'}")
    print("-" * 70)

    for run in runs:
        result = "✓" if run.success else "✗"
        time_s = run.total_duration_ms / 1000
        print(f"{run.id:<10} {run.citation:<30} {run.iterations_needed:<5} {time_s:>6.1f}s {result}")


# =========================================================================
# Session Commands
# =========================================================================

def cmd_session_start(args):
    """Start a new session."""
    db = ExperimentDB(args.db)
    session = db.start_session(model=args.model, cwd=args.cwd or str(Path.cwd()))

    # Output just the session ID for hooks to capture
    print(session.id)


def cmd_session_end(args):
    """End a session."""
    db = ExperimentDB(args.db)
    db.end_session(args.session)
    print(f"Session {args.session} ended")


def cmd_log_event(args):
    """Log an event to a session."""
    db = ExperimentDB(args.db)

    metadata = {}
    if args.metadata:
        try:
            metadata = json.loads(args.metadata)
        except json.JSONDecodeError:
            pass

    event = db.log_event(
        session_id=args.session,
        event_type=args.type,
        tool_name=args.tool,
        content=args.content,
        metadata=metadata,
    )

    print(f"Event {event.sequence}: {event.event_type}")


def cmd_sessions(args):
    """List recent sessions."""
    db = ExperimentDB(args.db)
    sessions = db.get_recent_sessions(limit=args.limit)

    if not sessions:
        print("No sessions found.")
        return

    print(f"{'ID':<10} {'Started':<20} {'Events':<8} {'Model':<15} {'Status'}")
    print("-" * 70)

    for s in sessions:
        started = s.started_at.strftime("%Y-%m-%d %H:%M") if s.started_at else "?"
        status = "ended" if s.ended_at else "active"
        model = s.model[:15] if s.model else "-"
        print(f"{s.id:<10} {started:<20} {s.event_count:<8} {model:<15} {status}")


def cmd_session_show(args):
    """Show a session transcript."""
    db = ExperimentDB(args.db)

    session = db.get_session(args.session_id)
    if not session:
        print(f"Session not found: {args.session_id}")
        sys.exit(1)

    events = db.get_session_events(args.session_id)

    if args.json:
        output = {
            "session": {
                "id": session.id,
                "started_at": session.started_at.isoformat() if session.started_at else None,
                "ended_at": session.ended_at.isoformat() if session.ended_at else None,
                "model": session.model,
                "cwd": session.cwd,
                "event_count": session.event_count,
            },
            "events": [
                {
                    "sequence": e.sequence,
                    "timestamp": e.timestamp.isoformat() if e.timestamp else None,
                    "type": e.event_type,
                    "tool": e.tool_name,
                    "content": e.content[:500] if e.content else "",  # Truncate long content
                    "metadata": e.metadata,
                }
                for e in events
            ]
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Session: {session.id}")
        print(f"Model: {session.model}")
        print(f"Started: {session.started_at}")
        print(f"Ended: {session.ended_at or 'active'}")
        print(f"Events: {session.event_count}")
        print("-" * 60)

        for e in events:
            time_str = e.timestamp.strftime("%H:%M:%S") if e.timestamp else "?"
            tool_str = f" [{e.tool_name}]" if e.tool_name else ""
            content_preview = (e.content[:80] + "...") if e.content and len(e.content) > 80 else (e.content or "")

            print(f"{e.sequence:3}. [{time_str}] {e.event_type}{tool_str}")
            if content_preview:
                print(f"     {content_preview}")


def cmd_session_stats(args):
    """Show session statistics."""
    db = ExperimentDB(args.db)
    stats = db.get_session_stats()

    print("=== Session Statistics ===")
    print(f"Total sessions: {stats['total_sessions']}")
    print(f"Avg events/session: {stats['avg_events_per_session']}")
    print()

    if stats['event_type_counts']:
        print("Event types:")
        for event_type, count in sorted(stats['event_type_counts'].items(), key=lambda x: -x[1]):
            print(f"  {event_type}: {count}")
        print()

    if stats['tool_usage']:
        print("Top tools:")
        for tool, count in list(stats['tool_usage'].items())[:10]:
            print(f"  {tool}: {count}")


if __name__ == "__main__":
    main()
