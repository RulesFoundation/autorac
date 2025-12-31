"""
AutoRAC CLI - Command line interface for encoding experiments.

Primary workflow:
  1. /encode (slash command) invokes RAC Encoder agent to write .rac file
  2. autorac validate <file.rac> runs CI + reviewers, outputs JSON scores
  3. autorac log records predictions vs actuals to experiment DB
  4. autorac calibration shows prediction accuracy over time
"""

import argparse
import json
import sys
from pathlib import Path

from . import (
    ExperimentDB,
    compute_calibration,
    print_calibration_report,
)
from .harness.validator_pipeline import ValidatorPipeline


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AutoRAC - AI-assisted RAC encoding infrastructure"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # validate command - run CI + reviewers on a .rac file
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate a .rac file (CI + reviewer agents)"
    )
    validate_parser.add_argument(
        "file",
        type=Path,
        help="Path to .rac file to validate"
    )
    validate_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    validate_parser.add_argument(
        "--skip-reviewers",
        action="store_true",
        help="Skip reviewer agents (CI only)"
    )

    # log command - record an encoding run to the experiment DB
    log_parser = subparsers.add_parser(
        "log",
        help="Log an encoding run to experiment DB"
    )
    log_parser.add_argument(
        "--citation",
        required=True,
        help="Legal citation (e.g., '26 USC 32(c)(2)(A)')"
    )
    log_parser.add_argument(
        "--file",
        type=Path,
        required=True,
        help="Path to .rac file"
    )
    log_parser.add_argument(
        "--predicted",
        type=str,
        help="Predicted scores as JSON (e.g., '{\"rac\":8,\"formula\":7}')"
    )
    log_parser.add_argument(
        "--actual",
        type=str,
        help="Actual scores as JSON (e.g., '{\"rac\":7,\"formula\":8}')"
    )
    log_parser.add_argument(
        "--db",
        type=Path,
        default=Path("experiments.db"),
        help="Path to experiments database"
    )

    # calibration command
    cal_parser = subparsers.add_parser(
        "calibration",
        help="View calibration metrics"
    )
    cal_parser.add_argument(
        "--db",
        type=Path,
        default=Path("experiments.db"),
        help="Path to experiments database"
    )
    cal_parser.add_argument(
        "--min-samples",
        type=int,
        default=10,
        help="Minimum samples required per metric"
    )

    # runs command
    runs_parser = subparsers.add_parser(
        "runs",
        help="List recent encoding runs"
    )
    runs_parser.add_argument(
        "--db",
        type=Path,
        default=Path("experiments.db"),
        help="Path to experiments database"
    )
    runs_parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Number of runs to show"
    )

    args = parser.parse_args()

    if args.command == "validate":
        cmd_validate(args)
    elif args.command == "log":
        cmd_log(args)
    elif args.command == "calibration":
        cmd_calibration(args)
    elif args.command == "runs":
        cmd_runs(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_validate(args):
    """Validate a .rac file with CI and reviewer agents."""
    if not args.file.exists():
        print(f"File not found: {args.file}")
        sys.exit(1)

    # Find rac repo for CI validation
    rac_file = args.file.resolve()
    rac_us = rac_file
    while rac_us.name != "rac-us" and rac_us.parent != rac_us:
        rac_us = rac_us.parent

    if rac_us.name != "rac-us":
        print("Warning: Could not find rac-us directory, CI validation may fail")
        rac_path = Path.home() / "CosilicoAI" / "rac"
    else:
        rac_path = rac_us.parent / "rac"

    # Find rac-us path
    rac_us = rac_file.parent
    while rac_us.name != "rac-us" and rac_us.parent != rac_us:
        rac_us = rac_us.parent
    if rac_us.name != "rac-us":
        rac_us = Path.home() / "CosilicoAI" / "rac-us"

    pipeline = ValidatorPipeline(
        rac_us_path=rac_us,
        rac_path=rac_path,
        enable_oracles=False,  # Skip PolicyEngine/TAXSIM for now
    )

    result = pipeline.validate(rac_file)
    scores = result.to_actual_scores()

    # Collect errors from all validators
    errors = []
    for name, vr in result.results.items():
        if vr.error:
            errors.append(f"{name}: {vr.error}")

    if args.json:
        output = {
            "file": str(rac_file),
            "ci_pass": scores.ci_pass,
            "scores": {
                "rac_reviewer": scores.rac_reviewer,
                "formula_reviewer": scores.formula_reviewer,
                "param_reviewer": scores.parameter_reviewer,
                "integration_reviewer": scores.integration_reviewer,
            },
            "all_passed": result.all_passed,
            "errors": errors,
            "duration_ms": result.total_duration_ms,
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"File: {rac_file}")
        print(f"CI Pass: {'✓' if scores.ci_pass else '✗'}")
        if not args.skip_reviewers:
            print(f"RAC Reviewer: {scores.rac_reviewer}/10")
            print(f"Formula Reviewer: {scores.formula_reviewer}/10")
            print(f"Param Reviewer: {scores.parameter_reviewer}/10")
            print(f"Integration Reviewer: {scores.integration_reviewer}/10")
        print(f"All Passed: {'✓' if result.all_passed else '✗'}")
        print(f"Duration: {result.total_duration_ms}ms")
        if errors:
            print("Errors:")
            for err in errors:
                print(f"  - {err}")

    sys.exit(0 if result.all_passed else 1)


def cmd_log(args):
    """Log an encoding run to the experiment database."""
    from .harness.experiment_db import EncodingRun, PredictedScores, ActualScores

    db = ExperimentDB(args.db)

    predicted = None
    if args.predicted:
        p = json.loads(args.predicted)
        predicted = PredictedScores(
            rac_reviewer=p.get("rac", 0),
            formula_reviewer=p.get("formula", 0),
            parameter_reviewer=p.get("param", 0),
            integration_reviewer=p.get("integration", 0),
            ci_pass=p.get("ci_pass", False),
        )

    actual = None
    if args.actual:
        a = json.loads(args.actual)
        actual = ActualScores(
            rac_reviewer=a.get("rac", 0),
            formula_reviewer=a.get("formula", 0),
            parameter_reviewer=a.get("param", 0),
            integration_reviewer=a.get("integration", 0),
            ci_pass=a.get("ci_pass", False),
        )

    import uuid
    from datetime import datetime

    # Read RAC content
    rac_content = ""
    if args.file.exists():
        rac_content = args.file.read_text()

    run = EncodingRun(
        id=str(uuid.uuid4())[:8],
        timestamp=datetime.now(),
        file_path=str(args.file),
        citation=args.citation,
        agent_type="encoder",
        agent_model="claude-opus-4-5-20251101",
        rac_content=rac_content,
        predicted=predicted,
        actual=actual,
    )

    db.log_run(run)
    print(f"Logged run: {run.id}")
    print(f"  Citation: {args.citation}")
    print(f"  File: {args.file}")
    if predicted:
        print(f"  Predicted RAC: {predicted.rac_reviewer}")
    if actual:
        print(f"  Actual RAC: {actual.rac_reviewer}")


def cmd_calibration(args):
    """Show calibration metrics."""
    if not args.db.exists():
        print(f"Database not found: {args.db}")
        sys.exit(1)

    db = ExperimentDB(args.db)
    snapshot = compute_calibration(db, min_samples=args.min_samples)
    report = print_calibration_report(snapshot)
    print(report)


def cmd_runs(args):
    """List recent encoding runs."""
    if not args.db.exists():
        print(f"Database not found: {args.db}")
        sys.exit(1)

    db = ExperimentDB(args.db)
    runs = db.get_recent_runs(limit=args.limit)

    if not runs:
        print("No encoding runs found.")
        return

    print(f"{'ID':<10} {'Citation':<25} {'Passed':<8} {'Time'}")
    print("-" * 60)

    for run in runs:
        passed = "Yes" if run.actual and run.actual.ci_pass else "No"
        print(f"{run.id:<10} {run.citation:<25} {passed:<8} {run.timestamp.isoformat()}")


if __name__ == "__main__":
    main()
