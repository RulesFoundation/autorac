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
)
from .harness.validator_pipeline import ValidatorPipeline


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
    log_parser.add_argument("--scores", type=str, help="Final scores as JSON")
    log_parser.add_argument("--db", type=Path, default=Path("experiments.db"))

    # stats command
    stats_parser = subparsers.add_parser(
        "stats",
        help="Show encoding statistics"
    )
    stats_parser.add_argument("--db", type=Path, default=Path("experiments.db"))

    # runs command
    runs_parser = subparsers.add_parser(
        "runs",
        help="List recent encoding runs"
    )
    runs_parser.add_argument("--db", type=Path, default=Path("experiments.db"))
    runs_parser.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()

    if args.command == "validate":
        cmd_validate(args)
    elif args.command == "log":
        cmd_log(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "runs":
        cmd_runs(args)
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
            "ci_pass": scores.ci_pass,
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
        print(f"CI: {'✓' if scores.ci_pass else '✗'}")
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

    # Parse scores
    final_scores = None
    if args.scores:
        s = json.loads(args.scores)
        final_scores = FinalScores(
            rac_reviewer=s.get("rac", 0),
            formula_reviewer=s.get("formula", 0),
            parameter_reviewer=s.get("param", 0),
            integration_reviewer=s.get("integration", 0),
        )

    # Read RAC content
    rac_content = ""
    if args.file.exists():
        rac_content = args.file.read_text()

    run = EncodingRun(
        citation=args.citation,
        file_path=str(args.file),
        iterations=iterations,
        total_duration_ms=args.duration,
        final_scores=final_scores,
        rac_content=rac_content,
    )

    db.log_run(run)

    print(f"Logged: {run.id}")
    print(f"  Citation: {args.citation}")
    print(f"  Iterations: {args.iterations}")
    print(f"  Duration: {args.duration}ms")
    if final_scores:
        print(f"  Scores: RAC {final_scores.rac_reviewer}/10")


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


if __name__ == "__main__":
    main()
