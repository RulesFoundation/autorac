"""
AutoRAC CLI - Command line interface for encoding experiments.
"""

import argparse
import sys
from pathlib import Path

from . import (
    ExperimentDB,
    EncoderHarness,
    EncoderConfig,
    compute_calibration,
    print_calibration_report,
)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AutoRAC - AI-assisted RAC encoding infrastructure"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

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

    # encode command
    encode_parser = subparsers.add_parser(
        "encode",
        help="Run encoding experiment"
    )
    encode_parser.add_argument(
        "citation",
        help="Legal citation (e.g., '26 USC 32(a)(1)')"
    )
    encode_parser.add_argument(
        "--statute-file",
        type=Path,
        help="Path to statute text file"
    )
    encode_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Output directory for RAC file"
    )
    encode_parser.add_argument(
        "--db",
        type=Path,
        default=Path("experiments.db"),
        help="Path to experiments database"
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

    if args.command == "calibration":
        cmd_calibration(args)
    elif args.command == "encode":
        cmd_encode(args)
    elif args.command == "runs":
        cmd_runs(args)
    else:
        parser.print_help()
        sys.exit(1)


def cmd_calibration(args):
    """Show calibration metrics."""
    if not args.db.exists():
        print(f"Database not found: {args.db}")
        sys.exit(1)

    db = ExperimentDB(args.db)
    snapshot = compute_calibration(db, min_samples=args.min_samples)
    report = print_calibration_report(snapshot)
    print(report)


def cmd_encode(args):
    """Run an encoding experiment."""
    statute_text = ""
    if args.statute_file:
        statute_text = args.statute_file.read_text()

    # Auto-detect rac paths
    output_dir = args.output_dir.resolve()
    rac_us = output_dir
    while rac_us.name != "rac-us" and rac_us.parent != rac_us:
        rac_us = rac_us.parent

    if rac_us.name != "rac-us":
        print("Error: Could not find rac-us directory")
        sys.exit(1)

    config = EncoderConfig(
        rac_us_path=rac_us,
        rac_path=rac_us.parent / "rac",
        db_path=args.db,
    )

    harness = EncoderHarness(config)

    # Derive output path from citation
    parts = args.citation.replace("USC", "").replace("(", "/").replace(")", "").split()
    title = parts[0]
    rest = "".join(parts[1:])
    output_path = output_dir / f"statute/{title}/{rest}.rac"

    print(f"Encoding: {args.citation}")
    print(f"Output: {output_path}")

    iterations = harness.iterate_until_pass(
        citation=args.citation,
        statute_text=statute_text,
        output_path=output_path,
    )

    for i, (run, result) in enumerate(iterations, 1):
        print(f"\nIteration {i}:")
        print(f"  All passed: {result.all_passed}")
        if run.predicted:
            print(f"  Predicted RAC score: {run.predicted.rac_reviewer}")
        if run.actual:
            print(f"  Actual RAC score: {run.actual.rac_reviewer}")


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
