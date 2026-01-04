#!/usr/bin/env python3
"""
Encode a statute with full logging to experiment database.

This script properly passes experiment_db to the SDK orchestrator
to capture all token usage, tool calls, and agent messages.

Usage:
    python scripts/encode_with_logging.py "26 USC 1(j)(2)" --output ~/CosilicoAI/rac-us/statute

    # Or for the full section 1:
    python scripts/encode_with_logging.py "26 USC 1" --output ~/CosilicoAI/rac-us/statute
"""

import argparse
import asyncio
from datetime import datetime
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autorac.harness.experiment_db import ExperimentDB
from autorac.harness.sdk_orchestrator import SDKOrchestrator


DEFAULT_DB_PATH = Path.home() / "CosilicoAI" / "autorac" / "experiments.db"
DEFAULT_OUTPUT = Path.home() / "CosilicoAI" / "rac-us" / "statute"


async def main():
    parser = argparse.ArgumentParser(description="Encode statute with full logging")
    parser.add_argument("citation", help="Statute citation (e.g., '26 USC 1(j)(2)')")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for .rac files"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to experiment database"
    )
    parser.add_argument(
        "--model",
        default="claude-opus-4-5-20251101",
        help="Model to use for encoding"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without actually encoding"
    )

    args = parser.parse_args()

    # Parse citation to get output path
    # "26 USC 1(j)(2)" -> statute/26/1/j/2.rac
    citation = args.citation.upper().replace("USC", "").replace("§", "").strip()
    parts = citation.split()
    if len(parts) >= 2:
        title = parts[0]
        section = parts[1]
    else:
        # Try parsing format like "26/1/j/2"
        path_parts = citation.replace(" ", "/").split("/")
        title = path_parts[0]
        section = "/".join(path_parts[1:])

    output_path = args.output / title / section.replace("(", "/").replace(")", "")

    print(f"=== Encoding: {args.citation} ===")
    print(f"Output: {output_path}")
    print(f"Model: {args.model}")
    print(f"DB: {args.db}")
    print()

    if args.dry_run:
        print("[DRY RUN] Would encode with these settings")
        return

    # Initialize experiment database
    args.db.parent.mkdir(parents=True, exist_ok=True)
    experiment_db = ExperimentDB(args.db)

    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Initialize orchestrator WITH experiment_db
    orchestrator = SDKOrchestrator(
        model=args.model,
        experiment_db=experiment_db,  # CRITICAL: This enables logging!
    )

    print(f"Starting encoding at {datetime.now().strftime('%H:%M:%S')}...")
    print("-" * 60)

    # Run encoding
    run = await orchestrator.encode(
        citation=args.citation,
        output_path=output_path,
    )

    print("-" * 60)
    print()

    # Print report
    report = orchestrator.print_report(run)
    print(report)

    # Show session ID for later lookup
    print()
    print(f"Session logged: {run.session_id}")
    print(f"View with: autorac session-show {run.session_id}")

    # Summary
    print()
    print("=== SUMMARY ===")
    print(f"Files created: {len(run.files_created)}")
    if run.total_tokens:
        print(f"Total tokens: {run.total_tokens.input_tokens:,} in + {run.total_tokens.output_tokens:,} out")
        print(f"Estimated cost: ${run.total_tokens.estimated_cost_usd:.2f}")
    if run.oracle_pe_match is not None:
        print(f"PE match: {run.oracle_pe_match}%")
    if run.oracle_taxsim_match is not None:
        print(f"TAXSIM match: {run.oracle_taxsim_match}%")

    # Return exit code based on success
    has_errors = any(a.error for a in run.agent_runs)
    sys.exit(1 if has_errors else 0)


if __name__ == "__main__":
    asyncio.run(main())
