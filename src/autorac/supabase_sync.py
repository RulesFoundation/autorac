"""
Supabase sync for autorac encoding runs.

Syncs local SQLite experiment DB to Supabase for the public dashboard.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

from supabase import create_client, Client


def get_supabase_client() -> Client:
    """Get Supabase client using environment variables."""
    url = os.environ.get("COSILICO_SUPABASE_URL")
    # Try service role key first (for writes), fall back to anon key (reads only)
    key = os.environ.get("COSILICO_SUPABASE_SECRET_KEY") or os.environ.get("COSILICO_SUPABASE_ANON_KEY")

    if not url or not key:
        raise ValueError(
            "Missing Supabase credentials. Set COSILICO_SUPABASE_URL and "
            "COSILICO_SUPABASE_SECRET_KEY (or COSILICO_SUPABASE_ANON_KEY for read-only)."
        )

    return create_client(url, key)


def sync_run_to_supabase(
    run: "EncodingRun",
    data_source: str,  # REQUIRED: 'reviewer_agent', 'ci_only', 'mock', 'manual_estimate'
    client: Optional[Client] = None,
) -> bool:
    """
    Sync a single encoding run to Supabase.

    Args:
        run: The EncodingRun to sync
        data_source: REQUIRED - one of 'reviewer_agent', 'ci_only', 'mock', 'manual_estimate'
        client: Optional Supabase client (creates one if not provided)

    Returns:
        True if sync succeeded
    """
    from .harness.experiment_db import EncodingRun  # Avoid circular import

    # Validate data_source - this is a hard requirement to prevent fake data
    valid_sources = {'reviewer_agent', 'ci_only', 'mock', 'manual_estimate'}
    if data_source not in valid_sources:
        raise ValueError(f"data_source must be one of {valid_sources}, got: {data_source}")

    if client is None:
        client = get_supabase_client()

    # Convert to Supabase format
    data = {
        "id": run.id,
        "timestamp": run.timestamp.isoformat(),
        "citation": run.citation,
        "file_path": run.file_path,
        "complexity": {
            "cross_references": run.complexity.cross_references,
            "has_nested_structure": run.complexity.has_nested_structure,
            "has_numeric_thresholds": run.complexity.has_numeric_thresholds,
            "has_phase_in_out": run.complexity.has_phase_in_out,
            "estimated_variables": run.complexity.estimated_variables,
            "estimated_parameters": run.complexity.estimated_parameters,
        },
        "iterations": [
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
        ],
        "total_duration_ms": run.total_duration_ms,
        "agent_type": run.agent_type,
        "agent_model": run.agent_model,
        "rac_content": run.rac_content,
        "session_id": run.session_id,
        "synced_at": datetime.now().isoformat(),
        "data_source": data_source,
    }

    # Add predicted scores if present
    if run.predicted_scores:
        data["predicted_scores"] = {
            "rac": run.predicted_scores.rac,
            "formula": run.predicted_scores.formula,
            "param": run.predicted_scores.param,
            "integration": run.predicted_scores.integration,
            "iterations": run.predicted_scores.iterations,
            "time_minutes": run.predicted_scores.time_minutes,
            "confidence": run.predicted_scores.confidence,
        }

    # Add final scores if present
    if run.final_scores:
        data["final_scores"] = {
            "rac_reviewer": run.final_scores.rac_reviewer,
            "formula_reviewer": run.final_scores.formula_reviewer,
            "parameter_reviewer": run.final_scores.parameter_reviewer,
            "integration_reviewer": run.final_scores.integration_reviewer,
            "policyengine_match": run.final_scores.policyengine_match,
            "taxsim_match": run.final_scores.taxsim_match,
        }

    try:
        # Upsert to handle both new and updated runs
        result = client.table("encoding_runs").upsert(data).execute()
        return len(result.data) > 0
    except Exception as e:
        print(f"Error syncing run {run.id}: {e}")
        return False


def sync_all_runs(db_path: Path, data_source: str, client: Optional[Client] = None) -> dict:
    """
    Sync all runs from local SQLite to Supabase.

    Args:
        db_path: Path to local experiments.db
        data_source: REQUIRED - one of 'reviewer_agent', 'ci_only', 'mock', 'manual_estimate'
        client: Optional Supabase client

    Returns:
        Dict with sync stats
    """
    from .harness.experiment_db import ExperimentDB

    if client is None:
        client = get_supabase_client()

    db = ExperimentDB(db_path)
    runs = db.get_recent_runs(limit=1000)  # Get all runs

    synced = 0
    failed = 0

    for run in runs:
        if sync_run_to_supabase(run, data_source, client):
            synced += 1
        else:
            failed += 1

    return {
        "total": len(runs),
        "synced": synced,
        "failed": failed,
    }


def fetch_runs_from_supabase(
    limit: int = 20,
    citation: Optional[str] = None,
    client: Optional[Client] = None,
) -> list[dict]:
    """
    Fetch encoding runs from Supabase.

    Args:
        limit: Maximum runs to fetch
        citation: Optional filter by citation
        client: Optional Supabase client

    Returns:
        List of run records
    """
    if client is None:
        client = get_supabase_client()

    query = client.table("encoding_runs").select("*")

    if citation:
        query = query.eq("citation", citation)

    query = query.order("timestamp", desc=True).limit(limit)

    result = query.execute()
    return result.data


if __name__ == "__main__":
    # CLI usage: python -m autorac.supabase_sync <db_path> <data_source>
    # data_source is REQUIRED to prevent syncing fake data
    import sys

    if len(sys.argv) < 3:
        print("Usage: python -m autorac.supabase_sync <db_path> <data_source>")
        print("")
        print("data_source must be one of:")
        print("  reviewer_agent  - Scores from actual reviewer agent runs")
        print("  ci_only         - Only CI tests ran, no reviewer scores")
        print("  mock            - Fake/placeholder data for testing")
        print("  manual_estimate - Human-estimated scores (NOT from agents)")
        print("")
        print("Example: python -m autorac.supabase_sync experiments.db ci_only")
        sys.exit(1)

    db_path = Path(sys.argv[1])
    data_source = sys.argv[2]

    print(f"Syncing from {db_path} with data_source={data_source}...")
    stats = sync_all_runs(db_path, data_source)
    print(f"Done! {stats['synced']} synced, {stats['failed']} failed of {stats['total']} total")
