"""Tests for encoding context injection in Orchestrator prompts."""

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from autorac.harness.orchestrator import Backend, Orchestrator, SubsectionTask


@pytest.fixture
def cli_orchestrator(temp_db_path):
    """CLI-backend orchestrator with a temp DB."""
    return Orchestrator(backend=Backend.CLI, db_path=temp_db_path)


@pytest.fixture
def api_orchestrator():
    """API-backend orchestrator (no DB needed for this test)."""
    return Orchestrator(backend=Backend.API, api_key="test-key")


class TestBuildContextSection:
    def test_cli_backend_returns_context(self, cli_orchestrator):
        result = cli_orchestrator._build_context_section()
        assert "Past encoding reference" in result
        assert "encoding_runs" in result
        assert "sqlite3" in result
        assert ".rac files" in result

    def test_cli_backend_includes_db_path(self, cli_orchestrator):
        result = cli_orchestrator._build_context_section()
        assert str(cli_orchestrator.encoding_db.db_path) in result

    def test_api_backend_returns_empty(self, api_orchestrator):
        result = api_orchestrator._build_context_section()
        assert result == ""

    def test_cli_no_db_uses_default_path(self):
        orch = Orchestrator(backend=Backend.CLI)
        result = orch._build_context_section()
        assert "encodings.db" in result
        assert "Past encoding reference" in result


class TestContextInPrompts:
    def test_subsection_prompt_includes_context_cli(self, cli_orchestrator):
        task = SubsectionTask(
            subsection_id="(a)",
            title="Allowance of credit",
            file_name="a.rac",
            dependencies=[],
        )
        prompt = cli_orchestrator._build_subsection_prompt(
            task=task,
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text="Test statute text",
        )
        assert "Past encoding reference" in prompt

    def test_fallback_prompt_includes_context_cli(self, cli_orchestrator):
        prompt = cli_orchestrator._build_fallback_encode_prompt(
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text="Test statute text",
        )
        assert "Past encoding reference" in prompt

    def test_subsection_prompt_no_context_api(self, api_orchestrator):
        task = SubsectionTask(
            subsection_id="(a)",
            title="Allowance of credit",
            file_name="a.rac",
            dependencies=[],
        )
        prompt = api_orchestrator._build_subsection_prompt(
            task=task,
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text="Test statute text",
        )
        assert "Past encoding reference" not in prompt

    def test_fallback_prompt_no_context_api(self, api_orchestrator):
        prompt = api_orchestrator._build_fallback_encode_prompt(
            citation="26 USC 21",
            output_path=Path("/tmp/test"),
            statute_text="Test statute text",
        )
        assert "Past encoding reference" not in prompt
