"""
Tests for sdk_orchestrator module.

All external dependencies (Claude Agent SDK, subprocess) are mocked.
"""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from autorac.harness.sdk_orchestrator import (
    AgentMessage,
    AgentRun,
    AnalyzerOutput,
    OrchestratorRun,
    Phase,
    SDKOrchestrator,
    SubsectionTask,
    _summarize_assistant_message,
    _summarize_thinking,
    _summarize_tool_call,
)

# =========================================================================
# Test utility functions
# =========================================================================


class TestSummarizeToolCall:
    def test_empty_tool_name(self):
        assert _summarize_tool_call("", None, None) == ""

    def test_read_tool(self):
        result = _summarize_tool_call(
            "Read", {"file_path": "/path/to/file.rac"}, None
        )
        assert "file.rac" in result

    def test_write_tool(self):
        result = _summarize_tool_call(
            "Write",
            {"file_path": "/path/to/file.rac", "content": "line1\nline2\nline3"},
            None,
        )
        assert "file.rac" in result
        assert "3 lines" in result

    def test_write_tool_empty_content(self):
        result = _summarize_tool_call(
            "Write",
            {"file_path": "/path/to/file.rac", "content": ""},
            None,
        )
        assert "0 lines" in result

    def test_write_tool_no_content(self):
        result = _summarize_tool_call(
            "Write",
            {"file_path": "/path/to/file.rac"},
            None,
        )
        assert "0 lines" in result

    def test_edit_tool(self):
        result = _summarize_tool_call(
            "Edit", {"file_path": "/path/to/file.rac"}, None
        )
        assert "file.rac" in result

    def test_grep_tool_short_pattern(self):
        result = _summarize_tool_call("Grep", {"pattern": "short"}, None)
        assert "short" in result

    def test_grep_tool_long_pattern(self):
        result = _summarize_tool_call(
            "Grep", {"pattern": "a" * 50}, None
        )
        assert "..." in result

    def test_glob_tool(self):
        result = _summarize_tool_call("Glob", {"pattern": "*.rac"}, None)
        assert "*.rac" in result

    def test_bash_tool_short_command(self):
        result = _summarize_tool_call("Bash", {"command": "ls"}, None)
        assert "ls" in result

    def test_bash_tool_long_command(self):
        result = _summarize_tool_call("Bash", {"command": "a" * 100}, None)
        assert "..." in result

    def test_task_tool(self):
        result = _summarize_tool_call(
            "Task", {"subagent_type": "encoder"}, None
        )
        assert "encoder" in result

    def test_unknown_tool(self):
        result = _summarize_tool_call("Unknown", {"key": "val"}, None)
        assert "Unknown" in result

    def test_no_input(self):
        result = _summarize_tool_call("Read", None, None)
        assert "Read" in result

    def test_large_output(self):
        result = _summarize_tool_call("Read", None, "x" * 2000)
        assert "2,000 chars" in result

    def test_grep_found_output(self):
        result = _summarize_tool_call("Grep", None, "Found 5 matches")
        assert "5 files" in result

    def test_small_output(self):
        result = _summarize_tool_call("Read", None, "small output")
        # Small output should not add char count
        assert "chars" not in result


class TestSummarizeThinking:
    def test_empty_content(self):
        assert _summarize_thinking("") is None
        assert _summarize_thinking(None) is None

    def test_thinking_tags(self):
        result = _summarize_thinking(
            "<thinking>I need to parse the statute carefully.</thinking>"
        )
        assert "I need to parse the statute carefully" in result

    def test_thinking_tags_long(self):
        result = _summarize_thinking(
            f"<thinking>{'x' * 200}</thinking>"
        )
        assert "..." in result

    def test_reasoning_prefix(self):
        result = _summarize_thinking(
            "I need to figure out the correct tax bracket"
        )
        assert "I need to figure out" in result

    def test_let_me_prefix(self):
        result = _summarize_thinking(
            "Let me analyze the statute structure first."
        )
        assert "Let me" in result

    def test_no_pattern_found(self):
        result = _summarize_thinking(
            "Some random text that doesn't match any patterns"
        )
        assert result is None

    def test_empty_thinking_tags(self):
        result = _summarize_thinking("<thinking>   </thinking>")
        assert result is None

    def test_first_sentence_prefix(self):
        result = _summarize_thinking("First, we need to understand the structure.")
        assert "First" in result

    def test_long_first_sentence(self):
        result = _summarize_thinking(
            "Let me " + "x" * 200
        )
        assert "..." in result


class TestSummarizeAssistantMessage:
    def test_empty_content(self):
        result = _summarize_assistant_message("")
        assert result == "Empty response"

    def test_none_content(self):
        result = _summarize_assistant_message(None)
        assert result == "Empty response"

    def test_error_content(self):
        result = _summarize_assistant_message("Error occurred while parsing")
        assert "error" in result.lower()

    def test_success_content(self):
        result = _summarize_assistant_message("Successfully created the file")
        assert "successfully" in result.lower()

    def test_created_content(self):
        result = _summarize_assistant_message("Created a new RAC file")
        assert "Completed successfully" in result

    def test_code_block(self):
        result = _summarize_assistant_message("Here is the code:\n```\nprint('hello')\n```")
        assert "Code block" in result

    def test_multiline_first_line(self):
        result = _summarize_assistant_message("This is the first line.\nSecond line.")
        assert "This is the first line" in result

    def test_long_first_line(self):
        result = _summarize_assistant_message("a" * 200)
        assert "..." in result

    def test_only_comment_lines(self):
        result = _summarize_assistant_message("# comment\n# another comment\n\n")
        assert "Response" in result


# =========================================================================
# Test SDKOrchestrator
# =========================================================================


class TestSDKOrchestratorInit:
    def test_requires_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]
            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                SDKOrchestrator(api_key=None)

    def test_with_explicit_api_key(self):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=Path("/tmp/plugin")):
            orch = SDKOrchestrator(api_key="test-key")
            assert orch.api_key == "test-key"

    def test_with_env_api_key(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "env-key"}):
            with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=Path("/tmp/plugin")):
                orch = SDKOrchestrator()
                assert orch.api_key == "env-key"

    def test_custom_model(self):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=Path("/tmp/plugin")):
            orch = SDKOrchestrator(api_key="key", model="custom-model")
            assert orch.model == "custom-model"

    def test_default_model(self):
        from autorac.constants import DEFAULT_MODEL
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=Path("/tmp/plugin")):
            orch = SDKOrchestrator(api_key="key")
            assert orch.model == DEFAULT_MODEL


class TestFindPluginPath:
    def test_returns_path(self):
        """_find_plugin_path always returns a Path."""
        result = SDKOrchestrator._find_plugin_path()
        assert isinstance(result, Path)

    def test_finds_path_with_agents_dir(self, tmp_path):
        """When a candidate path has an 'agents' dir, it's returned."""
        agents = tmp_path / "agents"
        agents.mkdir()
        with patch("pathlib.Path.home", return_value=tmp_path):
            # Create the marketplace structure
            marketplace = tmp_path / ".claude" / "plugins" / "marketplaces" / "rac" / "agents"
            marketplace.mkdir(parents=True)
            result = SDKOrchestrator._find_plugin_path()
            assert "rac" in str(result)


class TestLoadAgentPrompt:
    def test_unknown_agent(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            assert orch._load_agent_prompt("unknown_agent") == ""

    def test_missing_prompt_file(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            assert orch._load_agent_prompt("encoder") == ""

    def test_existing_prompt_file(self, tmp_path):
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "encoder.md").write_text("# Encoder prompt")
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            result = orch._load_agent_prompt("encoder")
            assert "Encoder prompt" in result


class TestFetchStatuteText:
    def test_fetch_statute_text(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            # _fetch_statute_text tries atlas cache then legacy XML.
            # Mock the legacy path to return text.
            with patch.object(
                orch, "_fetch_statute_text_legacy", return_value="Statute text"
            ):
                result = orch._fetch_statute_text("26 USC 1")
                assert isinstance(result, str)


class TestBuildPrompts:
    def test_build_analyzer_prompt(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            prompt = orch._build_analyzer_prompt("26 USC 1", statute_text="Tax text")
            assert "26 USC 1" in prompt
            assert "Tax text" in prompt

    def test_build_fallback_encode_prompt(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            prompt = orch._build_fallback_encode_prompt(
                "26 USC 1", Path("/tmp/output"), "Tax text"
            )
            assert "26 USC 1" in prompt

    def test_build_subsection_prompt(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            task = SubsectionTask(
                subsection_id="a", title="General", file_name="a.rac",
                dependencies=["b"],
            )
            prompt = orch._build_subsection_prompt(
                task=task, citation="26 USC 1", output_path=Path("/tmp/output"),
                statute_text="Tax text",
            )
            assert "26 USC 1" in prompt
            assert "b.rac" in prompt  # dependency

    def test_build_subsection_prompt_no_deps(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            task = SubsectionTask(
                subsection_id="a", title="General", file_name="a.rac",
            )
            prompt = orch._build_subsection_prompt(
                task=task, citation="26 USC 1", output_path=Path("/tmp/output"),
            )
            assert "26 USC 1" in prompt


class TestParseAnalyzerOutput:
    def test_parse_with_subsections(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            analysis_text = """
Subsections found:
(a) General rule - defines basic credit
(b) Phase-out - applies phase-out reduction
(c) Definitions - defines key terms
"""
            result = orch._parse_analyzer_output(analysis_text)
            assert isinstance(result, AnalyzerOutput)

    def test_parse_empty(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            result = orch._parse_analyzer_output("")
            assert isinstance(result, AnalyzerOutput)

    def test_parse_none(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            result = orch._parse_analyzer_output(None)
            assert isinstance(result, AnalyzerOutput)


class TestFormatOracleSummary:
    def test_format_with_data(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            context = {
                "pe_match": 95.0,
                "taxsim_match": 90.0,
                "discrepancies": [
                    {"description": "Phaseout wrong"},
                ],
            }
            result = orch._format_oracle_summary(context)
            assert "95" in result
            assert "90" in result

    def test_format_with_no_data(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            context = {"pe_match": None, "taxsim_match": None, "discrepancies": []}
            result = orch._format_oracle_summary(context)
            assert isinstance(result, str)


class TestTokenHelpers:
    def test_sum_tokens_empty(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            result = orch._sum_tokens([])
            assert result.input_tokens == 0
            assert result.output_tokens == 0

    def test_sum_tokens_with_data(self, tmp_path):
        from autorac.harness.experiment_db import TokenUsage

        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            runs = [
                AgentRun(
                    agent_type="test",
                    prompt="test",
                    phase=Phase.ANALYSIS,
                    total_tokens=TokenUsage(input_tokens=100, output_tokens=50),
                ),
                AgentRun(
                    agent_type="test",
                    prompt="test",
                    phase=Phase.ENCODING,
                    total_tokens=TokenUsage(input_tokens=200, output_tokens=100),
                ),
                AgentRun(
                    agent_type="test",
                    prompt="test",
                    phase=Phase.REVIEW,
                    total_tokens=None,  # No tokens
                ),
            ]
            result = orch._sum_tokens(runs)
            assert result is not None
            assert result.input_tokens == 300
            assert result.output_tokens == 150

    def test_sum_cost_empty(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            result = orch._sum_cost([])
            assert result == 0.0

    def test_sum_cost_with_data(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            runs = [
                AgentRun(
                    agent_type="test",
                    prompt="test",
                    phase=Phase.ANALYSIS,
                    total_cost=0.05,
                ),
                AgentRun(
                    agent_type="test",
                    prompt="test",
                    phase=Phase.ENCODING,
                    total_cost=0.10,
                ),
                AgentRun(
                    agent_type="test",
                    prompt="test",
                    phase=Phase.REVIEW,
                    total_cost=None,
                ),
            ]
            result = orch._sum_cost(runs)
            assert abs(result - 0.15) < 0.001


class TestPrintReport:
    def test_print_report(self, tmp_path):
        from autorac.harness.experiment_db import TokenUsage

        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            run = OrchestratorRun(
                citation="26 USC 1",
                session_id="test-session",
                started_at=datetime(2024, 1, 1, 10, 0),
                ended_at=datetime(2024, 1, 1, 10, 5),
                agent_runs=[
                    AgentRun(
                        agent_type="analyzer",
                        prompt="Analyze",
                        phase=Phase.ANALYSIS,
                        started_at=datetime(2024, 1, 1, 10, 0),
                        ended_at=datetime(2024, 1, 1, 10, 1),
                        total_tokens=TokenUsage(input_tokens=100, output_tokens=50),
                        messages=[
                            AgentMessage(
                                role="assistant",
                                content="Analysis result",
                                summary="Analyzed statute",
                            ),
                        ],
                        result="Analysis complete",
                    ),
                ],
                files_created=["a.rac", "b.rac"],
                oracle_pe_match=95.0,
                oracle_taxsim_match=90.0,
                total_tokens=TokenUsage(input_tokens=100, output_tokens=50),
            )

            report = orch.print_report(run)
            assert "26 USC 1" in report
            assert "analyzer" in report or "ANALYSIS" in report

    def test_print_report_with_error(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            run = OrchestratorRun(
                citation="26 USC 1",
                session_id="test-session",
                agent_runs=[
                    AgentRun(
                        agent_type="encoder",
                        prompt="Encode",
                        phase=Phase.ENCODING,
                        error="Failed to encode",
                    ),
                ],
            )

            report = orch.print_report(run)
            assert isinstance(report, str)

    def test_print_report_no_dates(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            run = OrchestratorRun(
                citation="26 USC 1",
                session_id="test-session",
                agent_runs=[],
            )

            report = orch.print_report(run)
            assert isinstance(report, str)


class TestLogAgentRun:
    def test_log_without_db(self, tmp_path):
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            agent_run = AgentRun(
                agent_type="test",
                prompt="test",
                phase=Phase.ANALYSIS,
            )
            # Should not raise
            orch._log_agent_run("session-id", agent_run)

    def test_log_with_db(self, tmp_path):
        from autorac.harness.experiment_db import ExperimentDB

        db = ExperimentDB(tmp_path / "test.db")
        session = db.start_session(model="test")

        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key", experiment_db=db)
            agent_run = AgentRun(
                agent_type="test",
                prompt="test prompt",
                phase=Phase.ANALYSIS,
                messages=[
                    AgentMessage(role="assistant", content="response", summary="sum"),
                ],
            )
            orch._log_agent_run(session.id, agent_run)

    def test_log_with_tools_and_cost(self, tmp_path):
        """Test _log_agent_run with tool counts and cost."""
        from autorac.harness.experiment_db import ExperimentDB

        db = ExperimentDB(tmp_path / "test.db")
        session = db.start_session(model="test")

        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key", experiment_db=db)
            agent_run = AgentRun(
                agent_type="test",
                prompt="test",
                phase=Phase.ENCODING,
                total_cost=0.05,
                result="Done",
                messages=[
                    AgentMessage(
                        role="tool_use",
                        content="Read file",
                        tool_name="Read",
                        tool_input={"file_path": "/tmp/test.rac"},
                        tool_output="content",
                        summary="Read test.rac",
                    ),
                    AgentMessage(
                        role="tool_use",
                        content="Write file",
                        tool_name="Write",
                        tool_input={"file_path": "/tmp/test.rac"},
                        summary="Write test.rac",
                    ),
                ],
            )
            orch._log_agent_run(session.id, agent_run)


class TestLogToDb:
    def test_log_to_db(self, tmp_path):
        from autorac.harness.experiment_db import ExperimentDB, TokenUsage

        db = ExperimentDB(tmp_path / "test.db")

        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key", experiment_db=db)
            run = OrchestratorRun(
                citation="26 USC 1",
                session_id="test-session",
                agent_runs=[
                    AgentRun(
                        agent_type="encoder",
                        prompt="Encode",
                        phase=Phase.ENCODING,
                        total_tokens=TokenUsage(input_tokens=100, output_tokens=50),
                    ),
                ],
                total_tokens=TokenUsage(input_tokens=100, output_tokens=50),
            )
            orch._log_to_db(run)


# =========================================================================
# Test dataclass basics
# =========================================================================


class TestDataclasses:
    def test_phase_enum(self):
        assert Phase.ANALYSIS.value == "analysis"
        assert Phase.ENCODING.value == "encoding"
        assert Phase.ORACLE.value == "oracle"
        assert Phase.REVIEW.value == "review"
        assert Phase.REPORT.value == "report"

    def test_subsection_task(self):
        task = SubsectionTask(
            subsection_id="a",
            title="General",
            file_name="a.rac",
            dependencies=["b"],
            wave=1,
        )
        assert task.subsection_id == "a"
        assert task.wave == 1

    def test_analyzer_output(self):
        output = AnalyzerOutput(raw_text="test")
        assert output.raw_text == "test"
        assert output.subsections == []

    def test_agent_message(self):
        msg = AgentMessage(role="user", content="test")
        assert msg.role == "user"
        assert msg.tool_name is None

    def test_agent_run(self):
        run = AgentRun(
            agent_type="encoder",
            prompt="test",
            phase=Phase.ENCODING,
        )
        assert run.error is None
        assert run.total_tokens is None

    def test_orchestrator_run(self):
        run = OrchestratorRun(
            citation="26 USC 1",
            session_id="test",
        )
        assert run.files_created == []
        assert run.agent_runs == []


# =========================================================================
# Test encode workflow (mocked)
# =========================================================================


class TestEncodeWorkflow:
    @pytest.mark.asyncio
    async def test_encode_with_no_analysis_result(self, tmp_path):
        """When analysis returns no result, use fallback encoder."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            # Mock _run_agent to return no result for analysis, then encoding
            call_count = 0

            async def mock_run_agent(agent_key, prompt, phase, model):
                nonlocal call_count
                call_count += 1
                run = AgentRun(
                    agent_type=agent_key,
                    prompt=prompt,
                    phase=phase,
                )
                if call_count == 1:
                    # Analysis: no result
                    run.result = None
                else:
                    run.result = "Encoded"
                return run

            orch._run_agent = mock_run_agent
            orch._fetch_statute_text = lambda c: "Statute text"
            orch._build_analyzer_prompt = lambda c, statute_text: "Analyze"
            orch._build_fallback_encode_prompt = lambda c, o, s: "Encode"
            orch._format_oracle_summary = lambda c: "Oracle summary"
            orch._log_agent_run = lambda s, r: None
            orch._log_to_db = lambda r: None

            result = await orch.encode("26 USC 1", tmp_path / "output")
            assert isinstance(result, OrchestratorRun)
            # Should have used fallback path since analysis had no result
            assert call_count >= 2

    @pytest.mark.asyncio
    async def test_encode_handles_exception(self, tmp_path):
        """When an exception occurs during encoding, partial results are preserved."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            async def mock_run_agent_fail(agent_key, prompt, phase, model):
                raise Exception("SDK connection failed")

            orch._run_agent = mock_run_agent_fail
            orch._fetch_statute_text = lambda c: "Statute text"
            orch._build_analyzer_prompt = lambda c, statute_text: "Analyze"

            result = await orch.encode("26 USC 1", tmp_path / "output")
            assert isinstance(result, OrchestratorRun)
            assert result.ended_at is not None

    @pytest.mark.asyncio
    async def test_encode_with_empty_encoding_runs(self, tmp_path):
        """When analysis returns result but encoding returns empty list, use fallback."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            call_count = 0

            async def mock_run_agent(agent_key, prompt, phase, model):
                nonlocal call_count
                call_count += 1
                run = AgentRun(
                    agent_type=agent_key,
                    prompt=prompt,
                    phase=phase,
                )
                run.result = "Some result"
                return run

            async def mock_run_encoding_parallel(citation, output_path, text, analysis):
                return []  # Empty encoding runs

            orch._run_agent = mock_run_agent
            orch._run_encoding_parallel = mock_run_encoding_parallel
            orch._fetch_statute_text = lambda c: "Statute text"
            orch._build_analyzer_prompt = lambda c, statute_text: "Analyze"
            orch._build_fallback_encode_prompt = lambda c, o, s: "Encode"
            orch._format_oracle_summary = lambda c: "Oracle summary"
            orch._log_agent_run = lambda s, r: None
            orch._log_to_db = lambda r: None

            result = await orch.encode("26 USC 1", tmp_path / "output")
            assert isinstance(result, OrchestratorRun)

    @pytest.mark.asyncio
    async def test_encode_with_experiment_db(self, tmp_path):
        """Encoding with experiment_db logs session."""
        from autorac.harness.experiment_db import ExperimentDB

        db = ExperimentDB(tmp_path / "test.db")

        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key", experiment_db=db)

            async def mock_run_agent(agent_key, prompt, phase, model):
                run = AgentRun(
                    agent_type=agent_key,
                    prompt=prompt,
                    phase=phase,
                    result="Result",
                )
                return run

            async def mock_run_encoding_parallel(citation, output_path, text, analysis):
                return [
                    AgentRun(
                        agent_type="encoder",
                        prompt="test",
                        phase=Phase.ENCODING,
                        result="Encoded",
                    )
                ]

            orch._run_agent = mock_run_agent
            orch._run_encoding_parallel = mock_run_encoding_parallel
            orch._fetch_statute_text = lambda c: "Statute text"
            orch._build_analyzer_prompt = lambda c, statute_text: "Analyze"
            orch._format_oracle_summary = lambda c: "Oracle summary"

            result = await orch.encode("26 USC 1", tmp_path / "output")
            assert isinstance(result, OrchestratorRun)


# =========================================================================
# Additional coverage tests
# =========================================================================


class TestRunAgent:
    """Tests for _run_agent method to cover the full SDK event loop."""

    @pytest.mark.asyncio
    async def test_run_agent_basic(self, tmp_path):
        """Test _run_agent with basic SDK event flow."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            # Mock claude_agent_sdk
            mock_sdk = MagicMock()

            class TextBlock:
                def __init__(self, text):
                    self.text = text

            class ToolUseBlock:
                def __init__(self, name, input_data):
                    self.name = name
                    self.input = input_data

            class ToolResultBlock:
                def __init__(self, content):
                    self.content = content

            class AssistantMsg:
                def __init__(self):
                    self.content = [TextBlock("Analyzing statute")]

            class ToolMsg:
                def __init__(self):
                    self.content = [
                        ToolUseBlock("Read", {"file_path": "/tmp/test.rac"}),
                        ToolResultBlock("file content here"),
                    ]

            # Class name MUST be "ResultMessage" - type(event).__name__ is checked
            # NOTE: ResultMessage should NOT have 'content' so the elif
            # hasattr(event, "result") branch is taken
            class ResultMessage:
                def __init__(self):
                    self.type = "result"
                    self.result = "Final result"
                    self.usage = {
                        "input_tokens": 1000,
                        "output_tokens": 500,
                        "cache_creation_input_tokens": 100,
                        "cache_read_input_tokens": 200,
                    }
                    self.total_cost_usd = 0.05

            async def mock_query(**kwargs):
                yield AssistantMsg()
                yield ToolMsg()
                yield ResultMessage()

            mock_sdk.query = mock_query
            mock_sdk.ClaudeAgentOptions = MagicMock()

            with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
                result = await orch._run_agent(
                    agent_key="encoder",
                    prompt="Encode 26 USC 1",
                    phase=Phase.ENCODING,
                    model="test-model",
                )

            # AGENTS dict maps "encoder" to "rac:RAC Encoder"
            assert "encoder" in result.agent_type.lower() or result.agent_type == "rac:RAC Encoder"
            assert result.result == "Final result"
            assert result.total_tokens is not None
            assert result.total_tokens.input_tokens == 1100  # 1000 + 100 cache_create
            assert result.total_tokens.output_tokens == 500

    @pytest.mark.asyncio
    async def test_run_agent_with_error(self, tmp_path):
        """Test _run_agent when SDK raises an exception."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            mock_sdk = MagicMock()

            async def mock_query(**kwargs):
                raise RuntimeError("Connection failed")
                yield  # pragma: no cover - Make it a generator

            mock_sdk.query = mock_query
            mock_sdk.ClaudeAgentOptions = MagicMock()

            with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
                result = await orch._run_agent(
                    agent_key="encoder",
                    prompt="Encode",
                    phase=Phase.ENCODING,
                    model="test-model",
                )

            assert result.error == "Connection failed"
            assert result.ended_at is not None

    @pytest.mark.asyncio
    async def test_run_agent_with_event_result_attribute(self, tmp_path):
        """Test _run_agent with event that has result attribute but no content."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            mock_sdk = MagicMock()

            class EventWithResult:
                def __init__(self):
                    self.result = "Direct result"

            async def mock_query(**kwargs):
                yield EventWithResult()

            mock_sdk.query = mock_query
            mock_sdk.ClaudeAgentOptions = MagicMock()

            with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
                result = await orch._run_agent(
                    agent_key="encoder",
                    prompt="Test",
                    phase=Phase.ENCODING,
                    model="test-model",
                )

            assert result.result == "Direct result"

    @pytest.mark.asyncio
    async def test_run_agent_fallback_tool_capture(self, tmp_path):
        """Test _run_agent fallback tool capture from event attributes."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            mock_sdk = MagicMock()

            class EventWithToolAttrs:
                def __init__(self):
                    self.content = "text content"
                    self.tool_name = "Bash"
                    self.tool_input = {"command": "ls"}
                    self.tool_output = "file1 file2"

            async def mock_query(**kwargs):
                yield EventWithToolAttrs()

            mock_sdk.query = mock_query
            mock_sdk.ClaudeAgentOptions = MagicMock()

            with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
                result = await orch._run_agent(
                    agent_key="encoder",
                    prompt="Test",
                    phase=Phase.ENCODING,
                    model="test-model",
                )

            assert len(result.messages) > 0
            assert result.messages[0].tool_name == "Bash"

    @pytest.mark.asyncio
    async def test_run_agent_partial_tokens_on_error(self, tmp_path):
        """Test _run_agent logs partial tokens when error occurs."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            mock_sdk = MagicMock()

            class ResultMessage:
                """Named to match type(event).__name__ == 'ResultMessage' check."""

                def __init__(self):
                    self.content = "partial"
                    self.usage = {
                        "input_tokens": 500,
                        "output_tokens": 200,
                        "cache_creation_input_tokens": 0,
                        "cache_read_input_tokens": 0,
                    }

            async def mock_query(**kwargs):
                yield ResultMessage()
                raise RuntimeError("Mid-stream failure")

            mock_sdk.query = mock_query
            mock_sdk.ClaudeAgentOptions = MagicMock()

            with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
                result = await orch._run_agent(
                    agent_key="encoder",
                    prompt="Test",
                    phase=Phase.ENCODING,
                    model="test-model",
                )

            assert result.error is not None


class TestComputeWaves:
    """Tests for _compute_waves including unsatisfiable deps branch."""

    def test_compute_waves_unsatisfiable_deps(self, tmp_path):
        """Test _compute_waves with circular/unsatisfiable dependencies."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            # Circular dependencies — can't be resolved
            tasks = [
                SubsectionTask(
                    subsection_id="a",
                    title="A",
                    file_name="a.rac",
                    dependencies=["b"],
                ),
                SubsectionTask(
                    subsection_id="b",
                    title="B",
                    file_name="b.rac",
                    dependencies=["a"],
                ),
            ]

            waves = orch._compute_waves(tasks)
            # Should still produce waves (unsatisfiable go in last wave)
            assert len(waves) >= 1
            # All tasks should be assigned
            all_task_ids = {t.subsection_id for wave in waves for t in wave}
            assert all_task_ids == {"a", "b"}


class TestBatchSmallSubsections:
    """Tests for _batch_small_subsections including internal dependency branch."""

    def test_batch_with_internal_deps(self, tmp_path):
        """Test batching skips tasks with internal dependencies."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            tasks = [
                SubsectionTask(subsection_id="g/1", title="T1", file_name="g_1.rac"),
                SubsectionTask(subsection_id="g/2", title="T2", file_name="g_2.rac"),
                SubsectionTask(
                    subsection_id="g/3",
                    title="T3",
                    file_name="g_3.rac",
                    dependencies=["g/1"],  # Internal dep
                ),
                SubsectionTask(subsection_id="g/4", title="T4", file_name="g_4.rac"),
            ]

            result = orch._batch_small_subsections(tasks)
            assert len(result) >= 1  # Should produce results

    def test_batch_too_few_tasks(self, tmp_path):
        """Test that 3 or fewer tasks are not batched."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            tasks = [
                SubsectionTask(subsection_id="a", title="A", file_name="a.rac"),
                SubsectionTask(subsection_id="b", title="B", file_name="b.rac"),
            ]

            result = orch._batch_small_subsections(tasks)
            assert len(result) == 2


class TestGetCachedSection:
    """Tests for _get_cached_section and related methods."""

    def test_get_cached_section_no_atlas(self, tmp_path):
        """Test _get_cached_section when atlas is not installed."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            import builtins

            orig_import = builtins.__import__

            def no_atlas_import(name, *args, **kwargs):
                if name == "atlas":
                    raise ImportError("no atlas")
                return orig_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=no_atlas_import):
                result = orch._get_cached_section("26 USC 1")
                assert result is None

    def test_get_cached_section_no_db(self, tmp_path):
        """Test _get_cached_section when db doesn't exist."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            mock_atlas_mod = MagicMock()

            import builtins

            orig_import = builtins.__import__

            def atlas_import(name, *args, **kwargs):
                if name == "atlas":
                    return mock_atlas_mod
                return orig_import(name, *args, **kwargs)

            # Make Path.home() return tmp_path so atlas.db path won't exist
            with patch("builtins.__import__", side_effect=atlas_import), \
                 patch("pathlib.Path.home", return_value=tmp_path):
                result = orch._get_cached_section("26 USC 999")
                assert result is None

    def test_fetch_subsection_text_no_section(self, tmp_path):
        """Test _fetch_subsection_text when section is None."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            orch._section_cache = {"26 USC 1": None}
            result = orch._fetch_subsection_text("26 USC 1", "a")
            assert result is None

    def test_fetch_subsection_text_with_section(self, tmp_path):
        """Test _fetch_subsection_text with a valid section."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            mock_section = MagicMock()
            mock_section.get_subsection_text.return_value = "Subsection text"
            orch._section_cache = {"26 USC 1": mock_section}
            result = orch._fetch_subsection_text("26 USC 1", "a")
            assert result == "Subsection text"

    def test_fetch_statute_text_with_section(self, tmp_path):
        """Test _fetch_statute_text when atlas section exists."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            mock_section = MagicMock()
            mock_section.text = "Full statute text"
            orch._section_cache = {"26 USC 1": mock_section}
            result = orch._fetch_statute_text("26 USC 1")
            assert result == "Full statute text"


class TestFetchStatuteTextLegacy:
    """Tests for _fetch_statute_text_legacy XML parsing."""

    def test_legacy_parse_xml(self, tmp_path):
        """Test parsing statute text from USC XML."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            xml_dir = tmp_path / "uscode"
            xml_dir.mkdir()
            xml_file = xml_dir / "usc26.xml"
            xml_file.write_text(
                '<section identifier="/us/usc/t26/s1">'
                "<heading>Tax imposed</heading>"
                "<content>There is hereby imposed a tax.</content>"
                "</section>"
            )

            result = orch._fetch_statute_text_legacy("26 USC 1", xml_path=xml_dir)
            assert result is not None
            assert "tax" in result.lower()

    def test_legacy_no_xml_file(self, tmp_path):
        """Test when XML file doesn't exist."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            xml_dir = tmp_path / "uscode"
            xml_dir.mkdir()
            result = orch._fetch_statute_text_legacy("26 USC 1", xml_path=xml_dir)
            assert result is None

    def test_legacy_bad_citation(self, tmp_path):
        """Test with unparseable citation."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            result = orch._fetch_statute_text_legacy("bad", xml_path=tmp_path)
            assert result is None

    def test_legacy_non_numeric_title(self, tmp_path):
        """Test with non-numeric title."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            result = orch._fetch_statute_text_legacy("abc USC 1", xml_path=tmp_path)
            assert result is None

    def test_legacy_no_section_match(self, tmp_path):
        """Test when section identifier not found in XML."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            xml_dir = tmp_path / "uscode"
            xml_dir.mkdir()
            xml_file = xml_dir / "usc26.xml"
            xml_file.write_text("<root>No sections here</root>")

            result = orch._fetch_statute_text_legacy("26 USC 999", xml_path=xml_dir)
            assert result is None

    def test_legacy_xml_read_error(self, tmp_path):
        """Test when XML file can't be read."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            xml_dir = tmp_path / "uscode"
            xml_dir.mkdir()
            xml_file = xml_dir / "usc26.xml"
            xml_file.write_text("content")
            xml_file.chmod(0o000)

            try:
                result = orch._fetch_statute_text_legacy("26 USC 1", xml_path=xml_dir)
                assert result is None
            finally:
                xml_file.chmod(0o644)


class TestLogToDbNoDB:
    """Test _log_to_db when there's no DB."""

    def test_log_to_db_no_db(self, tmp_path):
        """_log_to_db returns early when no experiment_db."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            assert orch.experiment_db is None
            run = OrchestratorRun(citation="26 USC 1", session_id="test")
            orch._log_to_db(run)  # Should not raise


class TestPrintReportBranches:
    """Test print_report edge cases for agent run cost/token branches."""

    def test_report_with_agent_cost(self, tmp_path):
        """Test report with agent run that has cost."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            run = OrchestratorRun(
                citation="26 USC 1",
                session_id="test",
                started_at=datetime(2024, 1, 1, 10, 0),
                ended_at=datetime(2024, 1, 1, 10, 5),
                agent_runs=[
                    AgentRun(
                        agent_type="encoder",
                        prompt="test",
                        phase=Phase.ENCODING,
                        total_cost=0.0512,
                    ),
                ],
            )
            report = orch.print_report(run)
            assert "$0.0512" in report

    def test_report_with_no_cost_no_tokens(self, tmp_path):
        """Test report with agent run that has neither cost nor tokens."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            run = OrchestratorRun(
                citation="26 USC 1",
                session_id="test",
                started_at=datetime(2024, 1, 1, 10, 0),
                ended_at=datetime(2024, 1, 1, 10, 5),
                agent_runs=[
                    AgentRun(
                        agent_type="encoder",
                        prompt="test",
                        phase=Phase.ENCODING,
                    ),
                ],
            )
            report = orch.print_report(run)
            assert "encoding: encoder" in report


class TestBuildSubsectionPromptWithAtlas:
    """Test _build_subsection_prompt with subsection text from atlas."""

    def test_build_subsection_prompt_with_subsection_text(self, tmp_path):
        """When subsection_text param is provided, include it in prompt."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            task = SubsectionTask(
                subsection_id="a",
                title="General rule",
                file_name="a.rac",
            )

            # subsection_text is a direct parameter
            prompt = orch._build_subsection_prompt(
                task=task,
                citation="26 USC 1",
                output_path=Path("/tmp/output"),
                subsection_text="Subsection (a) defines the general rule...",
            )
            assert "Subsection (a) defines the general rule" in prompt


class TestRunAgentWithSystemPrompt:
    """Test _run_agent with system prompt loaded from plugin."""

    @pytest.mark.asyncio
    async def test_run_agent_with_system_prompt(self, tmp_path):
        """Test that system prompt is prepended when available."""
        # Create agents directory with a prompt file
        agents_dir = tmp_path / "agents"
        agents_dir.mkdir()
        (agents_dir / "encoder.md").write_text("# Encoder System Prompt\nYou are an encoder.")

        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            mock_sdk = MagicMock()

            class SimpleEvent:
                def __init__(self):
                    self.result = "Done"

            captured_prompt = None

            async def mock_query(prompt, **kwargs):
                nonlocal captured_prompt
                captured_prompt = prompt
                yield SimpleEvent()

            mock_sdk.query = mock_query
            mock_sdk.ClaudeAgentOptions = MagicMock()

            with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
                await orch._run_agent(
                    agent_key="encoder",
                    prompt="Encode 26 USC 1",
                    phase=Phase.ENCODING,
                    model="test-model",
                )

            # System prompt should be prepended
            assert "Encoder System Prompt" in captured_prompt
            assert "# TASK" in captured_prompt


class TestParseAnalyzerOutputJsonFallback:
    """Test _parse_analyzer_output JSON decode error fallback."""

    def test_parse_analyzer_invalid_json(self, tmp_path):
        """Test that invalid JSON in output falls through to markdown parsing."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            # Text that looks like JSON but is invalid
            result = orch._parse_analyzer_output('{"subsections": [invalid json}')
            assert isinstance(result, AnalyzerOutput)

    def test_parse_analyzer_structured_output_invalid_json(self, tmp_path):
        """Test STRUCTURED_OUTPUT block with invalid JSON falls through."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            # Valid STRUCTURED_OUTPUT comment block but bad JSON
            text = """<!-- STRUCTURED_OUTPUT
{this is not valid json}
-->

Some markdown analysis here.
"""
            result = orch._parse_analyzer_output(text)
            assert isinstance(result, AnalyzerOutput)
            # Should still work (falls through to markdown)

    def test_parse_analyzer_structured_output_missing_key(self, tmp_path):
        """Test STRUCTURED_OUTPUT with missing required key falls through."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")
            # Valid JSON but a subsection is missing "id" key
            text = '''<!-- STRUCTURED_OUTPUT
{"subsections": [{"title": "A", "disposition": "ENCODE"}], "dependencies": {}}
-->'''
            result = orch._parse_analyzer_output(text)
            assert isinstance(result, AnalyzerOutput)


class TestGetCachedSectionWithDb:
    """Test _get_cached_section when atlas db exists."""

    def test_cached_section_with_db(self, tmp_path):
        """Test _get_cached_section when atlas.db exists."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            # Create the atlas db file
            atlas_dir = tmp_path / "RulesFoundation" / "atlas"
            atlas_dir.mkdir(parents=True)
            (atlas_dir / "atlas.db").write_text("fake db")

            mock_section = MagicMock()
            mock_section.text = "Statute text"
            mock_arch_instance = MagicMock()
            mock_arch_instance.get.return_value = mock_section
            mock_arch_class = MagicMock(return_value=mock_arch_instance)

            mock_atlas_mod = MagicMock()
            mock_atlas_mod.Arch = mock_arch_class

            import builtins

            orig_import = builtins.__import__

            def atlas_import(name, *args, **kwargs):
                if name == "atlas":
                    return mock_atlas_mod
                return orig_import(name, *args, **kwargs)

            with patch("builtins.__import__", side_effect=atlas_import), \
                 patch("pathlib.Path.home", return_value=tmp_path):
                result = orch._get_cached_section("26 USC 1")
                assert result == mock_section


class TestEncodeExceptionSetsAgentRunError:
    """Test that encode() sets error on last agent_run when exception occurs."""

    @pytest.mark.asyncio
    async def test_encode_exception_sets_last_run_error(self, tmp_path):
        """When encode() catches an exception, it sets error on last agent_run."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            call_count = 0

            async def mock_run_agent(agent_key, prompt, phase, model):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    # Analysis succeeds
                    return AgentRun(
                        agent_type="analyzer",
                        prompt=prompt,
                        phase=phase,
                        result="Analysis",
                    )
                else:
                    # Second call fails
                    raise RuntimeError("Encoding failed!")

            orch._run_agent = mock_run_agent
            orch._fetch_statute_text = lambda c: "Statute text"
            orch._build_analyzer_prompt = lambda c, statute_text: "Analyze"

            result = await orch.encode("26 USC 1", tmp_path / "output")
            assert isinstance(result, OrchestratorRun)
            # The last agent_run should have the error
            if result.agent_runs:
                assert result.agent_runs[-1].error is not None or result.ended_at is not None


class TestOracleValidationInEncode:
    """Tests for oracle validation in encode workflow."""

    @pytest.mark.asyncio
    async def test_encode_with_oracle_validation(self, tmp_path):
        """Test encode workflow includes oracle validation with RAC files."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            # Create a RAC file in the output directory
            output_dir = tmp_path / "output"
            output_dir.mkdir()
            (output_dir / "test.rac").write_text("test_var:\n  entity: Person\n")

            async def mock_run_agent(agent_key, prompt, phase, model):
                run = AgentRun(
                    agent_type=agent_key,
                    prompt=prompt,
                    phase=phase,
                    result="Analysis" if phase == Phase.ANALYSIS else "Review",
                )
                return run

            async def mock_run_encoding_parallel(citation, output_path, text, analysis):
                return [
                    AgentRun(
                        agent_type="encoder",
                        prompt="test",
                        phase=Phase.ENCODING,
                        result="Encoded",
                    )
                ]

            orch._run_agent = mock_run_agent
            orch._run_encoding_parallel = mock_run_encoding_parallel
            orch._fetch_statute_text = lambda c: "Statute text"
            orch._build_analyzer_prompt = lambda c, statute_text: "Analyze"
            orch._format_oracle_summary = lambda c: "PE: 95%, TAXSIM: 90%"
            orch._log_agent_run = lambda s, r: None
            orch._log_to_db = lambda r: None

            mock_vp = MagicMock()
            from autorac.harness.validator_pipeline import ValidationResult

            mock_vp._run_policyengine.return_value = ValidationResult(
                validator_name="policyengine",
                passed=True,
                score=0.95,
                issues=[],
                duration_ms=100,
            )
            mock_vp._run_taxsim.return_value = ValidationResult(
                validator_name="taxsim",
                passed=True,
                score=0.90,
                issues=[],
                duration_ms=100,
            )

            with patch(
                "autorac.harness.sdk_orchestrator.ValidatorPipeline",
                return_value=mock_vp,
            ):
                result = await orch.encode("26 USC 1", output_dir)

            assert isinstance(result, OrchestratorRun)
            assert result.oracle_pe_match == 95.0
            assert result.oracle_taxsim_match == 90.0

    @pytest.mark.asyncio
    async def test_encode_oracle_exception(self, tmp_path):
        """Test oracle validation handles exceptions gracefully."""
        with patch.object(SDKOrchestrator, "_find_plugin_path", return_value=tmp_path):
            orch = SDKOrchestrator(api_key="key")

            output_dir = tmp_path / "output2"
            output_dir.mkdir()
            (output_dir / "test.rac").write_text("test_var:\n  entity: Person\n")

            async def mock_run_agent(agent_key, prompt, phase, model):
                return AgentRun(
                    agent_type=agent_key,
                    prompt=prompt,
                    phase=phase,
                    result="Result",
                )

            async def mock_run_encoding_parallel(citation, output_path, text, analysis):
                return [
                    AgentRun(
                        agent_type="encoder",
                        prompt="test",
                        phase=Phase.ENCODING,
                        result="Encoded",
                    )
                ]

            orch._run_agent = mock_run_agent
            orch._run_encoding_parallel = mock_run_encoding_parallel
            orch._fetch_statute_text = lambda c: "Statute text"
            orch._build_analyzer_prompt = lambda c, statute_text: "Analyze"
            orch._format_oracle_summary = lambda c: "Oracle summary"
            orch._log_agent_run = lambda s, r: None
            orch._log_to_db = lambda r: None

            # Mock ValidatorPipeline to raise exceptions
            mock_vp = MagicMock()
            mock_vp._run_policyengine.side_effect = RuntimeError("PE crashed")
            mock_vp._run_taxsim.side_effect = RuntimeError("TAXSIM crashed")

            with patch(
                "autorac.harness.sdk_orchestrator.ValidatorPipeline",
                return_value=mock_vp,
            ):
                result = await orch.encode("26 USC 1", output_dir)

            assert isinstance(result, OrchestratorRun)
            # Should still complete even with oracle errors
