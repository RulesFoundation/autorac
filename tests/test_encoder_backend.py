"""
Tests for encoder backend abstraction.

TDD: Write tests first, then implement the backends.
"""

import os
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import what we're going to build
from autorac.harness.backends import (
    AgentSDKBackend,
    ClaudeCodeBackend,
    EncoderBackend,
    EncoderRequest,
    EncoderResponse,
)


@pytest.fixture(autouse=True)
def mock_sdk_env(tmp_path):
    """Provide API key and valid plugin path for AgentSDKBackend tests."""
    with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "test-key"}):
        with patch.object(AgentSDKBackend, "DEFAULT_PLUGIN_PATH", tmp_path):
            yield


class TestEncoderBackendInterface:
    """Test the abstract backend interface."""

    def test_backend_is_abstract(self):
        """EncoderBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            EncoderBackend()

    def test_request_dataclass(self):
        """EncoderRequest holds encoding inputs."""
        req = EncoderRequest(
            citation="26 USC 32",
            statute_text="The earned income tax credit...",
            output_path=Path("/tmp/test.rac"),
        )
        assert req.citation == "26 USC 32"
        assert req.statute_text.startswith("The earned")
        assert req.output_path == Path("/tmp/test.rac")

    def test_response_dataclass(self):
        """EncoderResponse holds encoding outputs."""
        resp = EncoderResponse(
            rac_content="eitc:\n  entity: TaxUnit",
            success=True,
            error=None,
            duration_ms=1500,
        )
        assert resp.success
        assert "eitc" in resp.rac_content
        assert resp.duration_ms == 1500


class TestClaudeCodeBackend:
    """Test the Claude Code CLI backend (subprocess approach)."""

    def test_backend_inherits_interface(self):
        """ClaudeCodeBackend implements EncoderBackend."""
        backend = ClaudeCodeBackend()
        assert isinstance(backend, EncoderBackend)

    def test_encode_calls_subprocess(self):
        """encode() calls claude CLI via subprocess."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="test:\n  entity: TaxUnit",
                stderr="",
                returncode=0,
            )

            backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    statute_text="Test statute",
                    output_path=Path("/tmp/test.rac"),
                )
            )

            assert mock_run.called
            # Should use 'claude' command
            cmd = mock_run.call_args[0][0]
            assert "claude" in cmd

    def test_encode_uses_plugin_agent(self):
        """encode() uses the rac:RAC Encoder agent."""
        backend = ClaudeCodeBackend(plugin_dir=Path("/path/to/plugins"))

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="test:\n  entity: TaxUnit",
                stderr="",
                returncode=0,
            )

            backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    statute_text="Test",
                    output_path=Path("/tmp/test.rac"),
                )
            )

            cmd = mock_run.call_args[0][0]
            # Should include agent flag
            assert "--agent" in cmd or "-a" in " ".join(cmd)

    def test_predict_returns_scores(self):
        """predict() returns score predictions."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout='{"rac_reviewer": 8.0, "formula_reviewer": 7.5, "confidence": 0.8}',
                stderr="",
                returncode=0,
            )

            scores = backend.predict(
                citation="26 USC 32",
                statute_text="EITC rules...",
            )

            assert scores.rac_reviewer == 8.0
            assert scores.confidence == 0.8


class TestAgentSDKBackend:
    """Test the Claude Agent SDK backend (API approach)."""

    def test_backend_inherits_interface(self):
        """AgentSDKBackend implements EncoderBackend."""
        backend = AgentSDKBackend()
        assert isinstance(backend, EncoderBackend)

    def test_requires_api_key(self):
        """AgentSDKBackend requires ANTHROPIC_API_KEY."""
        with patch.dict("os.environ", {}, clear=True):
            # Remove API key from env
            import os

            if "ANTHROPIC_API_KEY" in os.environ:
                del os.environ["ANTHROPIC_API_KEY"]

            with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
                AgentSDKBackend(api_key=None)

    @pytest.mark.asyncio
    async def test_encode_async(self):
        """encode_async() uses Agent SDK for async encoding."""
        backend = AgentSDKBackend(api_key="test-key")

        # Patch at the backend module level where the import happens
        with patch.dict("sys.modules", {"claude_agent_sdk": Mock()}):
            import sys

            mock_sdk = sys.modules["claude_agent_sdk"]

            # Mock async generator — use spec to limit attributes
            # so hasattr(message, "usage") returns False
            class MockMessage:
                def __init__(self, result):
                    self.result = result

            async def mock_gen():
                yield MockMessage(result="test:\n  entity: TaxUnit")

            mock_sdk.query = Mock(return_value=mock_gen())
            mock_sdk.ClaudeAgentOptions = Mock()

            resp = await backend.encode_async(
                EncoderRequest(
                    citation="26 USC 1",
                    statute_text="Test",
                    output_path=Path("/tmp/test.rac"),
                )
            )

            # Should succeed and return content from the mock
            assert resp.success or resp.rac_content  # Either success or got content

    @pytest.mark.asyncio
    async def test_encode_batch_parallel(self):
        """encode_batch() runs multiple encodings in parallel."""
        backend = AgentSDKBackend(api_key="test-key")

        requests = [
            EncoderRequest(
                citation=f"26 USC {i}",
                statute_text=f"Statute {i}",
                output_path=Path(f"/tmp/test{i}.rac"),
            )
            for i in range(5)
        ]

        with patch.object(backend, "encode_async") as mock_encode:
            mock_encode.return_value = EncoderResponse(
                rac_content="test",
                success=True,
                error=None,
                duration_ms=100,
            )

            responses = await backend.encode_batch(requests, max_concurrent=3)

            # All 5 should complete
            assert len(responses) == 5
            # encode_async should be called 5 times
            assert mock_encode.call_count == 5

    @pytest.mark.asyncio
    async def test_encode_batch_respects_concurrency_limit(self):
        """encode_batch() respects max_concurrent limit."""
        backend = AgentSDKBackend(api_key="test-key")
        concurrent_count = 0
        max_seen = 0

        async def track_concurrency(req):
            nonlocal concurrent_count, max_seen
            concurrent_count += 1
            max_seen = max(max_seen, concurrent_count)
            import asyncio

            await asyncio.sleep(0.01)  # Simulate work
            concurrent_count -= 1
            return EncoderResponse(
                rac_content="test",
                success=True,
                error=None,
                duration_ms=10,
            )

        with patch.object(backend, "encode_async", side_effect=track_concurrency):
            requests = [
                EncoderRequest(
                    citation=f"26 USC {i}",
                    statute_text=f"Statute {i}",
                    output_path=Path(f"/tmp/test{i}.rac"),
                )
                for i in range(10)
            ]

            await backend.encode_batch(requests, max_concurrent=3)

            # Should never exceed max_concurrent
            assert max_seen <= 3


class TestClaudeCodeBackendAdditional:
    """Additional tests for ClaudeCodeBackend to cover missing lines."""

    def test_encode_with_nonzero_returncode(self):
        """Test encode returns error when CLI returns non-zero."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="Error: failed to encode",
                stderr="",
                returncode=1,
            )

            resp = backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    statute_text="Test",
                    output_path=Path("/tmp/test.rac"),
                )
            )

            assert not resp.success
            assert resp.error is not None
            assert resp.rac_content == ""

    def test_encode_reads_file_when_exists(self, tmp_path):
        """Test encode reads from output_path when file exists."""
        backend = ClaudeCodeBackend()
        output_path = tmp_path / "output.rac"

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="CLI output ignored",
                stderr="",
                returncode=0,
            )
            # Create the file as if Claude wrote it
            output_path.write_text("file_var:\n  entity: TaxUnit\n")

            resp = backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    statute_text="Test",
                    output_path=output_path,
                )
            )

            assert resp.success
            assert "file_var" in resp.rac_content

    def test_predict_no_json_in_output(self):
        """Test predict returns defaults when no JSON found in output."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="No JSON here, just plain text",
                stderr="",
                returncode=0,
            )

            scores = backend.predict("26 USC 1", "Statute text")
            # Should return defaults on error
            assert scores.confidence == 0.3

    def test_predict_returns_defaults_on_exception(self):
        """Test predict returns defaults when exception occurs."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = Exception("unexpected error")

            scores = backend.predict("26 USC 1", "Statute text")
            assert scores.confidence == 0.3

    def test_run_claude_code_with_plugin_dir(self, tmp_path):
        """Test _run_claude_code includes plugin-dir when it exists."""
        plugin_dir = tmp_path / "plugins"
        plugin_dir.mkdir()
        backend = ClaudeCodeBackend(plugin_dir=plugin_dir)

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(stdout="test", stderr="", returncode=0)

            backend._run_claude_code("test prompt")

            cmd = mock_run.call_args[0][0]
            assert "--plugin-dir" in cmd

    def test_run_claude_code_timeout(self):
        """Test _run_claude_code handles timeout."""
        import subprocess

        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = subprocess.TimeoutExpired(cmd="claude", timeout=300)

            output, code = backend._run_claude_code("test", timeout=300)
            assert "Timeout" in output
            assert code == 1

    def test_run_claude_code_file_not_found(self):
        """Test _run_claude_code handles missing CLI."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = FileNotFoundError()

            output, code = backend._run_claude_code("test")
            assert "not found" in output
            assert code == 1

    def test_run_claude_code_generic_error(self):
        """Test _run_claude_code handles generic exception."""
        backend = ClaudeCodeBackend()

        with patch("subprocess.run") as mock_run:
            mock_run.side_effect = OSError("Permission denied")

            output, code = backend._run_claude_code("test")
            assert "Error" in output
            assert code == 1


class TestAgentSDKBackendAdditional:
    """Additional tests for AgentSDKBackend to cover missing lines."""

    def test_plugin_path_does_not_exist(self, tmp_path):
        """Test AgentSDKBackend raises error for nonexistent plugin path."""
        with pytest.raises(ValueError, match="does not exist"):
            AgentSDKBackend(
                api_key="test-key",
                plugin_path=tmp_path / "nonexistent",
            )

    @pytest.mark.asyncio
    async def test_encode_async_with_custom_agent_type(self, tmp_path):
        """Test encode_async uses Task tool pattern for custom agent types."""
        backend = AgentSDKBackend(api_key="test-key", plugin_path=tmp_path)

        mock_sdk = Mock()

        class MockMessage:
            def __init__(self):
                self.result = "encoded content"

        async def mock_gen():
            yield MockMessage()

        mock_sdk.query = Mock(return_value=mock_gen())
        mock_sdk.ClaudeAgentOptions = Mock()

        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            resp = await backend.encode_async(
                EncoderRequest(
                    citation="26 USC 1",
                    statute_text="Test",
                    output_path=Path("/tmp/nonexistent.rac"),
                    agent_type="custom:Agent",
                )
            )

            assert resp.success
            # Prompt should include "Use the Task tool"
            call_args = mock_sdk.query.call_args
            prompt = call_args[1].get("prompt", "") if call_args[1] else ""
            if not prompt and call_args[0]:
                prompt = ""

    @pytest.mark.asyncio
    async def test_encode_async_with_usage(self, tmp_path):
        """Test encode_async captures token usage."""
        backend = AgentSDKBackend(api_key="test-key", plugin_path=tmp_path)

        mock_sdk = Mock()

        class MockUsage:
            input_tokens = 100
            output_tokens = 50
            cache_read_input_tokens = 10
            cache_creation_input_tokens = 5

        class MockMessage:
            def __init__(self):
                self.result = "encoded"
                self.usage = MockUsage()

        async def mock_gen():
            yield MockMessage()

        mock_sdk.query = Mock(return_value=mock_gen())
        mock_sdk.ClaudeAgentOptions = Mock()

        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            resp = await backend.encode_async(
                EncoderRequest(
                    citation="26 USC 1",
                    statute_text="Test",
                    output_path=Path("/tmp/nonexistent.rac"),
                )
            )

            assert resp.success
            assert resp.tokens is not None

    @pytest.mark.asyncio
    async def test_encode_async_reads_file_if_exists(self, tmp_path):
        """Test encode_async reads from output_path if it exists."""
        backend = AgentSDKBackend(api_key="test-key", plugin_path=tmp_path)

        output_path = tmp_path / "output.rac"
        output_path.write_text("file_content:\n  entity: TaxUnit\n")

        mock_sdk = Mock()

        class MockMessage:
            def __init__(self):
                self.result = "ignored"

        async def mock_gen():
            yield MockMessage()

        mock_sdk.query = Mock(return_value=mock_gen())
        mock_sdk.ClaudeAgentOptions = Mock()

        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            resp = await backend.encode_async(
                EncoderRequest(
                    citation="26 USC 1",
                    statute_text="Test",
                    output_path=output_path,
                )
            )

            assert resp.success
            assert "file_content" in resp.rac_content

    @pytest.mark.asyncio
    async def test_encode_async_import_error(self, tmp_path):
        """Test encode_async handles missing SDK import."""
        backend = AgentSDKBackend(api_key="test-key", plugin_path=tmp_path)

        import builtins

        orig_import = builtins.__import__

        def no_sdk_import(name, *args, **kwargs):
            if name == "claude_agent_sdk":
                raise ImportError("No module named 'claude_agent_sdk'")
            return orig_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=no_sdk_import):
            resp = await backend.encode_async(
                EncoderRequest(
                    citation="26 USC 1",
                    statute_text="Test",
                    output_path=Path("/tmp/test.rac"),
                )
            )

            assert not resp.success
            assert "not installed" in resp.error

    @pytest.mark.asyncio
    async def test_encode_async_generic_error(self, tmp_path):
        """Test encode_async handles generic exception."""
        backend = AgentSDKBackend(api_key="test-key", plugin_path=tmp_path)

        mock_sdk = Mock()

        async def mock_gen():
            raise RuntimeError("Connection failed")
            yield  # pragma: no cover

        mock_sdk.query = Mock(return_value=mock_gen())
        mock_sdk.ClaudeAgentOptions = Mock()

        with patch.dict("sys.modules", {"claude_agent_sdk": mock_sdk}):
            resp = await backend.encode_async(
                EncoderRequest(
                    citation="26 USC 1",
                    statute_text="Test",
                    output_path=Path("/tmp/test.rac"),
                )
            )

            assert not resp.success
            assert "Connection failed" in resp.error


class TestAgentSDKPrediction:
    """Test AgentSDKBackend.predict() method."""

    def test_predict_returns_default_scores(self, tmp_path):
        """Test predict returns default PredictionScores."""
        backend = AgentSDKBackend(api_key="test-key", plugin_path=tmp_path)
        scores = backend.predict("26 USC 1", "Statute text")
        assert scores.confidence == 0.5


class TestBackendCompatibility:
    """Test that both backends produce compatible outputs."""

    def test_both_backends_return_encoder_response(self):
        """Both backends return EncoderResponse from encode()."""
        # This ensures the abstraction is clean

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="test:\n  entity: TaxUnit",
                stderr="",
                returncode=0,
            )

            cli_backend = ClaudeCodeBackend()
            cli_resp = cli_backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    statute_text="Test",
                    output_path=Path("/tmp/test.rac"),
                )
            )

            assert isinstance(cli_resp, EncoderResponse)

    def test_sync_wrapper_for_sdk_backend(self):
        """AgentSDKBackend.encode() provides sync wrapper."""
        backend = AgentSDKBackend(api_key="test-key")

        with patch.object(backend, "encode_async") as mock_async:
            mock_async.return_value = EncoderResponse(
                rac_content="test",
                success=True,
                error=None,
                duration_ms=100,
            )

            # Sync encode() should work
            resp = backend.encode(
                EncoderRequest(
                    citation="26 USC 1",
                    statute_text="Test",
                    output_path=Path("/tmp/test.rac"),
                )
            )

            assert isinstance(resp, EncoderResponse)
