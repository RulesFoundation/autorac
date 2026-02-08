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
            rac_content="variable eitc:\n  entity: TaxUnit",
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
                stdout="variable test:\n  entity: TaxUnit",
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
        """encode() uses the cosilico:RAC Encoder agent."""
        backend = ClaudeCodeBackend(plugin_dir=Path("/path/to/plugins"))

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="variable test:\n  entity: TaxUnit",
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
                yield MockMessage(result="variable test:\n  entity: TaxUnit")

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


class TestBackendCompatibility:
    """Test that both backends produce compatible outputs."""

    def test_both_backends_return_encoder_response(self):
        """Both backends return EncoderResponse from encode()."""
        # This ensures the abstraction is clean

        with patch("subprocess.run") as mock_run:
            mock_run.return_value = Mock(
                stdout="variable test:\n  entity: TaxUnit",
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
