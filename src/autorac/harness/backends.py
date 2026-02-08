"""
Encoder backends - abstraction for different Claude invocation methods.

Two backends:
1. ClaudeCodeBackend - uses Claude Code CLI (subprocess), works with Max subscription
2. AgentSDKBackend - uses Claude Agent SDK (API), enables massive parallelization

Both implement the same interface, so EncoderHarness can use either.
"""

import asyncio
import json
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class EncoderRequest:
    """Input for an encoding operation."""

    citation: str
    statute_text: str
    output_path: Path
    agent_type: str = "cosilico:RAC Encoder"
    model: str = "opus"
    timeout: int = 300


@dataclass
class TokenUsage:
    """Token usage from an encoding operation."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_creation_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def estimated_cost_usd(self) -> float:
        """Rough cost estimate (Opus pricing as of 2025)."""
        return (
            self.input_tokens * 15 / 1_000_000
            + self.output_tokens * 75 / 1_000_000
            + self.cache_read_tokens * 1.875 / 1_000_000
        )


@dataclass
class EncoderResponse:
    """Output from an encoding operation."""

    rac_content: str
    success: bool
    error: Optional[str] = None
    duration_ms: int = 0
    tokens: Optional[TokenUsage] = None


@dataclass
class PredictionScores:
    """Predicted scores from the encoder."""

    rac_reviewer: float = 7.0
    formula_reviewer: float = 7.0
    parameter_reviewer: float = 7.0
    integration_reviewer: float = 7.0
    ci_pass: bool = True
    policyengine_match: Optional[float] = None
    taxsim_match: Optional[float] = None
    confidence: float = 0.5


class EncoderBackend(ABC):
    """Abstract base class for encoder backends."""

    @abstractmethod
    def encode(self, request: EncoderRequest) -> EncoderResponse:
        """Encode a statute to RAC format (synchronous)."""
        pass

    @abstractmethod
    def predict(self, citation: str, statute_text: str) -> PredictionScores:
        """Predict quality scores before encoding."""
        pass


class ClaudeCodeBackend(EncoderBackend):
    """
    Backend using Claude Code CLI (subprocess).

    Works with Max subscription - no API billing.
    Best for interactive use.
    """

    def __init__(
        self,
        plugin_dir: Optional[Path] = None,
        cwd: Optional[Path] = None,
    ):
        self.plugin_dir = plugin_dir
        self.cwd = cwd or Path.cwd()

    def encode(self, request: EncoderRequest) -> EncoderResponse:
        """Encode using Claude Code CLI."""
        start = time.time()

        prompt = f"""Encode {request.citation} into RAC format.

Write the output to: {request.output_path}

Statute Text:
{request.statute_text}

Use the Write tool to create the .rac file at the specified path.
"""

        output, returncode = self._run_claude_code(
            prompt=prompt,
            agent=request.agent_type,
            model=request.model,
            timeout=request.timeout,
        )

        duration_ms = int((time.time() - start) * 1000)

        if returncode != 0:
            return EncoderResponse(
                rac_content="",
                success=False,
                error=output,
                duration_ms=duration_ms,
            )

        # Check if file was created
        if request.output_path.exists():
            rac_content = request.output_path.read_text()
        else:
            rac_content = output

        # Clean up markdown code blocks
        rac_content = re.sub(r"^```\w*\n", "", rac_content)
        rac_content = re.sub(r"\n```$", "", rac_content)
        rac_content = rac_content.strip()

        return EncoderResponse(
            rac_content=rac_content,
            success=True,
            error=None,
            duration_ms=duration_ms,
        )

    def predict(self, citation: str, statute_text: str) -> PredictionScores:
        """Predict scores using Claude Code CLI."""
        prompt = f"""Predict quality scores for encoding the following statute into RAC DSL.

Citation: {citation}

Statute Text:
{statute_text[:2000]}{"..." if len(statute_text) > 2000 else ""}

Score each dimension from 1-10. Output ONLY valid JSON:
{{
  "rac_reviewer": <float 1-10>,
  "formula_reviewer": <float 1-10>,
  "parameter_reviewer": <float 1-10>,
  "integration_reviewer": <float 1-10>,
  "ci_pass": <boolean>,
  "policyengine_match": <float 0-1>,
  "taxsim_match": <float 0-1>,
  "confidence": <float 0-1>
}}
"""

        try:
            output, returncode = self._run_claude_code(
                prompt=prompt,
                model="opus",
                timeout=60,
            )

            # Parse JSON from output
            json_match = re.search(r"\{[^{}]*\}", output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in output")

            return PredictionScores(
                rac_reviewer=float(data.get("rac_reviewer", 7.0)),
                formula_reviewer=float(data.get("formula_reviewer", 7.0)),
                parameter_reviewer=float(data.get("parameter_reviewer", 7.0)),
                integration_reviewer=float(data.get("integration_reviewer", 7.0)),
                ci_pass=bool(data.get("ci_pass", True)),
                policyengine_match=data.get("policyengine_match"),
                taxsim_match=data.get("taxsim_match"),
                confidence=float(data.get("confidence", 0.5)),
            )

        except Exception:
            # Return defaults on error
            return PredictionScores(confidence=0.3)

    def _run_claude_code(
        self,
        prompt: str,
        agent: Optional[str] = None,
        model: str = "sonnet",
        timeout: int = 300,
    ) -> tuple[str, int]:
        """Run Claude Code CLI as subprocess."""
        cmd = ["claude", "--print"]

        if model:
            cmd.extend(["--model", model])

        if self.plugin_dir and self.plugin_dir.exists():
            cmd.extend(["--plugin-dir", str(self.plugin_dir)])

        if agent:
            cmd.extend(["--agent", agent])

        cmd.extend(["-p", prompt])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=self.cwd,
            )
            return result.stdout + result.stderr, result.returncode
        except subprocess.TimeoutExpired:
            return f"Timeout after {timeout}s", 1
        except FileNotFoundError:
            return (
                "Claude CLI not found - install with: npm install -g @anthropic-ai/claude-code",
                1,
            )
        except Exception as e:
            return f"Error running Claude CLI: {e}", 1


class AgentSDKBackend(EncoderBackend):
    """
    Backend using Claude Agent SDK (API).

    Requires ANTHROPIC_API_KEY - pay per token.
    Enables massive parallelization for batch encoding.

    Uses the same agent definitions as the Claude Code plugin by loading
    the cosilico-claude plugin via the SDK's plugins option.
    """

    # Default path to cosilico-claude plugin (relative to CosilicoAI)
    DEFAULT_PLUGIN_PATH = (
        Path(__file__).parent.parent.parent.parent.parent / "cosilico-claude"
    )

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-5-20251101",
        plugin_path: Optional[Path] = None,
    ):
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required for AgentSDKBackend")
        self.model = model
        self.plugin_path = (
            Path(plugin_path) if plugin_path else self.DEFAULT_PLUGIN_PATH
        )

        if not self.plugin_path.exists():
            raise ValueError(f"Plugin path does not exist: {self.plugin_path}")

    def encode(self, request: EncoderRequest) -> EncoderResponse:
        """Synchronous encode using Agent SDK (runs async under the hood)."""
        return asyncio.run(self.encode_async(request))

    async def encode_async(self, request: EncoderRequest) -> EncoderResponse:
        """Async encode using Agent SDK with cosilico-claude plugin.

        Uses the same agent definitions as Claude Code CLI by loading
        the cosilico-claude plugin.
        """
        start = time.time()

        try:
            # Import here to avoid dependency if not using SDK backend
            from claude_agent_sdk import ClaudeAgentOptions, query

            prompt = f"""Encode {request.citation} into RAC format.

Write the output to: {request.output_path}

Statute Text:
{request.statute_text}

Use the Write tool to create the .rac file at the specified path.
"""

            # Configure SDK to load the cosilico-claude plugin
            # This gives access to the same agents as Claude Code CLI
            options = ClaudeAgentOptions(
                model=self.model,
                allowed_tools=["Read", "Write", "Edit", "Grep", "Glob", "Task", "Bash"],
                # Load cosilico-claude plugin for agent definitions
                plugins=[{"type": "local", "path": str(self.plugin_path)}],
                # Load agent definitions from plugin
                setting_sources=["project"],
            )

            # If a specific agent is requested, use the Task tool pattern
            # to invoke it (same as Claude Code CLI does)
            if request.agent_type and request.agent_type != "cosilico:RAC Encoder":
                prompt = f"""Use the Task tool to invoke the {request.agent_type} agent with this prompt:

{prompt}
"""

            result_content = ""
            token_usage = TokenUsage()

            async for message in query(prompt=prompt, options=options):
                if hasattr(message, "result"):
                    result_content = message.result
                # Capture token usage from API response
                if hasattr(message, "usage"):
                    usage = message.usage
                    token_usage = TokenUsage(
                        input_tokens=getattr(usage, "input_tokens", 0),
                        output_tokens=getattr(usage, "output_tokens", 0),
                        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0),
                        cache_creation_tokens=getattr(
                            usage, "cache_creation_input_tokens", 0
                        ),
                    )

            duration_ms = int((time.time() - start) * 1000)

            # Check if file was created
            if request.output_path.exists():
                rac_content = request.output_path.read_text()
            else:
                rac_content = result_content

            return EncoderResponse(
                rac_content=rac_content,
                success=True,
                error=None,
                duration_ms=duration_ms,
                tokens=token_usage if token_usage.total_tokens > 0 else None,
            )

        except ImportError:
            return EncoderResponse(
                rac_content="",
                success=False,
                error="claude_agent_sdk not installed. Run: pip install claude-agent-sdk",
                duration_ms=int((time.time() - start) * 1000),
            )
        except Exception as e:
            return EncoderResponse(
                rac_content="",
                success=False,
                error=str(e),
                duration_ms=int((time.time() - start) * 1000),
            )

    async def encode_batch(
        self,
        requests: List[EncoderRequest],
        max_concurrent: int = 5,
    ) -> List[EncoderResponse]:
        """
        Encode multiple statutes in parallel.

        This is the key advantage of the SDK backend - massive parallelization.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def encode_with_limit(req: EncoderRequest) -> EncoderResponse:
            async with semaphore:
                return await self.encode_async(req)

        tasks = [encode_with_limit(req) for req in requests]
        return await asyncio.gather(*tasks)

    def predict(self, citation: str, statute_text: str) -> PredictionScores:
        """Predict scores using Agent SDK."""
        # For now, use defaults - prediction is less critical than encoding
        return PredictionScores(confidence=0.5)
