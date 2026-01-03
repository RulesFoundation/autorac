"""
SDK-based Encoding Orchestrator with full logging.

Uses Claude Agent SDK for complete control over the encoding workflow.
Logs EVERYTHING: every message, tool call, response, token counts.

This is the scientific-grade orchestrator for calibration experiments.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, AsyncIterator
from enum import Enum

from .experiment_db import ExperimentDB, TokenUsage


class Phase(Enum):
    ANALYSIS = "analysis"
    ENCODING = "encoding"
    ORACLE = "oracle"
    REVIEW = "review"
    REPORT = "report"


@dataclass
class AgentMessage:
    """A single message in an agent conversation."""
    role: str  # "user", "assistant", "tool_use", "tool_result"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    tokens: Optional[TokenUsage] = None
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None


@dataclass
class AgentRun:
    """Complete record of a single agent invocation."""
    agent_type: str
    prompt: str
    phase: Phase
    messages: List[AgentMessage] = field(default_factory=list)
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    total_tokens: Optional[TokenUsage] = None
    result: Optional[str] = None
    error: Optional[str] = None


@dataclass
class OrchestratorRun:
    """Complete record of an orchestration run."""
    citation: str
    session_id: str
    started_at: datetime = field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None

    # All agent runs, in order
    agent_runs: List[AgentRun] = field(default_factory=list)

    # Results
    files_created: List[str] = field(default_factory=list)
    oracle_pe_match: Optional[float] = None
    oracle_taxsim_match: Optional[float] = None
    discrepancies: List[dict] = field(default_factory=list)

    # Totals
    total_tokens: Optional[TokenUsage] = None
    total_cost_usd: float = 0.0


class SDKOrchestrator:
    """
    SDK-based orchestrator with full logging.

    Uses Agent SDK to invoke agents with complete control and logging.
    Every message, tool call, and token count is captured.
    """

    # Agent definitions from cosilico-claude plugin
    AGENTS = {
        "analyzer": "cosilico:Statute Analyzer",
        "encoder": "cosilico:RAC Encoder",
        "validator": "cosilico:Encoding Validator",
        "rac_reviewer": "cosilico:rac-reviewer",
        "formula_reviewer": "cosilico:Formula Reviewer",
        "parameter_reviewer": "cosilico:Parameter Reviewer",
        "integration_reviewer": "cosilico:Integration Reviewer",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        plugin_path: Optional[Path] = None,
        experiment_db: Optional[ExperimentDB] = None,
    ):
        import os
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")

        self.model = model
        self.plugin_path = plugin_path or Path(__file__).parent.parent.parent.parent.parent / "cosilico-claude"
        self.experiment_db = experiment_db

    async def encode(
        self,
        citation: str,
        output_path: Path,
        statute_text: Optional[str] = None,
    ) -> OrchestratorRun:
        """
        Run the full 5-phase encoding workflow with complete logging.

        Returns an OrchestratorRun with every agent invocation recorded.
        """
        run = OrchestratorRun(
            citation=citation,
            session_id=f"sdk-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}",
        )

        try:
            # Phase 1: Analysis
            analysis = await self._run_agent(
                agent_type=self.AGENTS["analyzer"],
                prompt=f"Analyze {citation}. Report: subsection tree, encoding order, dependencies.",
                phase=Phase.ANALYSIS,
                model="claude-haiku-3-5-20241022",
            )
            run.agent_runs.append(analysis)

            # Phase 2: Encoding
            encode_prompt = f"""Encode {citation} into RAC format.
Output path: {output_path}
{f'Statute text: {statute_text[:5000]}' if statute_text else 'Fetch statute text as needed.'}

Write .rac files to the output path. Run tests after each file."""

            encoding = await self._run_agent(
                agent_type=self.AGENTS["encoder"],
                prompt=encode_prompt,
                phase=Phase.ENCODING,
                model="claude-sonnet-4-20250514",
            )
            run.agent_runs.append(encoding)

            # Check what files were created
            if output_path.exists():
                run.files_created = [str(f) for f in output_path.rglob("*.rac")]

            # Phase 3: Oracle validation
            oracle = await self._run_agent(
                agent_type=self.AGENTS["validator"],
                prompt=f"Validate {citation} encoding at {output_path} against PolicyEngine and TAXSIM. Report match rates and specific discrepancies.",
                phase=Phase.ORACLE,
                model="claude-sonnet-4-20250514",
            )
            run.agent_runs.append(oracle)

            # Extract oracle results from response
            oracle_context = self._extract_oracle_context(oracle.result or "")
            run.oracle_pe_match = oracle_context.get("pe_match")
            run.oracle_taxsim_match = oracle_context.get("taxsim_match")
            run.discrepancies = oracle_context.get("discrepancies", [])

            # Phase 4: LLM Review (parallel)
            oracle_summary = self._format_oracle_summary(oracle_context)

            reviews = await asyncio.gather(
                self._run_agent(
                    agent_type=self.AGENTS["formula_reviewer"],
                    prompt=f"Review {citation} formulas. Oracle found: {oracle_summary}",
                    phase=Phase.REVIEW,
                    model="claude-haiku-3-5-20241022",
                ),
                self._run_agent(
                    agent_type=self.AGENTS["parameter_reviewer"],
                    prompt=f"Review {citation} parameters. Oracle found: {oracle_summary}",
                    phase=Phase.REVIEW,
                    model="claude-haiku-3-5-20241022",
                ),
                self._run_agent(
                    agent_type=self.AGENTS["integration_reviewer"],
                    prompt=f"Review {citation} integration. Oracle found: {oracle_summary}",
                    phase=Phase.REVIEW,
                    model="claude-haiku-3-5-20241022",
                ),
            )
            run.agent_runs.extend(reviews)

            # Phase 5: Report (computed, not an agent)
            run.ended_at = datetime.utcnow()
            run.total_tokens = self._sum_tokens(run.agent_runs)
            run.total_cost_usd = run.total_tokens.estimated_cost_usd if run.total_tokens else 0.0

            # Log to experiment DB if available
            if self.experiment_db:
                self._log_to_db(run)

        except Exception as e:
            run.ended_at = datetime.utcnow()
            # Log error but don't lose partial results
            if run.agent_runs:
                run.agent_runs[-1].error = str(e)

        return run

    async def _run_agent(
        self,
        agent_type: str,
        prompt: str,
        phase: Phase,
        model: str,
    ) -> AgentRun:
        """Run a single agent and capture everything."""
        from claude_code_sdk import query, ClaudeCodeOptions

        run = AgentRun(
            agent_type=agent_type,
            prompt=prompt,
            phase=phase,
        )

        try:
            options = ClaudeCodeOptions(
                model=model,
                allowed_tools=["Read", "Write", "Edit", "Grep", "Glob", "Bash", "Task"],
            )

            # Collect all messages
            total_input = 0
            total_output = 0
            total_cache_read = 0

            async for event in query(prompt=prompt, options=options):
                msg = AgentMessage(
                    role=getattr(event, "type", "unknown"),
                    content="",
                )

                # Capture content based on event type
                if hasattr(event, "content"):
                    msg.content = str(event.content)[:10000]
                elif hasattr(event, "result"):
                    msg.content = str(event.result)[:10000]
                    run.result = event.result

                # Capture tool use
                if hasattr(event, "tool_name"):
                    msg.tool_name = event.tool_name
                if hasattr(event, "tool_input"):
                    msg.tool_input = event.tool_input
                if hasattr(event, "tool_output"):
                    msg.tool_output = str(event.tool_output)[:10000]

                # Capture tokens
                if hasattr(event, "usage"):
                    usage = event.usage
                    msg.tokens = TokenUsage(
                        input_tokens=getattr(usage, "input_tokens", 0),
                        output_tokens=getattr(usage, "output_tokens", 0),
                        cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0),
                    )
                    total_input += msg.tokens.input_tokens
                    total_output += msg.tokens.output_tokens
                    total_cache_read += msg.tokens.cache_read_tokens

                run.messages.append(msg)

            run.ended_at = datetime.utcnow()
            run.total_tokens = TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                cache_read_tokens=total_cache_read,
            )

        except Exception as e:
            run.error = str(e)
            run.ended_at = datetime.utcnow()

        return run

    def _extract_oracle_context(self, result: str) -> dict:
        """Extract structured oracle results from validator output."""
        context = {
            "pe_match": None,
            "taxsim_match": None,
            "discrepancies": [],
        }

        # Simple extraction - look for percentages
        import re
        pe_match = re.search(r'PolicyEngine[:\s]+(\d+(?:\.\d+)?)\s*%', result, re.I)
        if pe_match:
            context["pe_match"] = float(pe_match.group(1))

        taxsim_match = re.search(r'TAXSIM[:\s]+(\d+(?:\.\d+)?)\s*%', result, re.I)
        if taxsim_match:
            context["taxsim_match"] = float(taxsim_match.group(1))

        # Extract discrepancy lines
        for line in result.split('\n'):
            if 'discrepancy' in line.lower() or 'differs' in line.lower():
                context["discrepancies"].append({"description": line.strip()})

        return context

    def _format_oracle_summary(self, context: dict) -> str:
        """Format oracle context for reviewer prompts."""
        parts = []
        if context.get("pe_match") is not None:
            parts.append(f"PE match: {context['pe_match']}%")
        if context.get("taxsim_match") is not None:
            parts.append(f"TAXSIM match: {context['taxsim_match']}%")
        if context.get("discrepancies"):
            parts.append(f"Discrepancies: {len(context['discrepancies'])}")
            for d in context["discrepancies"][:3]:
                parts.append(f"  - {d['description'][:100]}")
        return "\n".join(parts) if parts else "No oracle data available"

    def _sum_tokens(self, runs: List[AgentRun]) -> TokenUsage:
        """Sum token usage across all agent runs."""
        total = TokenUsage()
        for run in runs:
            if run.total_tokens:
                total.input_tokens += run.total_tokens.input_tokens
                total.output_tokens += run.total_tokens.output_tokens
                total.cache_read_tokens += run.total_tokens.cache_read_tokens
        return total

    def _log_to_db(self, run: OrchestratorRun) -> None:
        """Log the complete run to experiment database."""
        if not self.experiment_db:
            return

        # Create session
        self.experiment_db.create_session(run.session_id, self.model, str(Path.cwd()))

        # Log each agent run as events
        for agent_run in run.agent_runs:
            # Log agent start
            self.experiment_db.log_event(
                session_id=run.session_id,
                event_type="agent_start",
                content=agent_run.prompt,
                metadata={
                    "agent_type": agent_run.agent_type,
                    "phase": agent_run.phase.value,
                }
            )

            # Log each message
            for msg in agent_run.messages:
                self.experiment_db.log_event(
                    session_id=run.session_id,
                    event_type=f"agent_{msg.role}",
                    tool_name=msg.tool_name,
                    content=msg.content,
                    metadata={
                        "agent_type": agent_run.agent_type,
                        "tool_input": msg.tool_input,
                        "tool_output": msg.tool_output[:1000] if msg.tool_output else None,
                        "tokens": {
                            "input": msg.tokens.input_tokens if msg.tokens else 0,
                            "output": msg.tokens.output_tokens if msg.tokens else 0,
                        } if msg.tokens else None,
                    }
                )

            # Log agent end
            self.experiment_db.log_event(
                session_id=run.session_id,
                event_type="agent_end",
                content=agent_run.result or "",
                metadata={
                    "agent_type": agent_run.agent_type,
                    "phase": agent_run.phase.value,
                    "error": agent_run.error,
                    "total_tokens": {
                        "input": agent_run.total_tokens.input_tokens,
                        "output": agent_run.total_tokens.output_tokens,
                    } if agent_run.total_tokens else None,
                }
            )

        # Update session totals
        if run.total_tokens:
            self.experiment_db.update_session_tokens(
                session_id=run.session_id,
                input_tokens=run.total_tokens.input_tokens,
                output_tokens=run.total_tokens.output_tokens,
                cache_read_tokens=run.total_tokens.cache_read_tokens,
            )

    def print_report(self, run: OrchestratorRun) -> str:
        """Generate human-readable report."""
        lines = [
            f"# Encoding Report: {run.citation}",
            f"Session: {run.session_id}",
            f"Duration: {(run.ended_at - run.started_at).total_seconds():.1f}s" if run.ended_at else "In progress",
            "",
            "## Oracle Match Rates",
            f"- PolicyEngine: {run.oracle_pe_match}%" if run.oracle_pe_match else "- PolicyEngine: N/A",
            f"- TAXSIM: {run.oracle_taxsim_match}%" if run.oracle_taxsim_match else "- TAXSIM: N/A",
            "",
            "## Files Created",
        ]
        for f in run.files_created:
            lines.append(f"- {f}")

        lines.extend([
            "",
            "## Agent Runs",
        ])
        for agent_run in run.agent_runs:
            tokens = agent_run.total_tokens
            lines.append(
                f"- {agent_run.phase.value}: {agent_run.agent_type} "
                f"({tokens.total_tokens if tokens else 0} tokens, "
                f"${tokens.estimated_cost_usd:.4f})" if tokens else f"- {agent_run.phase.value}: {agent_run.agent_type}"
            )

        if run.total_tokens:
            lines.extend([
                "",
                "## Totals",
                f"- Input tokens: {run.total_tokens.input_tokens:,}",
                f"- Output tokens: {run.total_tokens.output_tokens:,}",
                f"- Total cost: ${run.total_cost_usd:.4f}",
            ])

        return "\n".join(lines)
