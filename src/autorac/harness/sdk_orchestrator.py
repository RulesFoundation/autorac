"""
SDK-based Encoding Orchestrator with full logging.

Uses Claude Agent SDK for complete control over the encoding workflow.
Logs EVERYTHING: every message, tool call, response, token counts.

This is the scientific-grade orchestrator for calibration experiments.
"""

import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import List, Optional

from .experiment_db import ExperimentDB, TokenUsage
from .validator_pipeline import ValidatorPipeline


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
    summary: Optional[str] = None  # Human-readable summary of what happened


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
    total_cost: Optional[float] = None  # USD from SDK
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


def _summarize_tool_call(
    tool_name: str, tool_input: Optional[dict], tool_output: Optional[str]
) -> str:
    """Generate a human-readable summary of a tool call."""
    if not tool_name:
        return ""

    input_str = ""
    if tool_input:
        # Extract key info based on tool type
        if tool_name == "Read":
            path = tool_input.get("file_path", "unknown")
            input_str = f"'{path.split('/')[-1]}'"
        elif tool_name == "Write":
            path = tool_input.get("file_path", "unknown")
            content = tool_input.get("content", "")
            lines = content.count("\n") + 1 if content else 0
            input_str = f"'{path.split('/')[-1]}' ({lines} lines)"
        elif tool_name == "Edit":
            path = tool_input.get("file_path", "unknown")
            input_str = f"'{path.split('/')[-1]}'"
        elif tool_name == "Grep":
            pattern = tool_input.get("pattern", "")
            input_str = (
                f"pattern='{pattern[:30]}...'"
                if len(pattern) > 30
                else f"pattern='{pattern}'"
            )
        elif tool_name == "Glob":
            pattern = tool_input.get("pattern", "")
            input_str = f"'{pattern}'"
        elif tool_name == "Bash":
            cmd = tool_input.get("command", "")
            input_str = f"'{cmd[:50]}...'" if len(cmd) > 50 else f"'{cmd}'"
        elif tool_name == "Task":
            subagent = tool_input.get("subagent_type", "unknown")
            input_str = f"spawn {subagent}"

    # Add output summary
    output_str = ""
    if tool_output:
        output_len = len(tool_output)
        if output_len > 1000:
            output_str = f" → {output_len:,} chars"
        elif tool_name == "Grep" and "Found" in tool_output:
            # Extract match count
            output_str = f" → {tool_output.split()[1]} files"

    return f"{tool_name} {input_str}{output_str}".strip()


def _summarize_thinking(content: str) -> Optional[str]:
    """Extract first key insight from thinking/reasoning content."""
    if not content:
        return None

    # Look for thinking tags
    import re

    thinking_match = re.search(r"<thinking>([\s\S]*?)</thinking>", content)
    if thinking_match:
        thinking = thinking_match.group(1).strip()
        # Get first sentence or first 100 chars
        first_sentence = re.split(r"[.!?\n]", thinking)[0].strip()
        if first_sentence:
            return first_sentence[:150] + ("..." if len(first_sentence) > 150 else "")

    # Look for reasoning patterns
    for prefix in ["I need to", "Let me", "First,", "The statute", "This section"]:
        if prefix.lower() in content.lower()[:500]:
            idx = content.lower().find(prefix.lower())
            snippet = content[idx : idx + 150]
            first_sentence = re.split(r"[.!?\n]", snippet)[0].strip()
            if first_sentence:
                return first_sentence + ("..." if len(first_sentence) > 100 else "")

    return None


def _summarize_assistant_message(content: str) -> str:
    """Summarize an assistant message."""
    if not content:
        return "Empty response"

    # Check for common patterns
    if "Error" in content or "error" in content:
        return "Encountered an error"
    if "Successfully" in content or "Created" in content:
        return "Completed successfully"
    if "```" in content:
        return f"Code block response ({len(content):,} chars)"

    # Get first meaningful line
    lines = [
        line.strip()
        for line in content.split("\n")
        if line.strip() and not line.startswith("#")
    ]
    if lines:
        first = lines[0][:100]
        return first + ("..." if len(lines[0]) > 100 else "")

    return f"Response ({len(content):,} chars)"


class SDKOrchestrator:
    """
    SDK-based orchestrator with full logging.

    Uses Agent SDK to invoke agents with complete control and logging.
    Every message, tool call, and token count is captured.
    Summaries are generated for human readability.
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

    # Map agent keys to their prompt files
    AGENT_PROMPTS = {
        "analyzer": "statute-analyzer.md",
        "encoder": "encoder.md",
        "validator": "validator.md",
        "rac_reviewer": "rac-reviewer.md",
        "formula_reviewer": "formula-reviewer.md",
        "parameter_reviewer": "parameter-reviewer.md",
        "integration_reviewer": "integration-reviewer.md",
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-opus-4-5-20251101",
        plugin_path: Optional[Path] = None,
        experiment_db: Optional[ExperimentDB] = None,
    ):
        import os

        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY required")

        self.model = model
        self.plugin_path = plugin_path or self._find_plugin_path()
        self.experiment_db = experiment_db

    @staticmethod
    def _find_plugin_path() -> Path:
        """Find the cosilico plugin, checking marketplace and sibling locations."""
        candidates = [
            Path.home() / ".claude" / "plugins" / "marketplaces" / "cosilico",
            Path.home() / ".claude" / "plugins" / "cache" / "cosilico" / "cosilico",
            Path(__file__).parent.parent.parent.parent.parent / "cosilico-claude",
        ]
        for p in candidates:
            if (p / "agents").exists():
                return p
        return candidates[0]  # Default to marketplace path

    def _load_agent_prompt(self, agent_key: str) -> str:
        """Load the system prompt for an agent from the plugin."""
        prompt_file = self.AGENT_PROMPTS.get(agent_key)
        if not prompt_file:
            return ""
        prompt_path = self.plugin_path / "agents" / prompt_file
        if prompt_path.exists():
            return prompt_path.read_text()
        return ""

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
                agent_key="analyzer",
                prompt=f"Analyze {citation}. Report: subsection tree, encoding order, dependencies.",
                phase=Phase.ANALYSIS,
                model=self.model,  # Use configured model
            )
            run.agent_runs.append(analysis)

            # Phase 2: Encoding
            encode_prompt = f"""Encode {citation} into RAC format.

Output path: {output_path}
{f"Statute text: {statute_text[:5000]}" if statute_text else "Fetch statute text as needed."}

## CRITICAL RULES (violations = encoding failure):

1. **FILEPATH = CITATION** - File names MUST be subsection names:
   - ✓ `statute/26/1/j.rac` for § 1(j)
   - ✓ `statute/26/1/a.rac` for § 1(a)
   - ❌ `formulas.rac`, `parameters.rac`, `variables.rac` - WRONG

2. **One subsection per file** - Each .rac encodes exactly one statutory subsection

3. **Only statute values** - No indexed/derived/computed values

Write .rac files to the output path. Run tests after each file."""

            encoding = await self._run_agent(
                agent_key="encoder",
                prompt=encode_prompt,
                phase=Phase.ENCODING,
                model=self.model,  # Use configured model
            )
            run.agent_runs.append(encoding)

            # Check what files were created
            if output_path.exists():
                run.files_created = [str(f) for f in output_path.rglob("*.rac")]

            # Phase 3: Oracle validation (use actual ValidatorPipeline, not LLM agent)
            print(
                f"\n[{datetime.utcnow().strftime('%H:%M:%S')}] ORACLE: ValidatorPipeline (PE + TAXSIM)",
                flush=True,
            )
            oracle_start = time.time()

            # Find RAC files to validate
            rac_files = list(output_path.rglob("*.rac")) if output_path.exists() else []

            oracle_context = {
                "pe_match": None,
                "taxsim_match": None,
                "discrepancies": [],
            }

            if rac_files:
                # Use ValidatorPipeline for actual oracle validation
                pipeline = ValidatorPipeline(
                    rac_us_path=output_path.parent.parent
                    if "statute" in str(output_path)
                    else output_path,
                    rac_path=Path(__file__).parent.parent.parent.parent / "rac",
                    enable_oracles=True,
                    max_workers=2,
                )

                # Aggregate results across all RAC files
                pe_scores = []
                taxsim_scores = []
                all_issues = []

                for rac_file in rac_files:
                    print(f"  Validating: {rac_file.name}", flush=True)
                    try:
                        pe_result = pipeline._run_policyengine(rac_file)
                        if pe_result.score is not None:
                            pe_scores.append(pe_result.score)
                            print(f"    PE: {pe_result.score:.1%}", flush=True)
                        all_issues.extend(pe_result.issues)

                        taxsim_result = pipeline._run_taxsim(rac_file)
                        if taxsim_result.score is not None:
                            taxsim_scores.append(taxsim_result.score)
                            print(f"    TAXSIM: {taxsim_result.score:.1%}", flush=True)
                        all_issues.extend(taxsim_result.issues)
                    except Exception as e:
                        print(f"    Error: {e}", flush=True)
                        all_issues.append(str(e))

                # Average scores across files
                if pe_scores:
                    oracle_context["pe_match"] = (
                        sum(pe_scores) / len(pe_scores) * 100
                    )  # Convert to percentage
                if taxsim_scores:
                    oracle_context["taxsim_match"] = (
                        sum(taxsim_scores) / len(taxsim_scores) * 100
                    )
                oracle_context["discrepancies"] = [
                    {"description": issue} for issue in all_issues[:10]
                ]
            else:
                print("  No RAC files found to validate", flush=True)

            oracle_duration = time.time() - oracle_start
            print(
                f"  DONE: PE={oracle_context.get('pe_match', 'N/A')}%, TAXSIM={oracle_context.get('taxsim_match', 'N/A')}% ({oracle_duration:.1f}s)",
                flush=True,
            )

            run.oracle_pe_match = oracle_context.get("pe_match")
            run.oracle_taxsim_match = oracle_context.get("taxsim_match")
            run.discrepancies = oracle_context.get("discrepancies", [])

            # Phase 4: LLM Review (parallel)
            oracle_summary = self._format_oracle_summary(oracle_context)

            # Run reviewers sequentially (parallel has async issues with SDK)
            for reviewer_key, reviewer_type in [
                ("formula_reviewer", "formulas"),
                ("parameter_reviewer", "parameters"),
                ("integration_reviewer", "integration"),
            ]:
                review = await self._run_agent(
                    agent_key=reviewer_key,
                    prompt=f"Review {citation} {reviewer_type}. Oracle found: {oracle_summary}",
                    phase=Phase.REVIEW,
                    model=self.model,  # Use configured model
                )
                run.agent_runs.append(review)

            # Phase 5: Report (computed, not an agent)
            run.ended_at = datetime.utcnow()
            run.total_tokens = self._sum_tokens(run.agent_runs)
            run.total_cost_usd = (
                run.total_tokens.estimated_cost_usd if run.total_tokens else 0.0
            )

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
        agent_key: str,
        prompt: str,
        phase: Phase,
        model: str,
    ) -> AgentRun:
        """Run a single agent and capture everything."""

        from claude_agent_sdk import ClaudeAgentOptions, query

        agent_type = self.AGENTS.get(agent_key, agent_key)

        # Load the agent's system prompt from plugin
        system_prompt = self._load_agent_prompt(agent_key)
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n---\n\n# TASK\n\n{prompt}"
        else:
            full_prompt = prompt

        run = AgentRun(
            agent_type=agent_type,
            prompt=prompt,  # Store original prompt, not full
            phase=phase,
        )

        # ALWAYS print phase start with timestamp
        start_time = datetime.utcnow()
        print(
            f"\n[{start_time.strftime('%H:%M:%S')}] {phase.value.upper()}: {agent_type}",
            flush=True,
        )
        print(f"  Model: {model}", flush=True)

        try:
            options = ClaudeAgentOptions(
                model=model,
                allowed_tools=["Read", "Write", "Edit", "Grep", "Glob", "Bash", "Task"],
            )

            # Collect all messages - log tokens in REAL TIME
            total_input = 0
            total_output = 0
            total_cache_read = 0
            event_count = 0

            async for event in query(prompt=full_prompt, options=options):
                event_count += 1
                event_type = type(event).__name__

                msg = AgentMessage(
                    role=getattr(event, "type", event_type),
                    content="",
                )

                # Parse SDK content blocks properly
                if hasattr(event, "content"):
                    content = event.content
                    # content is usually a list of blocks
                    if isinstance(content, list):
                        text_parts = []
                        for block in content:
                            block_type = type(block).__name__
                            if block_type == "TextBlock" and hasattr(block, "text"):
                                text_parts.append(block.text)
                            elif block_type == "ToolUseBlock":
                                # Extract tool info
                                if hasattr(block, "name"):
                                    msg.tool_name = block.name
                                    print(f"  Tool: {block.name}", flush=True)
                                if hasattr(block, "input"):
                                    msg.tool_input = (
                                        block.input
                                        if isinstance(block.input, dict)
                                        else {"raw": str(block.input)}
                                    )
                                # Generate summary
                                msg.summary = _summarize_tool_call(
                                    msg.tool_name, msg.tool_input, None
                                )
                            elif block_type == "ToolResultBlock":
                                # Tool result content
                                if hasattr(block, "content"):
                                    result_text = str(block.content)[:5000]
                                    text_parts.append(
                                        f"[Tool Result: {len(result_text)} chars]"
                                    )
                                    msg.tool_output = result_text
                        msg.content = (
                            "\n".join(text_parts)
                            if text_parts
                            else str(content)[:10000]
                        )
                        # Generate thinking summary if we have text
                        if text_parts and not msg.summary:
                            msg.summary = _summarize_thinking(
                                "\n".join(text_parts)
                            ) or _summarize_assistant_message("\n".join(text_parts))
                    else:
                        msg.content = str(content)[:10000]
                elif hasattr(event, "result"):
                    msg.content = str(event.result)[:10000]
                    run.result = event.result
                    msg.summary = "Final result"

                # Fallback tool capture from event attributes
                if not msg.tool_name and hasattr(event, "tool_name"):
                    msg.tool_name = event.tool_name
                    print(f"  Tool: {event.tool_name}", flush=True)
                if not msg.tool_input and hasattr(event, "tool_input"):
                    msg.tool_input = event.tool_input
                if not msg.tool_output and hasattr(event, "tool_output"):
                    msg.tool_output = str(event.tool_output)[:10000]

                # Generate summary if not already set
                if not msg.summary and msg.tool_name:
                    msg.summary = _summarize_tool_call(
                        msg.tool_name, msg.tool_input, msg.tool_output
                    )

                # Capture tokens from ResultMessage (final event has real data)
                if event_type == "ResultMessage" and hasattr(event, "usage"):
                    usage = event.usage
                    # usage is a dict, not an object
                    in_tok = usage.get("input_tokens", 0)
                    out_tok = usage.get("output_tokens", 0)
                    cache_create = usage.get("cache_creation_input_tokens", 0)
                    cache_read = usage.get("cache_read_input_tokens", 0)

                    msg.tokens = TokenUsage(
                        input_tokens=in_tok + cache_create,
                        output_tokens=out_tok,
                        cache_read_tokens=cache_read,
                    )
                    total_input = in_tok + cache_create
                    total_output = out_tok
                    total_cache_read = cache_read

                    # Also capture total_cost_usd if available
                    if hasattr(event, "total_cost_usd"):
                        run.total_cost = event.total_cost_usd

                    # Print final token summary
                    print(
                        f"  Tokens: {in_tok + cache_create:,} in (+{cache_read:,} cache), {out_tok:,} out",
                        flush=True,
                    )
                    if hasattr(event, "total_cost_usd"):
                        print(f"  Cost: ${event.total_cost_usd:.4f}", flush=True)

                run.messages.append(msg)

            run.ended_at = datetime.utcnow()
            run.total_tokens = TokenUsage(
                input_tokens=total_input,
                output_tokens=total_output,
                cache_read_tokens=total_cache_read,
            )

            # Print phase summary
            duration = (run.ended_at - start_time).total_seconds()
            total_cost = run.total_tokens.estimated_cost_usd
            print(
                f"  DONE: {total_input:,} in + {total_output:,} out = ${total_cost:.4f} ({duration:.1f}s)",
                flush=True,
            )

        except Exception as e:
            run.error = str(e)
            run.ended_at = datetime.utcnow()
            print(f"  ERROR: {e}", flush=True)
            # Still print whatever tokens we captured
            if total_input or total_output:
                print(
                    f"  Partial tokens: {total_input:,} in + {total_output:,} out",
                    flush=True,
                )

        return run

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
        self.experiment_db.start_session(
            model=self.model, cwd=str(Path.cwd()), session_id=run.session_id
        )

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
                },
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
                        "summary": msg.summary,  # Human-readable summary
                        "tool_input": msg.tool_input,
                        "tool_output": msg.tool_output[:1000]
                        if msg.tool_output
                        else None,
                        "tokens": {
                            "input": msg.tokens.input_tokens if msg.tokens else 0,
                            "output": msg.tokens.output_tokens if msg.tokens else 0,
                        }
                        if msg.tokens
                        else None,
                    },
                )

            # Calculate phase cost
            phase_cost = (
                agent_run.total_tokens.estimated_cost_usd
                if agent_run.total_tokens
                else 0
            )

            # Generate phase summary
            tool_counts = {}
            for msg in agent_run.messages:
                if msg.tool_name:
                    tool_counts[msg.tool_name] = tool_counts.get(msg.tool_name, 0) + 1
            tools_summary = ", ".join(
                f"{t}×{c}" for t, c in sorted(tool_counts.items(), key=lambda x: -x[1])
            )

            phase_summary = (
                f"{agent_run.phase.value.upper()}: {len(agent_run.messages)} events"
            )
            if tools_summary:
                phase_summary += f" ({tools_summary})"
            if phase_cost > 0:
                phase_summary += f" - ${phase_cost:.2f}"

            # Log agent end
            self.experiment_db.log_event(
                session_id=run.session_id,
                event_type="agent_end",
                content=agent_run.result or "",
                metadata={
                    "agent_type": agent_run.agent_type,
                    "phase": agent_run.phase.value,
                    "summary": phase_summary,
                    "error": agent_run.error,
                    "total_tokens": {
                        "input": agent_run.total_tokens.input_tokens,
                        "output": agent_run.total_tokens.output_tokens,
                    }
                    if agent_run.total_tokens
                    else None,
                    "cost_usd": phase_cost,
                    "tools_used": tool_counts,
                },
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
            f"Duration: {(run.ended_at - run.started_at).total_seconds():.1f}s"
            if run.ended_at
            else "In progress",
            "",
            "## Oracle Match Rates",
            f"- PolicyEngine: {run.oracle_pe_match}%"
            if run.oracle_pe_match
            else "- PolicyEngine: N/A",
            f"- TAXSIM: {run.oracle_taxsim_match}%"
            if run.oracle_taxsim_match
            else "- TAXSIM: N/A",
            "",
            "## Files Created",
        ]
        for f in run.files_created:
            lines.append(f"- {f}")

        lines.extend(
            [
                "",
                "## Agent Runs",
            ]
        )
        for agent_run in run.agent_runs:
            cost = agent_run.total_cost
            tokens = agent_run.total_tokens
            if cost is not None:
                lines.append(
                    f"- {agent_run.phase.value}: {agent_run.agent_type} (${cost:.4f})"
                )
            elif tokens:
                lines.append(
                    f"- {agent_run.phase.value}: {agent_run.agent_type} "
                    f"({tokens.total_tokens} tokens)"
                )
            else:
                lines.append(f"- {agent_run.phase.value}: {agent_run.agent_type}")

        if run.total_tokens:
            lines.extend(
                [
                    "",
                    "## Totals",
                    f"- Input tokens: {run.total_tokens.input_tokens:,}",
                    f"- Output tokens: {run.total_tokens.output_tokens:,}",
                    f"- Total cost: ${run.total_cost_usd:.4f}",
                ]
            )

        return "\n".join(lines)
