"""
Encoder Harness - wraps encoder agent with prediction and logging.

The harness orchestrates:
1. Agent predicts scores before encoding
2. Agent encodes the statute
3. Agent suggests framework improvements
4. Validators run in parallel
5. Everything is logged for calibration

Uses Claude Code CLI (subprocess) for agent calls - cheaper than direct API.
"""

import subprocess
import json
import time
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime

from .experiment_db import (
    ExperimentDB,
    EncodingRun,
    PredictedScores,
    FinalScores,
    AgentSuggestion,
    create_run,
)
from .validator_pipeline import ValidatorPipeline, PipelineResult


def run_claude_code(
    prompt: str,
    agent: Optional[str] = None,
    model: str = "sonnet",
    timeout: int = 300,
    cwd: Optional[Path] = None,
    plugin_dir: Optional[Path] = None,
) -> tuple[str, int]:
    """
    Run Claude Code CLI as subprocess.

    Args:
        prompt: The prompt to send
        agent: Optional agent type (e.g., "cosilico:RAC Encoder")
        model: Model to use (sonnet, opus, haiku)
        timeout: Timeout in seconds
        cwd: Working directory
        plugin_dir: Directory containing Claude Code plugins

    Returns:
        Tuple of (output text, return code)
    """
    cmd = ["claude", "--print"]

    if model:
        cmd.extend(["--model", model])

    if plugin_dir and plugin_dir.exists():
        cmd.extend(["--plugin-dir", str(plugin_dir)])

    if agent:
        cmd.extend(["--agent", agent])

    # Add the prompt
    cmd.extend(["-p", prompt])

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        return result.stdout + result.stderr, result.returncode
    except subprocess.TimeoutExpired:
        return f"Timeout after {timeout}s", 1
    except FileNotFoundError:
        return "Claude CLI not found - install with: npm install -g @anthropic-ai/claude-code", 1
    except Exception as e:
        return f"Error running Claude CLI: {e}", 1


@dataclass
class EncoderConfig:
    """Configuration for the encoder harness."""
    rac_us_path: Path
    rac_path: Path
    db_path: Path = Path("experiments.db")
    cosilico_plugin_path: Optional[Path] = None  # Path to cosilico-claude plugin
    enable_oracles: bool = True
    max_iterations: int = 3
    score_threshold: float = 7.0  # Minimum score to accept

    def __post_init__(self):
        # Auto-detect cosilico plugin if not specified
        if self.cosilico_plugin_path is None:
            # Try common locations
            candidates = [
                self.rac_us_path.parent / "cosilico-claude",
                Path.home() / "CosilicoAI" / "cosilico-claude",
                Path.home() / ".claude" / "plugins" / "cosilico-claude",
            ]
            for candidate in candidates:
                if candidate.exists() and (candidate / "plugin.json").exists():
                    self.cosilico_plugin_path = candidate
                    break


class EncoderHarness:
    """
    Wraps encoder agent with prediction, validation, and logging.

    The harness implements the encode-predict-validate-learn loop:
    1. Before encoding, agent predicts expected scores
    2. Agent encodes statute to RAC
    3. Validators run in parallel
    4. Results logged for calibration analysis
    5. Agent suggests improvements based on errors
    """

    def __init__(self, config: EncoderConfig):
        self.config = config
        self.db = ExperimentDB(config.db_path)
        self.pipeline = ValidatorPipeline(
            rac_us_path=config.rac_us_path,
            rac_path=config.rac_path,
            enable_oracles=config.enable_oracles,
        )

    def encode_with_feedback(
        self,
        citation: str,
        statute_text: str,
        output_path: Path,
        agent_type: str = "autorac:encoder",
        agent_model: str = "claude-opus-4-5-20251101",
    ) -> tuple[EncodingRun, PipelineResult]:
        """
        Full encode-validate-log cycle.

        Returns the encoding run and validation results.
        """
        start = time.time()

        # Step 1: Get predictions from agent
        predicted = self._get_predictions(citation, statute_text)

        # Step 2: Encode
        rac_content = self._encode(citation, statute_text, output_path)

        encoding_duration = int((time.time() - start) * 1000)

        # Step 3: Validate
        validation_start = time.time()
        validation_result = self.pipeline.validate(output_path)
        validation_duration = int((time.time() - validation_start) * 1000)

        # Step 4: Get suggestions
        suggestions = self._get_suggestions(
            citation, rac_content, validation_result
        )

        # Step 5: Log everything
        run = create_run(
            file_path=str(output_path),
            citation=citation,
            agent_type=agent_type,
            agent_model=agent_model,
            rac_content=rac_content,
            statute_text=statute_text,
        )
        run.predicted = predicted
        run.actual = validation_result.to_actual_scores()
        run.suggestions = suggestions
        run.encoding_duration_ms = encoding_duration
        run.validation_duration_ms = validation_duration

        self.db.log_run(run)

        return run, validation_result

    def iterate_until_pass(
        self,
        citation: str,
        statute_text: str,
        output_path: Path,
        agent_type: str = "autorac:encoder",
        agent_model: str = "claude-opus-4-5-20251101",
    ) -> list[tuple[EncodingRun, PipelineResult]]:
        """
        Iteratively encode until all validators pass or max iterations.

        Returns list of (run, result) for each iteration.
        """
        iterations = []
        parent_run_id = None

        for i in range(self.config.max_iterations):
            run, result = self.encode_with_feedback(
                citation=citation,
                statute_text=statute_text,
                output_path=output_path,
                agent_type=agent_type,
                agent_model=agent_model,
            )

            if parent_run_id:
                run.parent_run_id = parent_run_id
                run.iteration = i + 1

            iterations.append((run, result))

            if result.all_passed:
                break

            # Prepare feedback for next iteration
            parent_run_id = run.id

        return iterations

    def _get_predictions(
        self, citation: str, statute_text: str
    ) -> PredictedScores:
        """
        Ask Claude Code to predict scores before encoding.

        Uses Claude Code CLI subprocess for cheaper execution.
        """
        prompt = f"""Predict quality scores for encoding the following statute into RAC DSL.

Citation: {citation}

Statute Text:
{statute_text[:2000]}{'...' if len(statute_text) > 2000 else ''}

Score each dimension from 1-10. Output ONLY valid JSON:
{{
  "rac_reviewer": <float 1-10>,
  "formula_reviewer": <float 1-10>,
  "parameter_reviewer": <float 1-10>,
  "integration_reviewer": <float 1-10>,
  "ci_pass": <boolean>,
  "policyengine_match": <float 0-1>,
  "taxsim_match": <float 0-1>,
  "confidence": <float 0-1>,
  "reasoning": "<brief explanation>"
}}
"""

        try:
            output, returncode = run_claude_code(
                prompt,
                model="opus",
                timeout=60,
                cwd=self.config.rac_us_path,
            )

            # Parse JSON from output
            json_match = re.search(r'\{[^{}]*\}', output, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON found in output")

            return PredictedScores(
                rac_reviewer=float(data.get("rac_reviewer", 7.0)),
                formula_reviewer=float(data.get("formula_reviewer", 7.0)),
                parameter_reviewer=float(data.get("parameter_reviewer", 7.0)),
                integration_reviewer=float(data.get("integration_reviewer", 7.0)),
                ci_pass=bool(data.get("ci_pass", True)),
                policyengine_match=float(data.get("policyengine_match", 0.85)) if data.get("policyengine_match") is not None else None,
                taxsim_match=float(data.get("taxsim_match", 0.85)) if data.get("taxsim_match") is not None else None,
                confidence=float(data.get("confidence", 0.5)),
            )

        except Exception as e:
            print(f"Warning: Failed to get predictions: {e}")
            return PredictedScores(
                rac_reviewer=6.0,
                formula_reviewer=6.0,
                parameter_reviewer=6.0,
                integration_reviewer=6.0,
                ci_pass=False,
                policyengine_match=0.7,
                taxsim_match=0.7,
                confidence=0.3,
            )

    def _encode(
        self, citation: str, statute_text: str, output_path: Path
    ) -> str:
        """
        Invoke Claude Code to encode the statute to RAC format.

        Uses the cosilico:RAC Encoder agent from the plugin.
        """
        # Derive variable name from citation for fallback
        var_name = citation.replace("USC", "").replace("(", "_").replace(")", "").replace(" ", "_").lower()
        var_name = re.sub(r'_+', '_', var_name).strip('_')

        # Use the cosilico plugin's encoder agent
        prompt = f"""Encode {citation} into RAC format.

Write the output to: {output_path}

Statute Text:
{statute_text}

Use the Write tool to create the .rac file at the specified path.
"""

        try:
            output, returncode = run_claude_code(
                prompt,
                agent="cosilico:RAC Encoder",
                model="opus",
                timeout=300,
                cwd=self.config.rac_us_path,
                plugin_dir=self.config.cosilico_plugin_path,
            )

            # Check if file was created
            if output_path.exists():
                rac_content = output_path.read_text()
            else:
                # Try to extract RAC content from output
                rac_content = output

            # Clean up any markdown code blocks if present
            rac_content = re.sub(r'^```\w*\n', '', rac_content)
            rac_content = re.sub(r'\n```$', '', rac_content)
            rac_content = rac_content.strip()

            # Ensure output directory exists and write
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(rac_content)

            return rac_content

        except Exception as e:
            print(f"Warning: Failed to encode: {e}")

            # Return a minimal valid RAC structure as fallback
            fallback = f'''text: """
{statute_text}
"""

variable {var_name}:
  entity: TaxUnit
  period: Year
  dtype: Money
  label: "{citation}"
  description: "Auto-generated placeholder - encoding failed"
  formula: |
    # TODO: Implement formula
    return 0
  default: 0
  tests:
    - name: "Placeholder test"
      period: 2024-01
      inputs: {{}}
      expect: 0
'''
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(fallback)
            return fallback

    def _get_suggestions(
        self,
        citation: str,
        rac_content: str,
        validation_result: PipelineResult,
    ) -> list[AgentSuggestion]:
        """
        Ask Claude Code for framework improvement suggestions.

        Based on validation errors, suggests improvements.
        """
        # Only get suggestions if there were failures
        failures = [
            (name, result)
            for name, result in validation_result.results.items()
            if not result.passed
        ]

        if not failures:
            return []

        # Build validation summary
        validation_summary = []
        for name, result in validation_result.results.items():
            status = "PASSED" if result.passed else "FAILED"
            score_str = f" (score: {result.score})" if result.score is not None else ""
            error_str = f" - {result.error}" if result.error else ""
            issues_str = f" Issues: {result.issues}" if result.issues else ""
            validation_summary.append(f"  {name}: {status}{score_str}{error_str}{issues_str}")

        prompt = f"""Analyze encoding attempt for {citation} and suggest framework improvements.

Validation Results:
{chr(10).join(validation_summary)}

Output ONLY valid JSON array:
[
  {{
    "category": "documentation" | "agent_prompt" | "validator" | "dsl",
    "description": "<what to improve>",
    "predicted_impact": "high" | "medium" | "low",
    "specific_change": "<exact change, or null>"
  }}
]
"""

        try:
            output, returncode = run_claude_code(
                prompt,
                model="opus",
                timeout=60,
                cwd=self.config.rac_us_path,
            )

            # Parse JSON array from output
            json_match = re.search(r'\[[\s\S]*\]', output)
            if json_match:
                data = json.loads(json_match.group())
            else:
                raise ValueError("No JSON array found in output")

            suggestions = []
            for item in data:
                suggestions.append(AgentSuggestion(
                    category=item.get("category", "documentation"),
                    description=item.get("description", ""),
                    predicted_impact=item.get("predicted_impact", "medium"),
                    specific_change=item.get("specific_change"),
                ))

            return suggestions

        except Exception as e:
            print(f"Warning: Failed to get suggestions: {e}")

            # Return basic suggestions based on failures
            suggestions = []
            for name, result in failures:
                suggestions.append(AgentSuggestion(
                    category="validator",
                    description=f"{name} failed: {result.error or 'unknown'}",
                    predicted_impact="medium",
                    specific_change=None,
                ))

            return suggestions


def run_encoding_experiment(
    citation: str,
    statute_text: str,
    output_dir: Path,
    config: Optional[EncoderConfig] = None,
) -> list[tuple[EncodingRun, PipelineResult]]:
    """
    Convenience function to run a full encoding experiment.

    Args:
        citation: Legal citation (e.g., "26 USC 1(h)(1)(E)")
        statute_text: Raw statute text to encode
        output_dir: Directory for output RAC file
        config: Optional encoder configuration

    Returns:
        List of (run, result) tuples for each iteration
    """
    if config is None:
        # Auto-detect paths
        rac_us = output_dir
        while rac_us.name != "rac-us" and rac_us.parent != rac_us:
            rac_us = rac_us.parent

        config = EncoderConfig(
            rac_us_path=rac_us,
            rac_path=rac_us.parent / "rac",
        )

    harness = EncoderHarness(config)

    # Derive output path from citation
    # "26 USC 1(h)(1)(E)" -> statute/26/1/h/1/E.rac
    parts = citation.replace("USC", "").replace("(", "/").replace(")", "").split()
    title = parts[0]
    rest = "".join(parts[1:])
    output_path = output_dir / f"statute/{title}/{rest}.rac"

    return harness.iterate_until_pass(
        citation=citation,
        statute_text=statute_text,
        output_path=output_path,
    )
