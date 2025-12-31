"""
Encoder Harness - wraps encoder agent with prediction and logging.

The harness orchestrates:
1. Agent predicts scores before encoding
2. Agent encodes the statute
3. Agent suggests framework improvements
4. Validators run in parallel
5. Everything is logged for calibration
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

from anthropic import Anthropic

from .experiment_db import (
    ExperimentDB,
    EncodingRun,
    PredictedScores,
    ActualScores,
    AgentSuggestion,
    create_run,
)
from .validator_pipeline import ValidatorPipeline, PipelineResult


# Initialize Anthropic client (uses ANTHROPIC_API_KEY env var)
_client: Optional[Anthropic] = None


def get_anthropic_client() -> Anthropic:
    """Get or create Anthropic client."""
    global _client
    if _client is None:
        _client = Anthropic()
    return _client


# DSL specification for RAC format
RAC_DSL_SPEC = """
# RAC DSL Format

Each .rac file defines variables using this structure:

```yaml
text: \"\"\"
Original statute text quoted verbatim
\"\"\"

variable variable_name:
  imports:
    - path/to/dependency#variable_name
    - path/to/other#other_variable as alias
  entity: TaxUnit | Person | Household
  period: Year | Month
  dtype: Money | Rate | Boolean | Integer
  unit: USD  # optional
  label: "Human readable name"
  description: "Detailed description"
  syntax: python
  formula: |
    # Python-like formula
    if condition:
        return value
    return other_value
  default: 0
  tests:
    - name: "Test case name"
      period: 2024-01
      inputs:
        input_var: value
      expect: expected_output
```

## Key Rules:

1. **Imports**: Use `path#variable` syntax, e.g., `26/32/c/2/A#earned_income`
2. **Entity hierarchy**: Person < TaxUnit < Household
3. **dtypes**: Money, Rate, Boolean, Integer
4. **No hardcoded values**: Only -1, 0, 1, 2, 3 allowed as literals. All other values must be parameters.
5. **Tests**: Include at least 3-5 test cases covering edge cases
6. **Formulas**: Python-style with `return` statements
7. **Filepath = Citation**: `statute/26/32/a/1.rac` encodes 26 USC 32(a)(1)
"""


PREDICTION_SYSTEM_PROMPT = """You are an expert at predicting code quality scores for tax/benefit statute encodings.

Given a legal citation and statute text, predict how well an encoder would perform on various quality dimensions.

Score each dimension from 1-10 where:
- 10: Perfect implementation
- 7-9: Good with minor issues
- 4-6: Acceptable but needs improvement
- 1-3: Significant problems

Output ONLY valid JSON with this structure:
{
  "rac_reviewer": <float 1-10>,
  "formula_reviewer": <float 1-10>,
  "parameter_reviewer": <float 1-10>,
  "integration_reviewer": <float 1-10>,
  "ci_pass": <boolean>,
  "policyengine_match": <float 0-1>,
  "taxsim_match": <float 0-1>,
  "confidence": <float 0-1>,
  "reasoning": "<brief explanation>"
}
"""


ENCODER_SYSTEM_PROMPT = """You are an expert encoder for the Cosilico RAC DSL (Rules as Code).

Your task is to encode tax and benefit statutes into executable .rac files.

""" + RAC_DSL_SPEC + """

## Output Format

Output ONLY the .rac file content. No markdown code blocks, no explanations.
Start directly with `text:` and the quoted statute text.
"""


SUGGESTIONS_SYSTEM_PROMPT = """You are an expert at improving tax/benefit encoding frameworks.

Given validation results from encoding attempts, suggest improvements to:
1. Documentation - clearer DSL documentation
2. Agent prompts - better encoding instructions
3. Validators - more accurate validation checks
4. DSL enhancements - language features to add

Output ONLY valid JSON array with this structure:
[
  {
    "category": "documentation" | "agent_prompt" | "validator" | "dsl",
    "description": "<what to improve>",
    "predicted_impact": "high" | "medium" | "low",
    "specific_change": "<exact change to make, or null>"
  }
]
"""


@dataclass
class EncoderConfig:
    """Configuration for the encoder harness."""
    rac_us_path: Path
    rac_path: Path
    db_path: Path = Path("experiments.db")
    enable_oracles: bool = True
    max_iterations: int = 3
    score_threshold: float = 7.0  # Minimum score to accept


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
        Ask Claude to predict scores before encoding.

        Calls the Claude API with a prediction prompt and parses
        the JSON response to extract predicted scores.
        """
        client = get_anthropic_client()

        user_prompt = f"""Predict quality scores for encoding the following statute:

Citation: {citation}

Statute Text:
{statute_text}

Based on the complexity of this statute and typical encoding challenges,
predict the scores that reviewers would assign to an encoding attempt.
"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1024,
                system=PREDICTION_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Extract text content from response
            response_text = response.content[0].text

            # Parse JSON from response
            # Try to extract JSON if wrapped in other text
            json_match = re.search(r'\{[^{}]*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = response_text

            data = json.loads(json_str)

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
            # Log error and return conservative defaults
            print(f"Warning: Failed to get predictions from Claude: {e}")
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
        Invoke Claude to encode the statute to RAC format.

        Calls the Claude API to generate a .rac file, then writes
        it to the output path and returns the content.
        """
        client = get_anthropic_client()

        # Derive variable name from citation
        # "26 USC 32(a)(1)" -> "eitc_credit" or similar
        var_name = citation.replace("USC", "").replace("(", "_").replace(")", "").replace(" ", "_").lower()
        var_name = re.sub(r'_+', '_', var_name).strip('_')

        user_prompt = f"""Encode the following statute into RAC DSL format:

Citation: {citation}
Output File: {output_path}

Statute Text:
{statute_text}

Requirements:
1. Include the full statute text in the `text:` block
2. Define appropriate variable(s) for the computation
3. Use proper imports for any dependencies (e.g., income, filing_status)
4. Include at least 3 test cases covering normal and edge cases
5. Follow the exact RAC DSL syntax - no markdown formatting
"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4096,
                system=ENCODER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Extract text content from response
            rac_content = response.content[0].text

            # Clean up any markdown code blocks if present
            rac_content = re.sub(r'^```\w*\n', '', rac_content)
            rac_content = re.sub(r'\n```$', '', rac_content)
            rac_content = rac_content.strip()

            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Write the RAC file
            output_path.write_text(rac_content)

            return rac_content

        except Exception as e:
            # Log error and return a minimal placeholder
            print(f"Warning: Failed to encode with Claude: {e}")

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
  syntax: python
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
        Ask Claude for framework improvement suggestions.

        Based on validation errors, Claude suggests:
        - Documentation improvements
        - Agent prompt changes
        - Validator fixes
        - DSL enhancements
        """
        # Only get suggestions if there were failures
        failures = [
            (name, result)
            for name, result in validation_result.results.items()
            if not result.passed
        ]

        if not failures:
            return []

        client = get_anthropic_client()

        # Build validation summary
        validation_summary = []
        for name, result in validation_result.results.items():
            status = "PASSED" if result.passed else "FAILED"
            score_str = f" (score: {result.score})" if result.score is not None else ""
            error_str = f" - {result.error}" if result.error else ""
            issues_str = f" Issues: {result.issues}" if result.issues else ""
            validation_summary.append(f"  {name}: {status}{score_str}{error_str}{issues_str}")

        user_prompt = f"""Analyze the following encoding attempt and suggest framework improvements:

Citation: {citation}

RAC Content:
```
{rac_content[:2000]}{'...' if len(rac_content) > 2000 else ''}
```

Validation Results:
{chr(10).join(validation_summary)}

Based on these results, suggest improvements to:
1. Documentation - How could the DSL docs be clearer?
2. Agent prompts - How could encoding instructions be better?
3. Validators - Are any validation checks incorrect or missing?
4. DSL - What language features would help?

Focus on actionable, specific suggestions.
"""

        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2048,
                system=SUGGESTIONS_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            # Extract text content from response
            response_text = response.content[0].text

            # Try to parse JSON array from response
            # Handle potential wrapper text
            json_match = re.search(r'\[[\s\S]*\]', response_text)
            if json_match:
                json_str = json_match.group()
            else:
                json_str = response_text

            data = json.loads(json_str)

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
            # Log error and return basic suggestions based on failures
            print(f"Warning: Failed to get suggestions from Claude: {e}")

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
