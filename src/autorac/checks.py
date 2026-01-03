"""RAC validation checks."""

import re
from dataclasses import dataclass, field


@dataclass
class ParamCheckResult:
    """Result of parameter value check."""
    passed: bool
    missing_values: list = field(default_factory=list)
    details: str = ""
    error: str = ""


def check_param_values_in_text(rac_content: str) -> ParamCheckResult:
    """Check that all parameter values appear in the statute text.

    Every numeric value defined in parameters must appear somewhere in the
    text: field. This catches hallucinated values or copy-paste errors.

    Handles:
    - Percentage conversion: 0.075 matches "7.5%", "7.5 percent"
    - Currency/commas: 1000 matches "$1,000"
    - Always allowed: 0, 1 (common defaults)

    Does NOT check:
    - Effective dates (the keys like 2024-01-01)
    """
    # Extract text field
    text_match = re.search(r'text:\s*"""(.*?)"""', rac_content, re.DOTALL)
    if not text_match:
        # Try single-line text
        text_match = re.search(r'text:\s*"([^"]*)"', rac_content)

    if not text_match:
        return ParamCheckResult(
            passed=False,
            error="No text field found in RAC content"
        )

    text = text_match.group(1)

    # Extract all parameter values
    # Pattern: parameter name:\n  values:\n    date: value
    param_pattern = r'parameter\s+(\w+):\s*\n\s*values:\s*\n((?:\s+[\d-]+:\s*[\d.]+\n?)+)'

    missing = []
    details_parts = []

    for match in re.finditer(param_pattern, rac_content):
        param_name = match.group(1)
        values_block = match.group(2)

        # Extract individual values (skip the date keys)
        value_pattern = r'[\d-]+:\s*([\d.]+)'
        for val_match in re.finditer(value_pattern, values_block):
            value_str = val_match.group(1)
            value = float(value_str)

            # Always allow 0 and 1
            if value in (0, 0.0, 1, 1.0):
                continue

            if not _value_in_text(value, text):
                missing.append(value)
                details_parts.append(f"{param_name}: {value}")

    if missing:
        return ParamCheckResult(
            passed=False,
            missing_values=missing,
            details=f"Values not found in text: {', '.join(details_parts)}"
        )

    return ParamCheckResult(passed=True)


def _value_in_text(value: float, text: str) -> bool:
    """Check if a numeric value appears in text in any common format.

    Handles:
    - Exact match: 65
    - With commas: 100,000
    - With dollar sign: $500
    - As percentage: 0.075 -> 7.5%, 7.5 percent
    - Decimal variations: 0.10 -> 10%, 10 percent
    """
    # Normalize text: lowercase, remove extra whitespace
    text_lower = text.lower()

    # Check exact value (as int if whole number)
    if value == int(value):
        int_val = int(value)
        # Exact match with word boundaries
        if re.search(rf'\b{int_val}\b', text):
            return True
        # With commas (e.g., 100,000)
        formatted = f"{int_val:,}"
        if formatted in text:
            return True
    else:
        # Decimal - check exact
        if str(value) in text:
            return True

    # Check as percentage (for values < 1 that look like rates)
    if 0 < value < 1:
        # Convert to percentage: 0.075 -> 7.5
        pct = value * 100
        # Check various percentage formats
        pct_patterns = [
            rf'{pct}%',
            rf'{pct} percent',
            rf'{pct}percent',
            rf'{int(pct)}%' if pct == int(pct) else None,
            rf'{int(pct)} percent' if pct == int(pct) else None,
        ]
        for pattern in pct_patterns:
            if pattern and pattern in text_lower:
                return True

        # Also check if the raw decimal is there
        if str(pct) in text:
            return True

    # Check with dollar sign
    if value == int(value):
        int_val = int(value)
        if f"${int_val}" in text or f"${int_val:,}" in text:
            return True

    return False
