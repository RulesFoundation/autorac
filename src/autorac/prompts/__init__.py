"""
Embedded agent prompts for AutoRAC.

All prompts are self-contained -- no dependency on external plugins.
Each module exports a get_*_prompt() function that returns the full
prompt string with citation/path interpolated.
"""

from .encoder import ENCODER_PROMPT, get_encoder_prompt
from .reviewers import (
    FORMULA_REVIEWER_PROMPT,
    INTEGRATION_REVIEWER_PROMPT,
    PARAMETER_REVIEWER_PROMPT,
    RAC_REVIEWER_PROMPT,
    get_formula_reviewer_prompt,
    get_integration_reviewer_prompt,
    get_parameter_reviewer_prompt,
    get_rac_reviewer_prompt,
)
from .validator import VALIDATOR_PROMPT, get_validator_prompt

__all__ = [
    "ENCODER_PROMPT",
    "get_encoder_prompt",
    "VALIDATOR_PROMPT",
    "get_validator_prompt",
    "RAC_REVIEWER_PROMPT",
    "get_rac_reviewer_prompt",
    "FORMULA_REVIEWER_PROMPT",
    "get_formula_reviewer_prompt",
    "PARAMETER_REVIEWER_PROMPT",
    "get_parameter_reviewer_prompt",
    "INTEGRATION_REVIEWER_PROMPT",
    "get_integration_reviewer_prompt",
]
