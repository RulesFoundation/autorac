"""Tests for parameter value validation - values must appear in statute text."""

import pytest
from autorac.checks import check_param_values_in_text


class TestParamValuesInText:
    """Parameter values must appear in the statute text field."""

    def test_value_found_in_text(self):
        """Parameter value that appears in text passes."""
        rac_content = '''
text: """
In the case of a taxpayer who has attained age 65...
"""

parameter elderly_age_threshold:
  values:
    2024-01-01: 65
'''
        result = check_param_values_in_text(rac_content)
        assert result.passed
        assert len(result.missing_values) == 0

    def test_value_not_in_text_fails(self):
        """Parameter value not in text is flagged."""
        rac_content = '''
text: """
Some statute text with no numbers...
"""

parameter some_threshold:
  values:
    2024-01-01: 500
'''
        result = check_param_values_in_text(rac_content)
        assert not result.passed
        assert 500 in result.missing_values

    def test_percentage_normalized(self):
        """7.5% in text matches 0.075 parameter."""
        rac_content = '''
text: """
...exceeds 7.5 percent of adjusted gross income...
"""

parameter medical_expense_threshold_rate:
  values:
    2024-01-01: 0.075
'''
        result = check_param_values_in_text(rac_content)
        assert result.passed

    def test_percentage_as_decimal_in_text(self):
        """0.075 in text also matches."""
        rac_content = '''
text: """
...the applicable rate is 0.075...
"""

parameter some_rate:
  values:
    2024-01-01: 0.075
'''
        result = check_param_values_in_text(rac_content)
        assert result.passed

    def test_dollar_amount_with_comma(self):
        """$1,000 in text matches 1000 parameter."""
        rac_content = '''
text: """
...shall not exceed $1,000...
"""

parameter max_credit:
  values:
    2024-01-01: 1000
'''
        result = check_param_values_in_text(rac_content)
        assert result.passed

    def test_dollar_amount_with_dollar_sign(self):
        """$500 in text matches 500 parameter."""
        rac_content = '''
text: """
...a credit of $500...
"""

parameter credit_amount:
  values:
    2024-01-01: 500
'''
        result = check_param_values_in_text(rac_content)
        assert result.passed

    def test_multiple_values_all_found(self):
        """Multiple parameter values all found in text."""
        rac_content = '''
text: """
...age 65 or older...income below $50,000...rate of 15 percent...
"""

parameter age_threshold:
  values:
    2024-01-01: 65

parameter income_limit:
  values:
    2024-01-01: 50000

parameter rate:
  values:
    2024-01-01: 0.15
'''
        result = check_param_values_in_text(rac_content)
        assert result.passed

    def test_multiple_values_one_missing(self):
        """One missing value fails the check."""
        rac_content = '''
text: """
...age 65 or older...
"""

parameter age_threshold:
  values:
    2024-01-01: 65

parameter mystery_value:
  values:
    2024-01-01: 999
'''
        result = check_param_values_in_text(rac_content)
        assert not result.passed
        assert 999 in result.missing_values
        assert 65 not in result.missing_values

    def test_effective_dates_ignored(self):
        """Effective dates (keys) are not checked, only values."""
        rac_content = '''
text: """
...the amount is 100...
"""

parameter amount:
  values:
    2024-01-01: 100
    2023-01-01: 90
'''
        result = check_param_values_in_text(rac_content)
        # 2024 and 2023 are dates, not checked
        # 100 is in text, 90 is not - should fail
        assert not result.passed
        assert 90 in result.missing_values

    def test_zero_and_one_allowed(self):
        """0 and 1 are common defaults, always allowed."""
        rac_content = '''
text: """
Some text with no numbers.
"""

parameter flag:
  values:
    2024-01-01: 0

parameter multiplier:
  values:
    2024-01-01: 1
'''
        result = check_param_values_in_text(rac_content)
        assert result.passed

    def test_no_text_field_fails(self):
        """RAC without text field fails validation."""
        rac_content = '''
parameter some_value:
  values:
    2024-01-01: 500
'''
        result = check_param_values_in_text(rac_content)
        assert not result.passed
        assert "no text field" in result.error.lower()

    def test_no_parameters_passes(self):
        """RAC with no parameters passes (nothing to check)."""
        rac_content = '''
text: """
Some statute text.
"""

variable something:
  formula: return 0
'''
        result = check_param_values_in_text(rac_content)
        assert result.passed

    def test_large_numbers_with_commas(self):
        """Large numbers like 100,000 match 100000."""
        rac_content = '''
text: """
...income exceeding $100,000...
"""

parameter income_threshold:
  values:
    2024-01-01: 100000
'''
        result = check_param_values_in_text(rac_content)
        assert result.passed

    def test_decimal_percentage_variations(self):
        """Various percentage formats: 10%, 10 percent, .10"""
        rac_content = '''
text: """
...at a rate of 10 percent...
"""

parameter rate:
  values:
    2024-01-01: 0.10
'''
        result = check_param_values_in_text(rac_content)
        assert result.passed

    def test_reports_parameter_name_with_missing_value(self):
        """Result includes which parameter has the missing value."""
        rac_content = '''
text: """
Just some text.
"""

parameter mystery_param:
  values:
    2024-01-01: 12345
'''
        result = check_param_values_in_text(rac_content)
        assert not result.passed
        assert "mystery_param" in result.details
