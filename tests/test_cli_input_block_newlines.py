"""Companion input repair must preserve neighboring YAML blocks."""

import pytest
import yaml

from axiom_encode.cli import (
    _expand_empty_inline_yaml_input_blocks,
    _insert_input_default_in_test_cases,
)


@pytest.mark.parametrize("newline", ["\n", "\r\n", ""])
@pytest.mark.parametrize("suffix", ["", " # retained comment"])
@pytest.mark.parametrize("anchor", ["", " &shared"])
def test_empty_input_expansion_preserves_line_ending(newline, suffix, anchor):
    line = f"  input:{anchor} {{}}{suffix}{newline}"
    assert _expand_empty_inline_yaml_input_blocks([line]) == [
        f"  input:{anchor}{suffix}{newline}"
    ]


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("suffix", ["", " # retained comment"])
def test_input_default_repair_keeps_table_rows_and_outputs(newline, suffix):
    content = newline.join(
        [
            "- name: shared_pairs",
            f"  input: {{}}{suffix}",
            "  tables:",
            "    CandidateChildPair:",
            "    - de:statutes/bgb/1591#input.child_identifier: child-1",
            "    - de:statutes/bgb/1591#input.child_identifier: child-2",
            "  output:",
            "    de:statutes/bgb/1591#motherhood_established_by_recorded_birth:",
            "    - holds",
            "    - not_holds",
            "",
        ]
    )
    input_ref = "de:statutes/bgb/1591#input.candidate_person_identifier"
    original = yaml.safe_load(content)
    repaired = _insert_input_default_in_test_cases(content, input_ref, "person-a")
    parsed = yaml.safe_load(repaired)
    assert parsed[0]["input"] == {input_ref: "person-a"}
    assert parsed[0]["tables"] == original[0]["tables"]
    assert parsed[0]["output"] == original[0]["output"]
    assert (
        _insert_input_default_in_test_cases(repaired, input_ref, "person-a") == repaired
    )
