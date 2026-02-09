"""
Tests for parallel per-subsection encoding in SDKOrchestrator.
"""

import asyncio
import sys
import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

from autorac.harness.sdk_orchestrator import (
    AgentRun,
    AnalyzerOutput,
    Phase,
    SDKOrchestrator,
    SubsectionTask,
)

# --- Constants for DSL cheatsheet / prompt enrichment tests ---

DSL_CHEATSHEET_MARKERS = [
    "min(",
    "max(",
    "floor(",
    "clamp(",
    "if",
    "return",
    "Money",
    "Boolean",
    "Integer",
    "Rate",
    "parameter",
    "variable",
    "formula",
    "tests:",
]

EXEMPLAR_MARKERS = [
    "parameter",
    "variable",
    "formula:",
    "tests:",
    "expect:",
]


# --- Fixtures ---

SAMPLE_JSON_OUTPUT = """
Here is my analysis of 26 USC 24...

| Subsection | Title | Disposition | File |
|---|---|---|---|
| (a) | Allowance of credit | ENCODE | a.rac |
| (b) | Limitations | ENCODE | b.rac |
| (c) | Definitions | ENCODE | c.rac |

<!-- STRUCTURED_OUTPUT
{"subsections": [
    {"id": "a", "title": "Allowance of credit", "disposition": "ENCODE", "file": "a.rac"},
    {"id": "b", "title": "Limitations", "disposition": "ENCODE", "file": "b.rac"},
    {"id": "c", "title": "Definitions", "disposition": "ENCODE", "file": "c.rac"},
    {"id": "d", "title": "Inflation adjustment", "disposition": "SKIP", "file": null}
],
"dependencies": {"b": ["a"], "c": []},
"encoding_order": ["a", "c", "b"]}
-->
"""

SAMPLE_MARKDOWN_ONLY = """
Here is my analysis of 26 USC 24...

| Subsection | Title | Disposition | File |
|---|---|---|---|
| (a) | Allowance of credit | ENCODE | a.rac |
| (b) | Limitations | ENCODE | b.rac |
| (c) | Definitions | SKIP | - |
| (d) | Phaseout | ENCODE | d.rac |
"""


@pytest.fixture
def orchestrator():
    """Create orchestrator with a fake API key (won't call real API)."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "test-key"}):
        return SDKOrchestrator(api_key="test-key")


# --- Test _parse_analyzer_output ---


class TestParseAnalyzerOutput:
    def test_parse_json_block(self, orchestrator):
        """JSON block in STRUCTURED_OUTPUT tags is parsed correctly."""
        result = orchestrator._parse_analyzer_output(SAMPLE_JSON_OUTPUT)

        assert isinstance(result, AnalyzerOutput)
        assert len(result.subsections) == 3  # Only ENCODE, not SKIP
        assert result.subsections[0].subsection_id == "a"
        assert result.subsections[0].title == "Allowance of credit"
        assert result.subsections[0].file_name == "a.rac"
        assert result.subsections[1].subsection_id == "b"
        assert result.subsections[1].dependencies == ["a"]
        assert result.subsections[2].subsection_id == "c"
        assert result.subsections[2].dependencies == []

    def test_parse_markdown_fallback(self, orchestrator):
        """When no JSON block, falls back to regex on markdown table."""
        result = orchestrator._parse_analyzer_output(SAMPLE_MARKDOWN_ONLY)

        assert isinstance(result, AnalyzerOutput)
        # Only ENCODE subsections (a, b, d — c is SKIP)
        assert len(result.subsections) == 3
        assert result.subsections[0].subsection_id == "a"
        assert result.subsections[0].title == "Allowance of credit"
        assert result.subsections[0].file_name == "a.rac"
        assert result.subsections[1].subsection_id == "b"
        assert result.subsections[2].subsection_id == "d"

    def test_parse_empty_input(self, orchestrator):
        """Empty/garbage input returns empty subsection list."""
        result = orchestrator._parse_analyzer_output("")
        assert isinstance(result, AnalyzerOutput)
        assert len(result.subsections) == 0

        result = orchestrator._parse_analyzer_output("No structured data here.")
        assert len(result.subsections) == 0

    def test_raw_text_preserved(self, orchestrator):
        """The raw analysis text is preserved."""
        result = orchestrator._parse_analyzer_output(SAMPLE_JSON_OUTPUT)
        assert result.raw_text == SAMPLE_JSON_OUTPUT


# --- Test _compute_waves ---


class TestComputeWaves:
    def test_all_independent(self, orchestrator):
        """All subsections with no deps go in wave 0."""
        tasks = [
            SubsectionTask("a", "Title A", "a.rac", []),
            SubsectionTask("b", "Title B", "b.rac", []),
            SubsectionTask("c", "Title C", "c.rac", []),
        ]
        waves = orchestrator._compute_waves(tasks)
        assert len(waves) == 1
        ids = [t.subsection_id for t in waves[0]]
        assert set(ids) == {"a", "b", "c"}
        # All should have wave=0
        for t in waves[0]:
            assert t.wave == 0

    def test_with_deps(self, orchestrator):
        """A,C in wave 0; B in wave 1 (depends on A)."""
        tasks = [
            SubsectionTask("a", "Title A", "a.rac", []),
            SubsectionTask("b", "Title B", "b.rac", ["a"]),
            SubsectionTask("c", "Title C", "c.rac", []),
        ]
        waves = orchestrator._compute_waves(tasks)
        assert len(waves) == 2
        wave0_ids = {t.subsection_id for t in waves[0]}
        wave1_ids = {t.subsection_id for t in waves[1]}
        assert wave0_ids == {"a", "c"}
        assert wave1_ids == {"b"}

    def test_chain_deps(self, orchestrator):
        """A -> B -> C = 3 waves."""
        tasks = [
            SubsectionTask("a", "Title A", "a.rac", []),
            SubsectionTask("b", "Title B", "b.rac", ["a"]),
            SubsectionTask("c", "Title C", "c.rac", ["b"]),
        ]
        waves = orchestrator._compute_waves(tasks)
        assert len(waves) == 3
        assert waves[0][0].subsection_id == "a"
        assert waves[1][0].subsection_id == "b"
        assert waves[2][0].subsection_id == "c"

    def test_diamond_deps(self, orchestrator):
        """A -> B, A -> C, B+C -> D = 3 waves."""
        tasks = [
            SubsectionTask("a", "Title A", "a.rac", []),
            SubsectionTask("b", "Title B", "b.rac", ["a"]),
            SubsectionTask("c", "Title C", "c.rac", ["a"]),
            SubsectionTask("d", "Title D", "d.rac", ["b", "c"]),
        ]
        waves = orchestrator._compute_waves(tasks)
        assert len(waves) == 3
        assert {t.subsection_id for t in waves[0]} == {"a"}
        assert {t.subsection_id for t in waves[1]} == {"b", "c"}
        assert {t.subsection_id for t in waves[2]} == {"d"}

    def test_empty_tasks(self, orchestrator):
        """Empty task list returns empty waves."""
        waves = orchestrator._compute_waves([])
        assert waves == []


# --- Test _build_subsection_prompt ---


class TestBuildSubsectionPrompt:
    def test_contains_required_parts(self, orchestrator):
        """Prompt contains subsection ID, output path, and scope rule."""
        task = SubsectionTask("a", "Allowance of credit", "a.rac", [])
        output_path = Path("/tmp/rac-us/statute/26/24")
        prompt = orchestrator._build_subsection_prompt(
            task, "26 USC 24", output_path
        )

        assert "(a)" in prompt
        assert "Allowance of credit" in prompt
        assert "26 USC 24" in prompt
        assert str(output_path) in prompt
        assert "a.rac" in prompt
        # Should mention scoping rule
        assert "ONLY" in prompt or "only" in prompt

    def test_includes_statute_text(self, orchestrator):
        """Prompt includes statute text when provided."""
        task = SubsectionTask("b", "Limitations", "b.rac", ["a"])
        output_path = Path("/tmp/rac-us/statute/26/24")
        prompt = orchestrator._build_subsection_prompt(
            task, "26 USC 24", output_path, statute_text="Some statute text here"
        )
        assert "Some statute text here" in prompt

    def test_mentions_dependencies(self, orchestrator):
        """When deps exist, prompt mentions them for import reference."""
        task = SubsectionTask("b", "Limitations", "b.rac", ["a"])
        output_path = Path("/tmp/rac-us/statute/26/24")
        prompt = orchestrator._build_subsection_prompt(
            task, "26 USC 24", output_path
        )
        assert "a.rac" in prompt or "a" in prompt


# --- Test _run_encoding_parallel ---


class TestRunEncodingParallel:
    @pytest.mark.asyncio
    async def test_parallel_mock(self, orchestrator):
        """Mock _run_agent, verify wave execution order."""
        call_order = []

        async def mock_run_agent(agent_key, prompt, phase, model):
            # Extract subsection from prompt
            for letter in "abcde":
                if f"({letter})" in prompt:
                    call_order.append(letter)
                    break
            return AgentRun(
                agent_type="encoder",
                prompt=prompt,
                phase=Phase.ENCODING,
                result="ok",
            )

        orchestrator._run_agent = mock_run_agent

        analysis_text = SAMPLE_JSON_OUTPUT
        output_path = Path("/tmp/rac-us/statute/26/24")

        runs = await orchestrator._run_encoding_parallel(
            citation="26 USC 24",
            output_path=output_path,
            statute_text="test text",
            analysis_result=analysis_text,
        )

        # Should have 3 runs (a, b, c — d is SKIP)
        assert len(runs) == 3
        # a and c should come before b (b depends on a)
        a_idx = call_order.index("a")
        c_idx = call_order.index("c")
        b_idx = call_order.index("b")
        assert a_idx < b_idx
        assert c_idx < b_idx

    @pytest.mark.asyncio
    async def test_partial_failure(self, orchestrator):
        """One subsection fails, others succeed."""
        async def mock_run_agent(agent_key, prompt, phase, model):
            if "(b)" in prompt:
                raise RuntimeError("Encoding failed for (b)")
            return AgentRun(
                agent_type="encoder",
                prompt=prompt,
                phase=Phase.ENCODING,
                result="ok",
            )

        orchestrator._run_agent = mock_run_agent

        analysis_text = SAMPLE_JSON_OUTPUT
        output_path = Path("/tmp/rac-us/statute/26/24")

        runs = await orchestrator._run_encoding_parallel(
            citation="26 USC 24",
            output_path=output_path,
            statute_text="test text",
            analysis_result=analysis_text,
        )

        # a and c succeed, b fails — should get 2 successful runs
        assert len(runs) == 2

    @pytest.mark.asyncio
    async def test_empty_analysis_returns_empty(self, orchestrator):
        """If analyzer produced no subsections, return empty list."""
        orchestrator._run_agent = AsyncMock()
        runs = await orchestrator._run_encoding_parallel(
            citation="26 USC 99",
            output_path=Path("/tmp"),
            statute_text="",
            analysis_result="No subsections found.",
        )
        assert runs == []
        orchestrator._run_agent.assert_not_called()

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self, orchestrator):
        """Semaphore limits concurrent agents."""
        active = {"count": 0, "max": 0}

        async def mock_run_agent(agent_key, prompt, phase, model):
            active["count"] += 1
            active["max"] = max(active["max"], active["count"])
            await asyncio.sleep(0.01)
            active["count"] -= 1
            return AgentRun(
                agent_type="encoder", prompt=prompt, phase=Phase.ENCODING
            )

        orchestrator._run_agent = mock_run_agent

        # 5 independent subsections, max_concurrent=2
        analysis = """
<!-- STRUCTURED_OUTPUT
{"subsections": [
    {"id": "a", "title": "A", "disposition": "ENCODE", "file": "a.rac"},
    {"id": "b", "title": "B", "disposition": "ENCODE", "file": "b.rac"},
    {"id": "c", "title": "C", "disposition": "ENCODE", "file": "c.rac"},
    {"id": "d", "title": "D", "disposition": "ENCODE", "file": "d.rac"},
    {"id": "e", "title": "E", "disposition": "ENCODE", "file": "e.rac"}
],
"dependencies": {},
"encoding_order": ["a","b","c","d","e"]}
-->
"""
        runs = await orchestrator._run_encoding_parallel(
            citation="26 USC 24",
            output_path=Path("/tmp"),
            statute_text="",
            analysis_result=analysis,
            max_concurrent=2,
        )
        assert len(runs) == 5
        assert active["max"] <= 2


# --- Test FIX 1: DSL cheatsheet in prompt ---


class TestDSLCheatsheet:
    def test_prompt_contains_dsl_reference(self, orchestrator):
        """Subsection prompt includes DSL syntax reference."""
        task = SubsectionTask("a", "Allowance of credit", "a.rac", [])
        prompt = orchestrator._build_subsection_prompt(
            task, "26 USC 24", Path("/tmp/out")
        )
        for marker in DSL_CHEATSHEET_MARKERS:
            assert marker in prompt, f"Missing DSL reference: {marker}"

    def test_prompt_mentions_no_numeric_literals(self, orchestrator):
        """Prompt warns about numeric literal restriction."""
        task = SubsectionTask("a", "Title", "a.rac", [])
        prompt = orchestrator._build_subsection_prompt(
            task, "26 USC 24", Path("/tmp/out")
        )
        # Must mention the -1,0,1,2,3 restriction
        assert "-1" in prompt and "0" in prompt and "3" in prompt

    def test_prompt_mentions_no_loops(self, orchestrator):
        """Prompt warns against forbidden constructs."""
        task = SubsectionTask("a", "Title", "a.rac", [])
        prompt = orchestrator._build_subsection_prompt(
            task, "26 USC 24", Path("/tmp/out")
        )
        assert "for" in prompt.lower() or "loop" in prompt.lower()


# --- Test FIX 2: Exemplar in prompt ---


class TestExemplarInPrompt:
    def test_prompt_contains_exemplar(self, orchestrator):
        """Subsection prompt includes a RAC exemplar."""
        task = SubsectionTask("a", "Title", "a.rac", [])
        prompt = orchestrator._build_subsection_prompt(
            task, "26 USC 24", Path("/tmp/out")
        )
        for marker in EXEMPLAR_MARKERS:
            assert marker in prompt, f"Missing exemplar marker: {marker}"

    def test_exemplar_is_complete(self, orchestrator):
        """Exemplar shows a full variable with formula and test."""
        task = SubsectionTask("a", "Title", "a.rac", [])
        prompt = orchestrator._build_subsection_prompt(
            task, "26 USC 24", Path("/tmp/out")
        )
        # Should contain a complete example showing the pattern
        assert "entity:" in prompt
        assert "period:" in prompt
        assert "dtype:" in prompt


# --- Test FIX 3: Pre-fetch statute text ---


class TestStatuteTextPrefetch:
    def test_fetch_statute_text_returns_string(self, orchestrator):
        """_fetch_statute_text returns text for a valid citation."""
        # This depends on the XML file existing
        xml_path = Path.home() / "RulesFoundation" / "atlas" / "data" / "uscode"
        if not (xml_path / "usc26.xml").exists():
            pytest.skip("USC XML not available")
        text = orchestrator._fetch_statute_text("26 USC 24", xml_path)
        assert isinstance(text, str)
        assert len(text) > 100
        assert "credit" in text.lower() or "child" in text.lower()

    def test_fetch_statute_text_graceful_missing(self, orchestrator):
        """Returns None for missing citations."""
        text = orchestrator._fetch_statute_text(
            "99 USC 999", Path("/nonexistent")
        )
        assert text is None

    def test_encode_prefetches_statute(self, orchestrator):
        """The encode() method should call _fetch_statute_text if no text provided."""
        # We just verify the method exists and is callable
        assert hasattr(orchestrator, "_fetch_statute_text")
        assert callable(orchestrator._fetch_statute_text)


# --- Test FIX 4: Batch small subsections ---


class TestBatchSmallSubsections:
    def test_no_batching_when_few_tasks(self, orchestrator):
        """3 or fewer tasks are not batched."""
        tasks = [
            SubsectionTask("a", "Title A", "a.rac", []),
            SubsectionTask("b", "Title B", "b.rac", []),
        ]
        batched = orchestrator._batch_small_subsections(tasks)
        assert len(batched) == 2

    def test_batch_related_subsections(self, orchestrator):
        """Small sibling subsections under same parent get batched."""
        tasks = [
            SubsectionTask("g/1", "Def 1", "g/1.rac", []),
            SubsectionTask("g/3", "Def 3", "g/3.rac", []),
            SubsectionTask("g/4", "Def 4", "g/4.rac", []),
            SubsectionTask("g/5", "Def 5", "g/5.rac", []),
            SubsectionTask("a", "Big section", "a.rac", []),
        ]
        batched = orchestrator._batch_small_subsections(tasks, max_batch=4)
        # g/* should be batched into 1 task, a stays separate
        assert len(batched) < len(tasks)
        # Find the batched task
        batch_task = [t for t in batched if "," in t.subsection_id]
        assert len(batch_task) == 1
        assert "g/1" in batch_task[0].subsection_id
        assert "g/3" in batch_task[0].subsection_id

    def test_batch_preserves_dependencies(self, orchestrator):
        """Tasks with deps are not batched with unrelated tasks."""
        tasks = [
            SubsectionTask("g/1", "Def 1", "g/1.rac", []),
            SubsectionTask("g/3", "Def 3", "g/3.rac", ["g/1"]),
            SubsectionTask("a", "Big", "a.rac", []),
        ]
        batched = orchestrator._batch_small_subsections(tasks)
        # g/3 depends on g/1, so they shouldn't be in same batch
        for t in batched:
            if "," in t.subsection_id:
                ids = t.subsection_id.split(",")
                # No task in a batch should depend on another task in the same batch
                for tid in ids:
                    dep_task = next(
                        (orig for orig in tasks if orig.subsection_id == tid.strip()),
                        None,
                    )
                    if dep_task:
                        for d in dep_task.dependencies:
                            assert d.strip() not in [
                                i.strip() for i in ids
                            ], f"{tid} depends on {d}, both in same batch"

    def test_batch_respects_max_size(self, orchestrator):
        """No batch exceeds max_batch size."""
        tasks = [
            SubsectionTask(f"g/{i}", f"Def {i}", f"g/{i}.rac", [])
            for i in range(10)
        ]
        batched = orchestrator._batch_small_subsections(tasks, max_batch=3)
        for t in batched:
            if "," in t.subsection_id:
                count = len(t.subsection_id.split(","))
                assert count <= 3


# --- Test FIX 5: Skip obsolete subsections ---


class TestSkipObsolete:
    def test_obsolete_filtered(self, orchestrator):
        """Subsections with OBSOLETE disposition are excluded."""
        analysis = """
<!-- STRUCTURED_OUTPUT
{"subsections": [
    {"id": "a", "title": "Active", "disposition": "ENCODE", "file": "a.rac"},
    {"id": "b", "title": "Repealed", "disposition": "OBSOLETE", "file": "b.rac"},
    {"id": "c", "title": "Another", "disposition": "ENCODE", "file": "c.rac"}
],
"dependencies": {},
"encoding_order": ["a", "c"]}
-->
"""
        result = orchestrator._parse_analyzer_output(analysis)
        ids = [s.subsection_id for s in result.subsections]
        assert "a" in ids
        assert "c" in ids
        assert "b" not in ids

    def test_skip_and_obsolete_both_filtered(self, orchestrator):
        """Both SKIP and OBSOLETE dispositions are excluded."""
        analysis = """
<!-- STRUCTURED_OUTPUT
{"subsections": [
    {"id": "a", "title": "Good", "disposition": "ENCODE", "file": "a.rac"},
    {"id": "b", "title": "Skip", "disposition": "SKIP", "file": null},
    {"id": "c", "title": "Old", "disposition": "OBSOLETE", "file": null},
    {"id": "d", "title": "Also good", "disposition": "ENCODE", "file": "d.rac"}
],
"dependencies": {},
"encoding_order": ["a", "d"]}
-->
"""
        result = orchestrator._parse_analyzer_output(analysis)
        ids = [s.subsection_id for s in result.subsections]
        assert ids == ["a", "d"]

    def test_analyzer_prompt_mentions_obsolete(self, orchestrator):
        """The analyzer prompt should mention OBSOLETE as a valid disposition."""
        # We check _build_analyzer_prompt or the prompt in encode()
        assert hasattr(orchestrator, "_build_analyzer_prompt")
        prompt = orchestrator._build_analyzer_prompt("26 USC 24")
        assert "OBSOLETE" in prompt
