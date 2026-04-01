from pathlib import Path

from agentic_memory_long_context_bench.generator import generate_dataset, write_dataset
from agentic_memory_long_context_bench.runner import _build_mode_prompt, _select_judged_keys, render_report


def test_memory_mode_exposes_all_memory_types(tmp_path: Path):
    path = tmp_path / "sample.jsonl"
    rows = generate_dataset(examples=1, seed=7)
    write_dataset(rows, path)
    example = rows[0]

    prompt, prompt_metadata, memory_trace = _build_mode_prompt(
        example=example,
        mode="memory_enabled",
        short_context_tokens=8000,
        full_context_budget=250000,
    )

    assert "SEMANTIC MEMORY" in prompt
    assert "EPISODIC MEMORY (RANKED)" in prompt
    assert "EPISODIC MEMORY (RECENT)" in prompt
    assert "PROCEDURAL MEMORY" in prompt
    assert memory_trace is not None
    assert memory_trace["stored_counts"]["semantic"] >= 1
    assert memory_trace["stored_counts"]["episodic"] >= 1
    assert memory_trace["stored_counts"]["procedural"] >= 1
    assert prompt_metadata["mode"] == "memory_enabled"


def test_short_context_prompt_trims_turns():
    rows = generate_dataset(examples=1, seed=7, min_tokens=2000, context_tier="test")
    example = rows[0]

    prompt, prompt_metadata, memory_trace = _build_mode_prompt(
        example=example,
        mode="short_context",
        short_context_tokens=300,
        full_context_budget=250000,
    )

    assert "TRANSCRIPT" in prompt
    assert "RELEVANT_FACTS:" in prompt
    assert "Prefer newer corrected facts" in prompt
    assert prompt_metadata["included_turns"] < len(example.conversation)
    assert memory_trace is None


def test_memory_prompt_uses_shared_answer_scaffold(tmp_path: Path):
    path = tmp_path / "sample.jsonl"
    rows = generate_dataset(examples=1, seed=7)
    write_dataset(rows, path)
    example = rows[0]

    prompt, _, _ = _build_mode_prompt(
        example=example,
        mode="memory_enabled",
        short_context_tokens=8000,
        full_context_budget=250000,
    )

    assert "RELEVANT_FACTS:" in prompt
    assert "ANSWER:" in prompt
    assert "Prefer newer corrected facts" in prompt


def test_select_judged_keys_uses_balanced_ratio():
    rows = generate_dataset(examples=10, seed=7)
    judged = _select_judged_keys(
        examples=rows,
        modes=["short_context", "full_context", "memory_enabled"],
        sample_ratio=0.4,
        sample_size=0,
        seed=7,
        enabled=True,
    )

    assert len([key for key in judged if key.endswith("::short_context")]) == 4
    assert len([key for key in judged if key.endswith("::full_context")]) == 4
    assert len([key for key in judged if key.endswith("::memory_enabled")]) == 4


def test_render_report_includes_judged_rows():
    from agentic_memory_long_context_bench.runner import HarnessResult

    results = [
        HarnessResult(
            example_id="ex1",
            mode="short_context",
            model="gemini-2.5-flash-lite",
            judge_model="gemini-3-flash-preview",
            prompt_tokens=10,
            completion_tokens=5,
            latency_ms=100.0,
            cost_usd=0.001,
            response_text="ok",
            rule_score={"passed": True, "hallucination_flag": False},
            judge_score={"overall_score": 0.8},
            memory_trace=None,
            prompt_metadata={},
        ),
        HarnessResult(
            example_id="ex2",
            mode="short_context",
            model="gemini-2.5-flash-lite",
            judge_model="gemini-3-flash-preview",
            prompt_tokens=12,
            completion_tokens=6,
            latency_ms=120.0,
            cost_usd=0.001,
            response_text="ok",
            rule_score={"passed": False, "hallucination_flag": False},
            judge_score=None,
            memory_trace=None,
            prompt_metadata={},
        ),
    ]

    report = render_report(results)
    assert "judged_rows: 1/2 (50.0%)" in report
