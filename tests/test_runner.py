from pathlib import Path

from agentic_memory_long_context_bench.generator import generate_dataset, write_dataset
from agentic_memory_long_context_bench.runner import _build_mode_prompt


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
    assert prompt_metadata["included_turns"] < len(example.conversation)
    assert memory_trace is None
