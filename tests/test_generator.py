import json
from pathlib import Path

from agentic_memory_long_context_bench.generator import generate_dataset, write_dataset


def test_generator_produces_stable_shape(tmp_path: Path):
    rows = generate_dataset(examples=5, seed=7)
    assert len(rows) == 5
    assert rows[0].scenario_type == "profile_recall"
    assert rows[1].scenario_type == "troubleshooting_continuity"
    assert rows[2].scenario_type == "contradiction_resolution"
    assert rows[3].scenario_type == "procedure_reuse"
    assert rows[4].scenario_type == "mixed_long_context"

    output = tmp_path / "sample.jsonl"
    write_dataset(rows, output)
    lines = output.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 5

    first = json.loads(lines[0])
    assert "conversation" in first
    assert "task" in first
    assert "gold" in first
    assert first["gold"]["must_include"]
    assert first["task"]["requires"]
