from pathlib import Path

from agentic_memory_long_context_bench.generator import generate_dataset, write_dataset
from agentic_memory_long_context_bench.llm import _extract_classifier_payload
from agentic_memory_long_context_bench.memory_mode import (
    AgenticMemoryAdapter,
    HeuristicTurnClassifier,
    _extract_steps_from_text,
)
from agentic_memory_long_context_bench.reporting import build_summary_payload, summarize_rows
from agentic_memory_long_context_bench.runner import _build_mode_prompt, _select_judged_keys, render_report
from agentic_memory_long_context_bench.schema import BenchmarkExample, Gold, Task, Turn


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
    assert memory_trace["classification"]["accuracy"] > 0
    assert "rows" not in memory_trace["classification"]
    assert memory_trace["classification"]["tokens"] == {"input": 0, "output": 0}
    assert prompt_metadata["mode"] == "memory_enabled"
    assert prompt_metadata["oracle_labels"] is False


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


def test_memory_mode_can_use_oracle_labels():
    rows = generate_dataset(examples=1, seed=7)
    example = rows[0]

    _, prompt_metadata, memory_trace = _build_mode_prompt(
        example=example,
        mode="memory_enabled",
        short_context_tokens=8000,
        full_context_budget=250000,
        oracle_labels=True,
    )

    assert memory_trace is not None
    assert memory_trace["classification"]["source"] == "oracle_labels"
    assert memory_trace["classification"]["accuracy"] == 1.0
    assert prompt_metadata["oracle_labels"] is True


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


def test_classifier_parse_fallback_marks_error():
    payload = _extract_classifier_payload("not valid json")

    assert payload["type"] == "noise"
    assert payload["_parse_error"] is True


def test_extract_steps_from_text_falls_back_to_text_segments():
    steps = _extract_steps_from_text(
        "A previous successful procedure for this class of issue was: Troubleshoot a login loop after an SSO or cookie mismatch."
    )

    assert steps == ["Troubleshoot a login loop after an SSO or cookie mismatch."]


def test_heuristic_classifier_uses_general_signals():
    classifier = HeuristicTurnClassifier()

    correction = classifier.classify(role="user", text="Actually, I'm on the pro plan.")
    procedure = classifier.classify(role="assistant", text="Troubleshoot the login failure with this runbook.")
    event = classifier.classify(role="user", text="I'm seeing an error after I tried the reset flow.")

    assert correction.type == "correction"
    assert procedure.type == "procedure"
    assert event.type == "event"


def test_non_oracle_adapter_does_not_store_kind_or_procedure_key():
    class FakeMemory:
        def __init__(self) -> None:
            self.episodes: list[dict] = []
            self.procedures: list[dict] = []

        def remember_episode(self, content: str, **kwargs):
            self.episodes.append({"content": content, **kwargs})

        def remember_procedure(self, content: str, **kwargs):
            self.procedures.append({"content": content, **kwargs})

        def recall(self, query: str, top_k: int, types: list[str]):
            return []

        def recall_episodes(self, mode: str, limit: int):
            return []

        def recall_procedures(self, query: str, top_k: int):
            return []

    class StubClassifier:
        def __init__(self, forced_type: str) -> None:
            self.forced_type = forced_type

        def classify(self, *, role: str, text: str):
            from agentic_memory_long_context_bench.llm import ClassificationResult

            return ClassificationResult(
                type=self.forced_type,
                field=None,
                supersedes_description=None,
                raw={},
            )

    example = BenchmarkExample(
        id="ex",
        seed=1,
        scenario_type="procedure_reuse",
        difficulty="easy",
        conversation=[],
        task=Task(prompt="prompt", requires=[]),
        gold=Gold(must_include=[], must_not_include=[], supporting_fact_ids=[], stale_fact_ids=[]),
        metadata={},
    )
    memory = FakeMemory()

    event_adapter = AgenticMemoryAdapter(
        memory=memory,
        example=example,
        classifier=StubClassifier("event"),
        oracle_labels=False,
    )
    event_adapter.ingest(turn=Turn(role="user", turn_index=0, kind="event", text="I am seeing an error."))

    procedure_adapter = AgenticMemoryAdapter(
        memory=memory,
        example=example,
        classifier=StubClassifier("procedure"),
        oracle_labels=False,
    )
    procedure_adapter.ingest(
        turn=Turn(role="assistant", turn_index=1, kind="procedure_outcome", text="Procedure: Restart the service.")
    )

    assert memory.episodes[0]["summary"] == "event"
    assert "kind" not in memory.episodes[0]["metadata"]
    assert "procedure_key" not in memory.procedures[0]["metadata"]
    assert memory.procedures[0]["steps"] == ["Restart the service."]


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


def test_summary_payload_includes_classification_metrics():
    rows = [
        {
            "example_id": "profile_recall_easy_0000",
            "mode": "memory_enabled",
            "model": "gemini-2.5-flash-lite",
            "judge_model": "gemini-3-flash-preview",
            "latency_ms": 100.0,
            "prompt_tokens": 10,
            "cost_usd": 0.001,
            "response_text": "ok",
            "rule_score": {"passed": True, "hallucination_flag": False},
            "judge_score": {"overall_score": 0.8},
            "prompt_metadata": {"estimated_transcript_tokens": 100},
            "memory_trace": {
                "classification": {
                    "source": "classifier",
                    "backend": "HeuristicTurnClassifier",
                    "total_turns": 5,
                    "evaluated_turns": 4,
                    "correct_type": 3,
                    "wrong_type": 1,
                    "missed_correction": 1,
                    "missed_procedure": 0,
                    "parse_errors": 1,
                    "tokens": {"input": 20, "output": 5},
                    "cost_usd": 0.000123,
                    "accuracy": 0.75,
                }
            },
        },
        {
            "example_id": "profile_recall_easy_0000",
            "mode": "full_context",
            "model": "gemini-2.5-flash-lite",
            "judge_model": "gemini-3-flash-preview",
            "latency_ms": 120.0,
            "prompt_tokens": 20,
            "cost_usd": 0.002,
            "response_text": "ok",
            "rule_score": {"passed": False, "hallucination_flag": False},
            "judge_score": {"overall_score": 0.6},
            "prompt_metadata": {"estimated_transcript_tokens": 100},
            "memory_trace": None,
        },
        {
            "example_id": "profile_recall_easy_0000",
            "mode": "short_context",
            "model": "gemini-2.5-flash-lite",
            "judge_model": "gemini-3-flash-preview",
            "latency_ms": 80.0,
            "prompt_tokens": 5,
            "cost_usd": 0.0005,
            "response_text": "ok",
            "rule_score": {"passed": False, "hallucination_flag": False},
            "judge_score": {"overall_score": 0.4},
            "prompt_metadata": {"estimated_transcript_tokens": 100},
            "memory_trace": None,
        },
    ]

    payload = build_summary_payload(
        rows=rows,
        summaries=summarize_rows(rows),
        title="Test",
        results_path=Path("results.jsonl"),
    )

    assert payload["classification_summary"]["accuracy"] == 0.75
    assert payload["classification_summary"]["parse_errors"] == 1
    assert payload["classification_summary"]["cost_usd"] == 0.000123
    assert "classification accuracy was 75.0%" in payload["key_findings"][-2]
    assert "added 25 tokens and $0.000123" in payload["key_findings"][-1]
