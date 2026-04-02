from __future__ import annotations

import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from agentic_memory import Memory

from .deterministic import HashingEmbedder
from .generator import _estimate_text_tokens
from .llm import ClassificationResult, ClassifierLLM, TurnClassifier
from .pricing import estimate_cost_usd
from .schema import BenchmarkExample, Turn


class MemoryAdapter(Protocol):
    def ingest(self, *, turn: Turn) -> None:
        """Ingest a raw conversation turn into the adapter's memory system."""

    def recall(self, *, query: str) -> dict[str, list[Any]]:
        """Return retrieved memory buckets for prompt construction."""


@dataclass(frozen=True)
class MemoryTrace:
    semantic: list[dict[str, Any]]
    episodic_ranked: list[dict[str, Any]]
    episodic_recent: list[dict[str, Any]]
    procedural: list[dict[str, Any]]
    stored_counts: dict[str, int]
    classification: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_memory_prompt(
    example: BenchmarkExample,
    *,
    classifier_model: str | None = None,
    oracle_labels: bool = False,
) -> tuple[str, MemoryTrace]:
    with tempfile.TemporaryDirectory(prefix="long_context_harness_") as chroma_dir:
        with tempfile.TemporaryDirectory(prefix="long_context_media_") as media_dir:
            memory_kwargs: dict[str, Any] = {
                "chroma_path": str(Path(chroma_dir)),
                "media_root": str(Path(media_dir)),
            }
            if not os.getenv("GEMINI_API_KEY"):
                memory_kwargs["embedder"] = HashingEmbedder(dimensions=64)
                memory_kwargs["embedding_dimensions"] = 64

            memory = Memory(**memory_kwargs)
            classifier = _build_classifier(classifier_model=classifier_model, oracle_labels=oracle_labels)
            adapter = AgenticMemoryAdapter(
                memory=memory,
                example=example,
                classifier=classifier,
                oracle_labels=oracle_labels,
            )
            for turn in example.conversation:
                adapter.ingest(turn=turn)

            recalled = adapter.recall(query=example.task.prompt)
            semantic_results = recalled["semantic"]
            episodic_ranked = recalled["episodic_ranked"]
            episodic_recent = recalled["episodic_recent"]
            procedural_matches = recalled["procedural"]

            prompt = _format_memory_prompt(
                example=example,
                semantic_results=semantic_results,
                episodic_ranked=episodic_ranked,
                episodic_recent=episodic_recent,
                procedural_matches=procedural_matches,
            )
            trace = MemoryTrace(
                semantic=[_ranked_result_to_dict(result) for result in semantic_results],
                episodic_ranked=[_ranked_result_to_dict(result) for result in episodic_ranked],
                episodic_recent=[_record_to_dict(record) for record in episodic_recent],
                procedural=[_procedural_match_to_dict(match) for match in procedural_matches],
                stored_counts=adapter.stored_counts,
                classification=adapter.classification_summary(),
            )
            return prompt, trace


class AgenticMemoryAdapter:
    def __init__(
        self,
        *,
        memory: Memory,
        example: BenchmarkExample,
        classifier: TurnClassifier,
        oracle_labels: bool,
    ):
        self.memory = memory
        self.example = example
        self.classifier = classifier
        self.oracle_labels = oracle_labels
        self.stored_counts = {"semantic": 0, "episodic": 0, "procedural": 0}
        self.semantic_field_map: dict[str, str] = {}
        self.fact_id_to_memory_id: dict[str, str] = {}
        self.classification_rows: list[dict[str, Any]] = []
        self.classification_tokens = {"input": 0, "output": 0}
        self.classification_cost_usd = 0.0

    def ingest(self, *, turn: Turn) -> None:
        classification = (
            _classify_turn_with_oracle(turn)
            if self.oracle_labels
            else self.classifier.classify(role=turn.role, text=turn.text)
        )
        self._record_classification_usage(classification)
        self.classification_rows.append(
            {
                "turn_index": turn.turn_index,
                "role": turn.role,
                "oracle_kind": turn.kind,
                "predicted_type": classification.type,
                "field": classification.field,
                "supersedes_description": classification.supersedes_description,
                "parse_error": bool(classification.raw.get("parse_error")),
            }
        )

        if classification.type in {"fact", "correction"}:
            self._store_semantic(turn=turn, classification=classification)
            return
        if classification.type == "event":
            self.memory.remember_episode(
                turn.text,
                session=self.example.id,
                turn=turn.turn_index,
                participants=[turn.role],
                summary=classification.type,
                metadata=self._memory_metadata(turn=turn, classification=classification),
            )
            self.stored_counts["episodic"] += 1
            return
        if classification.type == "procedure":
            self.memory.remember_procedure(
                turn.text,
                steps=_extract_steps_from_text(turn.text),
                importance=0.9,
                metadata=self._memory_metadata(turn=turn, classification=classification),
            )
            self.stored_counts["procedural"] += 1

    def recall(self, *, query: str) -> dict[str, list[Any]]:
        return {
            "semantic": self.memory.recall(query, top_k=6, types=["semantic"]),
            "episodic_ranked": self.memory.recall(query, top_k=6, types=["episodic"]),
            "episodic_recent": self.memory.recall_episodes(mode="recent", limit=6),
            "procedural": self.memory.recall_procedures(query, top_k=3),
        }

    def classification_summary(self, *, include_rows: bool = False) -> dict[str, Any]:
        totals = {
            "total_turns": len(self.classification_rows),
            "correct_type": 0,
            "wrong_type": 0,
            "missed_correction": 0,
            "missed_procedure": 0,
            "parse_errors": 0,
            "source": "oracle_labels" if self.oracle_labels else "classifier",
            "backend": "oracle" if self.oracle_labels else self.classifier.__class__.__name__,
            "tokens": dict(self.classification_tokens),
            "cost_usd": round(self.classification_cost_usd, 6),
        }
        for row in self.classification_rows:
            expected_type = _expected_classification_type(row["oracle_kind"])
            if expected_type is None:
                continue
            if row["parse_error"]:
                totals["parse_errors"] += 1
            if row["predicted_type"] == expected_type:
                totals["correct_type"] += 1
            else:
                totals["wrong_type"] += 1
                if expected_type == "correction" and row["predicted_type"] == "fact":
                    totals["missed_correction"] += 1
                if expected_type == "procedure" and row["predicted_type"] == "noise":
                    totals["missed_procedure"] += 1
        classified_total = totals["correct_type"] + totals["wrong_type"]
        totals["evaluated_turns"] = classified_total
        totals["accuracy"] = round(totals["correct_type"] / classified_total, 4) if classified_total else 0.0
        if include_rows:
            totals["rows"] = self.classification_rows
        return totals

    def _store_semantic(self, *, turn: Turn, classification: ClassificationResult) -> None:
        field_name = classification.field or _infer_semantic_field(turn.text)
        supersedes = None
        if classification.type == "correction" and field_name:
            supersedes = self.semantic_field_map.get(field_name)
        memory_id = self.memory.remember(
            turn.text,
            category=_semantic_category(turn.text, field=field_name),
            domain="long_context_bench",
            supersedes=supersedes,
            metadata=self._memory_metadata(
                turn=turn,
                classification=classification,
                field=field_name,
            ),
        )
        self.stored_counts["semantic"] += 1
        if turn.fact_id:
            self.fact_id_to_memory_id[turn.fact_id] = memory_id
        if field_name:
            self.semantic_field_map[field_name] = memory_id

    def _memory_metadata(
        self,
        *,
        turn: Turn,
        classification: ClassificationResult,
        field: str | None = None,
    ) -> dict[str, Any]:
        metadata: dict[str, Any] = {
            "turn_index": turn.turn_index,
            "classified_type": classification.type,
        }
        if field:
            metadata["field"] = field
        if classification.supersedes_description:
            metadata["supersedes_description"] = classification.supersedes_description
        if self.oracle_labels:
            metadata["kind"] = turn.kind
        return metadata

    def _record_classification_usage(self, classification: ClassificationResult) -> None:
        response = classification.raw.get("response", {})
        input_tokens = int(response.get("input_tokens", 0) or 0)
        output_tokens = int(response.get("output_tokens", 0) or 0)
        self.classification_tokens["input"] += input_tokens
        self.classification_tokens["output"] += output_tokens
        model_name = str(classification.raw.get("model", "") or "")
        if model_name:
            self.classification_cost_usd += estimate_cost_usd(
                model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )


def _build_classifier(*, classifier_model: str | None, oracle_labels: bool) -> TurnClassifier:
    if oracle_labels:
        return HeuristicTurnClassifier()
    if os.getenv("GEMINI_API_KEY"):
        return ClassifierLLM(model=classifier_model or "gemini-2.5-flash-lite")
    return HeuristicTurnClassifier()


class HeuristicTurnClassifier:
    """Baseline classifier using broad linguistic cues rather than generator templates."""

    _FACT_PATTERNS = (
        "my name is",
        "i'm on the",
        "my timezone",
        "i use a",
        "i prefer",
        "i'm in",
        "my email",
        "my role",
    )
    _EVENT_PATTERNS = (
        "i tried",
        "already tried",
        "i attempted",
        "i'm seeing",
        "i'm getting",
        "started happening",
        "broke",
        "failed",
        "error",
        "issue",
        "problem",
        "bug",
    )
    _PROCEDURE_SIGNALS = (
        "step 1",
        "step 2",
        "first,",
        "then,",
        "finally,",
        "procedure",
        "workflow",
        "runbook",
        "troubleshoot",
        "troubleshooting",
    )
    _CORRECTION_SIGNALS = ("actually", "instead", "updated", "correction")

    def classify(self, *, role: str, text: str) -> ClassificationResult:
        lowered = text.lower()
        if role == "assistant" and _count_matches(lowered, self._PROCEDURE_SIGNALS) >= 1:
            return ClassificationResult(
                type="procedure",
                field=None,
                supersedes_description=None,
                raw={"backend": "heuristic"},
            )
        if any(signal in lowered for signal in self._CORRECTION_SIGNALS) and any(
            pattern in lowered for pattern in self._FACT_PATTERNS
        ):
            return ClassificationResult(
                type="correction",
                field=_infer_semantic_field(text),
                supersedes_description=_infer_semantic_field(text),
                raw={"backend": "heuristic"},
            )
        if any(pattern in lowered for pattern in self._EVENT_PATTERNS):
            return ClassificationResult(
                type="event",
                field=None,
                supersedes_description=None,
                raw={"backend": "heuristic"},
            )
        if role == "user" and any(pattern in lowered for pattern in self._FACT_PATTERNS):
            field = _infer_semantic_field(text)
            return ClassificationResult(
                type="fact",
                field=field,
                supersedes_description=None,
                raw={"backend": "heuristic"},
            )
        return ClassificationResult(
            type="noise",
            field=None,
            supersedes_description=None,
            raw={"backend": "heuristic"},
        )


def _classify_turn_with_oracle(turn: Turn) -> ClassificationResult:
    mapping = {
        "durable_fact": "fact",
        "correction": "correction",
        "event": "event",
        "attempted_step": "event",
        "ack": "noise",
        "procedure_outcome": "procedure",
    }
    return ClassificationResult(
        type=mapping.get(turn.kind, "noise"),
        field=_infer_semantic_field(turn.text) if turn.kind in {"durable_fact", "correction"} else None,
        supersedes_description=_infer_semantic_field(turn.text) if turn.kind == "correction" else None,
        raw={"backend": "oracle", "kind": turn.kind},
    )


def _format_memory_prompt(
    *,
    example: BenchmarkExample,
    semantic_results: list[Any],
    episodic_ranked: list[Any],
    episodic_recent: list[Any],
    procedural_matches: list[Any],
) -> str:
    semantic_lines = "\n".join(
        f"- {result.record.content}" for result in semantic_results
    ) or "- none"
    episodic_ranked_lines = "\n".join(
        f"- {result.record.content}" for result in episodic_ranked
    ) or "- none"
    episodic_recent_lines = "\n".join(
        f"- {record.content}" for record in episodic_recent
    ) or "- none"
    procedural_lines = "\n".join(
        f"- {match.record.content}\n  steps: {' | '.join(match.record.steps)}"
        for match in procedural_matches
    ) or "- none"

    return (
        "You are evaluating long-context memory retrieval quality.\n"
        "Use only the memory evidence below.\n"
        "First identify the facts directly relevant to the task.\n"
        "Prefer newer corrected facts over older stale facts.\n"
        "Do not suggest troubleshooting steps the user already tried.\n"
        "Ignore irrelevant memory items.\n\n"
        f"TASK:\n{example.task.prompt}\n\n"
        "SEMANTIC MEMORY:\n"
        f"{semantic_lines}\n\n"
        "EPISODIC MEMORY (RANKED):\n"
        f"{episodic_ranked_lines}\n\n"
        "EPISODIC MEMORY (RECENT):\n"
        f"{episodic_recent_lines}\n\n"
        "PROCEDURAL MEMORY:\n"
        f"{procedural_lines}\n\n"
        "Return this exact structure:\n"
        "RELEVANT_FACTS:\n"
        "- fact 1\n"
        "- fact 2\n"
        "ANSWER:\n"
        "your concise final answer"
    )


def _infer_semantic_field(text: str) -> str:
    lowered = text.lower()
    if "my name is" in lowered:
        return "name"
    if "plan" in lowered:
        return "plan"
    if "timezone" in lowered:
        return "timezone"
    if "theme" in lowered or "notifications" in lowered:
        return "preference"
    if "i use a" in lowered:
        return "device"
    return "general"


def _semantic_category(text: str, *, field: str | None = None) -> str:
    field = field or _infer_semantic_field(text)
    if field in {"name", "plan", "timezone", "device"}:
        return "profile"
    if field == "preference":
        return "preference"
    return "general"


def _looks_like_fact(lowered: str) -> bool:
    return any(
        marker in lowered
        for marker in (
            "my name is",
            "i'm on the",
            "my timezone is",
            "i prefer the",
            "i use a",
            "i prefer ",
        )
    )


def _count_matches(text: str, patterns: tuple[str, ...]) -> int:
    return sum(1 for pattern in patterns if pattern in text)


def _extract_steps_from_text(text: str) -> list[str]:
    candidate = text.partition(":")[2].strip() or text.strip()
    steps: list[str] = []
    for raw_line in candidate.splitlines():
        stripped = re.sub(r"^\s*(?:[-*]|\d+[.)-]?)\s*", "", raw_line).strip()
        if stripped and len(stripped.split()) >= 3:
            steps.append(stripped)
    if steps:
        return steps
    sentence_parts = [
        segment.strip()
        for segment in re.split(r";|\n", candidate)
        if segment.strip()
    ]
    return sentence_parts or [candidate]


def _expected_classification_type(kind: str) -> str | None:
    if kind == "durable_fact":
        return "fact"
    if kind == "correction":
        return "correction"
    if kind in {"event", "attempted_step"}:
        return "event"
    if kind == "procedure_outcome":
        return "procedure"
    if kind in {"ack", "distractor", "distractor_block"}:
        return "noise"
    return None


def _ranked_result_to_dict(result: Any) -> dict[str, Any]:
    return {
        "id": result.record.id,
        "memory_type": result.record.memory_type,
        "content": result.record.content,
        "raw_similarity": round(getattr(result, "raw_similarity", 0.0), 4),
        "estimated_tokens": _estimate_text_tokens(result.record.content),
    }


def _record_to_dict(record: Any) -> dict[str, Any]:
    return {
        "id": record.id,
        "memory_type": record.memory_type,
        "content": record.content,
        "estimated_tokens": _estimate_text_tokens(record.content),
    }


def _procedural_match_to_dict(match: Any) -> dict[str, Any]:
    return {
        "id": match.record.id,
        "memory_type": match.record.memory_type,
        "content": match.record.content,
        "steps": list(match.record.steps),
        "combined_score": round(getattr(match, "combined_score", 0.0), 4),
        "estimated_tokens": _estimate_text_tokens(match.record.content + " " + " ".join(match.record.steps)),
    }
