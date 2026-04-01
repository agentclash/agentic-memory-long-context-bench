from __future__ import annotations

import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agentic_memory import Memory

from .deterministic import HashingEmbedder
from .generator import _estimate_text_tokens
from .schema import BenchmarkExample, Turn
from .vocab import PROCEDURES


@dataclass(frozen=True)
class MemoryTrace:
    semantic: list[dict[str, Any]]
    episodic_ranked: list[dict[str, Any]]
    episodic_recent: list[dict[str, Any]]
    procedural: list[dict[str, Any]]
    stored_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_memory_prompt(example: BenchmarkExample) -> tuple[str, MemoryTrace]:
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
            state = _ingest_example(memory, example)

            semantic_results = memory.recall(example.task.prompt, top_k=6, types=["semantic"])
            episodic_ranked = memory.recall(example.task.prompt, top_k=6, types=["episodic"])
            episodic_recent = memory.recall_episodes(mode="recent", limit=6)
            procedural_matches = memory.recall_procedures(example.task.prompt, top_k=3)

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
                stored_counts=state["stored_counts"],
            )
            return prompt, trace


def _ingest_example(memory: Memory, example: BenchmarkExample) -> dict[str, Any]:
    stored_counts = {"semantic": 0, "episodic": 0, "procedural": 0}
    semantic_field_map: dict[str, str] = {}
    fact_id_to_memory_id: dict[str, str] = {}

    for turn in example.conversation:
        if turn.kind in {"durable_fact", "correction"}:
            field_name = _infer_semantic_field(turn.text)
            supersedes = semantic_field_map.get(field_name) if turn.kind == "correction" and field_name else None
            memory_id = memory.remember(
                turn.text,
                category=_semantic_category(turn.text),
                domain="long_context_bench",
                supersedes=supersedes,
                metadata={"turn_index": turn.turn_index, "kind": turn.kind, "field": field_name},
            )
            stored_counts["semantic"] += 1
            if turn.fact_id:
                fact_id_to_memory_id[turn.fact_id] = memory_id
            if field_name:
                semantic_field_map[field_name] = memory_id
            continue

        if turn.kind in {"event", "attempted_step", "ack"}:
            memory.remember_episode(
                turn.text,
                session=example.id,
                turn=turn.turn_index,
                participants=[turn.role],
                summary=turn.kind,
                metadata={"turn_index": turn.turn_index, "kind": turn.kind},
            )
            stored_counts["episodic"] += 1
            continue

        if turn.kind == "procedure_outcome":
            procedure = PROCEDURES[example.metadata["procedure_key"]]
            memory.remember_procedure(
                turn.text,
                steps=list(procedure["steps"]),
                importance=0.9,
                metadata={"turn_index": turn.turn_index, "kind": turn.kind, "procedure_key": example.metadata["procedure_key"]},
            )
            stored_counts["procedural"] += 1

    return {"stored_counts": stored_counts, "fact_id_to_memory_id": fact_id_to_memory_id}


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
    if "theme" in lowered:
        return "theme"
    if "notifications" in lowered:
        return "notifications"
    if "i use a" in lowered:
        return "device"
    return "general"


def _semantic_category(text: str) -> str:
    field = _infer_semantic_field(text)
    if field in {"name", "plan", "timezone", "device"}:
        return "profile"
    if field in {"theme", "notifications"}:
        return "preference"
    return "general"


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
