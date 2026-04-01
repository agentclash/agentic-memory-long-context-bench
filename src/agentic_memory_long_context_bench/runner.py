from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .dataset import load_dataset
from .generator import _estimate_conversation_tokens, _estimate_text_tokens
from .llm import GeminiJudge, GeminiModel
from .memory_mode import build_memory_prompt
from .pricing import estimate_cost_usd
from .scoring import score_response
from .schema import BenchmarkExample, Turn


@dataclass(frozen=True)
class HarnessResult:
    example_id: str
    mode: str
    model: str
    judge_model: str | None
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
    cost_usd: float
    response_text: str
    rule_score: dict[str, Any]
    judge_score: dict[str, Any] | None
    memory_trace: dict[str, Any] | None
    prompt_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run long-context evaluation modes against a dataset.")
    parser.add_argument("--dataset", type=Path, required=True, help="JSONL dataset path.")
    parser.add_argument("--output", type=Path, default=Path("results/latest_results.jsonl"))
    parser.add_argument("--model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--judge-model", type=str, default="gemini-2.5-flash-lite")
    parser.add_argument("--modes", type=str, default="short_context,full_context,memory_enabled")
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on examples to run.")
    parser.add_argument("--short-context-tokens", type=int, default=8000)
    parser.add_argument("--full-context-budget", type=int, default=250000)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_harness(
        dataset_path=args.dataset,
        output_path=args.output,
        model_name=args.model,
        judge_model_name=args.judge_model,
        modes=[mode.strip() for mode in args.modes.split(",") if mode.strip()],
        limit=args.limit,
        short_context_tokens=args.short_context_tokens,
        full_context_budget=args.full_context_budget,
        use_judge=not args.skip_judge,
        print_report=args.report,
    )


def run_harness(
    *,
    dataset_path: Path,
    output_path: Path,
    model_name: str,
    judge_model_name: str,
    modes: list[str],
    limit: int,
    short_context_tokens: int,
    full_context_budget: int,
    use_judge: bool,
    print_report: bool,
) -> list[HarnessResult]:
    examples = load_dataset(dataset_path)
    if limit > 0:
        examples = examples[:limit]

    model = GeminiModel(model=model_name)
    judge = GeminiJudge(model=judge_model_name) if use_judge else None
    results: list[HarnessResult] = []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for example in examples:
            for mode in modes:
                prompt, prompt_metadata, memory_trace = _build_mode_prompt(
                    example=example,
                    mode=mode,
                    short_context_tokens=short_context_tokens,
                    full_context_budget=full_context_budget,
                )
                response = model.generate(prompt=prompt)
                rule_score = score_response(example, response.text)
                judge_score = (
                    _judge_example(example=example, answer=response.text, judge=judge)
                    if judge is not None
                    else None
                )
                total_cost = estimate_cost_usd(
                    model_name,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                )
                if judge_score is not None and "_response" in judge_score:
                    judge_response = judge_score["_response"]
                    total_cost += estimate_cost_usd(
                        judge_model_name,
                        input_tokens=judge_response["input_tokens"],
                        output_tokens=judge_response["output_tokens"],
                    )

                result = HarnessResult(
                    example_id=example.id,
                    mode=mode,
                    model=model_name,
                    judge_model=judge_model_name if judge is not None else None,
                    prompt_tokens=response.input_tokens,
                    completion_tokens=response.output_tokens,
                    latency_ms=response.latency_ms,
                    cost_usd=round(total_cost, 6),
                    response_text=response.text,
                    rule_score=rule_score.to_dict(),
                    judge_score=judge_score,
                    memory_trace=memory_trace,
                    prompt_metadata=prompt_metadata,
                )
                handle.write(json.dumps(result.to_dict(), sort_keys=True) + "\n")
                results.append(result)

    if print_report:
        print(render_report(results))
    return results


def render_report(results: list[HarnessResult]) -> str:
    lines = [f"Generated at {datetime.now(timezone.utc).isoformat()}"]
    for mode in sorted({result.mode for result in results}):
        bucket = [result for result in results if result.mode == mode]
        passed = sum(1 for result in bucket if result.rule_score["passed"])
        total = len(bucket)
        hallucinations = sum(1 for result in bucket if result.rule_score["hallucination_flag"])
        avg_latency = sum(result.latency_ms for result in bucket) / total if total else 0.0
        avg_cost = sum(result.cost_usd for result in bucket) / total if total else 0.0
        avg_prompt_tokens = sum(result.prompt_tokens for result in bucket) / total if total else 0.0
        judge_scores = [
            result.judge_score["overall_score"]
            for result in bucket
            if result.judge_score is not None and "overall_score" in result.judge_score
        ]
        avg_judge = sum(judge_scores) / len(judge_scores) if judge_scores else 0.0
        lines.extend(
            [
                "",
                mode,
                f"  rule_pass_rate: {passed}/{total} ({round((passed / total) * 100, 1) if total else 0.0}%)",
                f"  hallucination_rate: {hallucinations}/{total} ({round((hallucinations / total) * 100, 1) if total else 0.0}%)",
                f"  avg_judge_score: {round(avg_judge, 4)}",
                f"  avg_latency_ms: {round(avg_latency, 2)}",
                f"  avg_prompt_tokens: {round(avg_prompt_tokens, 1)}",
                f"  avg_cost_usd: {round(avg_cost, 6)}",
            ]
        )
    return "\n".join(lines)


def _build_mode_prompt(
    *,
    example: BenchmarkExample,
    mode: str,
    short_context_tokens: int,
    full_context_budget: int,
) -> tuple[str, dict[str, Any], dict[str, Any] | None]:
    if mode == "short_context":
        turns = _trim_turns_to_budget(example.conversation, short_context_tokens)
        prompt = _build_transcript_prompt(example, turns)
        return prompt, {"mode": mode, "included_turns": len(turns), "context_budget": short_context_tokens}, None

    if mode == "full_context":
        turns = _trim_turns_to_budget(example.conversation, full_context_budget)
        prompt = _build_transcript_prompt(example, turns)
        return prompt, {"mode": mode, "included_turns": len(turns), "context_budget": full_context_budget}, None

    if mode == "memory_enabled":
        prompt, memory_trace = build_memory_prompt(example)
        return (
            prompt,
            {
                "mode": mode,
                "estimated_transcript_tokens": example.metadata.get("estimated_transcript_tokens"),
                "stored_counts": memory_trace.stored_counts,
            },
            memory_trace.to_dict(),
        )

    raise ValueError(f"Unsupported mode '{mode}'")


def _build_transcript_prompt(example: BenchmarkExample, turns: list[Turn]) -> str:
    transcript = "\n".join(f"{turn.role.upper()}: {turn.text}" for turn in turns)
    return (
        "You are evaluating long-context reasoning quality.\n"
        "Answer the task using only the transcript below.\n\n"
        f"TASK:\n{example.task.prompt}\n\n"
        "TRANSCRIPT:\n"
        f"{transcript}\n\n"
        "Write a concise answer that satisfies the task."
    )


def _trim_turns_to_budget(turns: list[Turn], budget: int) -> list[Turn]:
    selected: list[Turn] = []
    running = 0
    for turn in reversed(turns):
        turn_tokens = _estimate_text_tokens(turn.text) + 8
        if selected and running + turn_tokens > budget:
            break
        selected.append(turn)
        running += turn_tokens
    return list(reversed(selected))


def _judge_example(*, example: BenchmarkExample, answer: str, judge: GeminiJudge | None) -> dict[str, Any]:
    if judge is None:
        return {}
    prompt = (
        "You are grading an answer for a long-context memory benchmark.\n"
        "Return strict JSON with keys overall_score, groundedness_score, helpfulness_score, hallucination, reasoning.\n"
        "overall_score, groundedness_score, and helpfulness_score must be numbers between 0 and 1.\n\n"
        f"TASK:\n{example.task.prompt}\n\n"
        f"MUST_INCLUDE:\n{json.dumps(example.gold.must_include)}\n"
        f"MUST_NOT_INCLUDE:\n{json.dumps(example.gold.must_not_include)}\n\n"
        f"MODEL_ANSWER:\n{answer}\n"
    )
    return judge.judge(prompt=prompt)
