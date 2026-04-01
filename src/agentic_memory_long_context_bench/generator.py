from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

from .schema import BenchmarkExample, Gold, Task, Turn
from .vocab import (
    ATTEMPTED_STEPS,
    CHANNELS,
    DEVICES,
    DISTRACTOR_MESSAGES,
    FIRST_NAMES,
    ISSUES,
    PLANS,
    PROCEDURES,
    THEMES,
    TIMEZONES,
)

SCENARIO_TYPES = [
    "profile_recall",
    "troubleshooting_continuity",
    "contradiction_resolution",
    "procedure_reuse",
    "mixed_long_context",
]

DIFFICULTY_BY_INDEX = ["easy", "medium", "hard"]
DEFAULT_MIN_TOKENS = 0
DEFAULT_CONTEXT_TIER = "standard"


def generate_dataset(
    *,
    examples: int,
    seed: int,
    min_tokens: int = DEFAULT_MIN_TOKENS,
    context_tier: str = DEFAULT_CONTEXT_TIER,
) -> list[BenchmarkExample]:
    rng = random.Random(seed)
    rows: list[BenchmarkExample] = []
    for index in range(examples):
        scenario_type = SCENARIO_TYPES[index % len(SCENARIO_TYPES)]
        difficulty = DIFFICULTY_BY_INDEX[index % len(DIFFICULTY_BY_INDEX)]
        example_seed = seed + index
        row_rng = random.Random(example_seed)
        rows.append(
            _make_example(
                index=index,
                seed=example_seed,
                scenario_type=scenario_type,
                difficulty=difficulty,
                rng=row_rng,
                min_tokens=min_tokens,
                context_tier=context_tier,
            )
        )
    return rows


def _make_example(
    *,
    index: int,
    seed: int,
    scenario_type: str,
    difficulty: str,
    rng: random.Random,
    min_tokens: int,
    context_tier: str,
) -> BenchmarkExample:
    persona = {
        "name": rng.choice(FIRST_NAMES),
        "plan": rng.choice(PLANS),
        "timezone": rng.choice(TIMEZONES),
        "theme": rng.choice(THEMES),
        "device": rng.choice(DEVICES),
        "channel": rng.choice(CHANNELS),
    }
    issue = rng.choice(ISSUES)
    attempted = rng.sample(ATTEMPTED_STEPS, k=2)
    stale_plan = rng.choice([plan for plan in PLANS if plan != persona["plan"]])
    procedure_key = "login_loop" if "login loop" in issue else "webhook_failures"
    procedure = PROCEDURES[procedure_key]

    turns: list[Turn] = []
    fact_counter = 1

    def add_turn(role: str, kind: str, text: str, *, attach_fact: bool = False) -> str | None:
        nonlocal fact_counter
        fact_id = None
        if attach_fact:
            fact_id = f"fact_{fact_counter}"
            fact_counter += 1
        turns.append(Turn(role=role, turn_index=len(turns), kind=kind, text=text, fact_id=fact_id))
        return fact_id

    support_ids: list[str] = []
    stale_ids: list[str] = []

    support_ids.append(add_turn("user", "durable_fact", f"My name is {persona['name']}.", attach_fact=True) or "")
    add_turn("assistant", "ack", "Got it, I will remember that.")
    add_turn("user", "distractor", rng.choice(DISTRACTOR_MESSAGES))
    support_ids.append(add_turn("user", "durable_fact", f"I'm on the {stale_plan} plan.", attach_fact=True) or "")
    stale_ids.append(support_ids[-1])
    add_turn("assistant", "ack", "Thanks, noted.")
    add_turn("user", "distractor", rng.choice(DISTRACTOR_MESSAGES))
    support_ids.append(add_turn("user", "durable_fact", f"My timezone is {persona['timezone']}.", attach_fact=True) or "")
    add_turn("user", "durable_fact", f"I prefer the {persona['theme']} theme.", attach_fact=True)
    add_turn("user", "durable_fact", f"I use a {persona['device']}.", attach_fact=True)
    add_turn("user", "durable_fact", f"I prefer {persona['channel']} notifications.", attach_fact=True)
    add_turn("user", "event", f"My issue is {issue}.", attach_fact=True)
    support_ids.append(add_turn("user", "attempted_step", f"I already tried {attempted[0]} and {attempted[1]}.", attach_fact=True) or "")
    add_turn("assistant", "distractor", "I can help with that.")
    add_turn("user", "correction", f"Correction: I'm actually on the {persona['plan']} plan.", attach_fact=True)
    support_ids.append(turns[-1].fact_id or "")

    if difficulty in {"medium", "hard"}:
        for _ in range(8):
            add_turn("user", "distractor", rng.choice(DISTRACTOR_MESSAGES))
            add_turn("assistant", "distractor", "Thanks for the extra context.")

    proc_id = add_turn(
        "assistant",
        "procedure_outcome",
        f"A previous successful procedure for this class of issue was: {procedure['content']}",
        attach_fact=True,
    )
    support_ids.append(proc_id or "")

    if min_tokens > 0:
        block_index = 0
        while _estimate_conversation_tokens(turns) < min_tokens:
            add_turn(
                "user",
                "distractor_block",
                _make_distractor_block(
                    rng=rng,
                    persona=persona,
                    issue=issue,
                    attempted=attempted,
                    block_index=block_index,
                ),
            )
            add_turn(
                "assistant",
                "distractor_block",
                _make_ack_block(issue=issue, block_index=block_index),
            )
            block_index += 1

    task_prompt, must_include, must_not_include, required = _build_task(
        scenario_type=scenario_type,
        persona=persona,
        issue=issue,
        attempted=attempted,
        procedure=procedure,
        stale_plan=stale_plan,
    )

    estimated_transcript_tokens = _estimate_conversation_tokens(turns)
    supporting_fact_tokens = _estimate_text_tokens(" ".join(
        turn.text for turn in turns if turn.fact_id in {fact_id for fact_id in support_ids if fact_id}
    ))
    distractor_tokens = _estimate_text_tokens(" ".join(turn.text for turn in turns if "distractor" in turn.kind))

    return BenchmarkExample(
        id=f"{scenario_type}_{difficulty}_{index:04d}",
        seed=seed,
        scenario_type=scenario_type,
        difficulty=difficulty,
        conversation=turns,
        task=Task(prompt=task_prompt, requires=required),
        gold=Gold(
            must_include=must_include,
            must_not_include=must_not_include,
            supporting_fact_ids=[fact_id for fact_id in support_ids if fact_id],
            stale_fact_ids=stale_ids,
        ),
        metadata={
            "target_length_turns": len(turns),
            "estimated_transcript_tokens": estimated_transcript_tokens,
            "supporting_fact_tokens": supporting_fact_tokens,
            "distractor_tokens": distractor_tokens,
            "context_tier": context_tier,
            "target_min_tokens": min_tokens,
            "persona": persona,
            "issue": issue,
            "attempted_steps": attempted,
            "procedure_key": procedure_key,
        },
    )


def _build_task(
    *,
    scenario_type: str,
    persona: dict[str, str],
    issue: str,
    attempted: list[str],
    procedure: dict[str, object],
    stale_plan: str,
) -> tuple[str, list[str], list[str], list[str]]:
    if scenario_type == "profile_recall":
        return (
            "Summarize the user's profile details that matter for support routing.",
            [persona["name"].lower(), persona["plan"], persona["timezone"]],
            [stale_plan],
            ["name", "current_plan", "timezone"],
        )
    if scenario_type == "troubleshooting_continuity":
        return (
            "Summarize the current issue and propose the next best action without repeating steps already attempted.",
            [attempted[0].split()[0], attempted[1].split()[0], issue.split()[0]],
            [],
            ["current_issue", "attempted_steps"],
        )
    if scenario_type == "contradiction_resolution":
        return (
            "Answer what plan the user is currently on, ignoring outdated details.",
            [persona["plan"]],
            [stale_plan],
            ["current_plan", "stale_plan"],
        )
    if scenario_type == "procedure_reuse":
        return (
            "Recommend the next best troubleshooting procedure for the user's issue.",
            [token.lower() for token in procedure["must_include"]],  # type: ignore[index]
            [],
            ["current_issue", "procedure"],
        )
    return (
        "Summarize the user's current support context and propose the next best action.",
        [persona["plan"], persona["timezone"], issue.split()[0], attempted[0].split()[0]],
        [stale_plan],
        ["current_plan", "timezone", "current_issue", "attempted_steps", "procedure"],
    )


def _estimate_text_tokens(text: str) -> int:
    coarse = math.ceil(len(text) / 4)
    lexical = len(re.findall(r"\w+|[^\w\s]", text))
    return max(coarse, lexical)


def _estimate_conversation_tokens(turns: list[Turn]) -> int:
    return sum(_estimate_text_tokens(turn.text) + 8 for turn in turns)


def _make_distractor_block(
    *,
    rng: random.Random,
    persona: dict[str, str],
    issue: str,
    attempted: list[str],
    block_index: int,
) -> str:
    fragments = [
        f"Block {block_index}: this paragraph contains non-critical support chatter about dashboards, project status, planning notes, and release coordination.",
        f"The user repeats incidental observations about the {persona['theme']} theme, their {persona['device']}, and unrelated planning for the {persona['channel']} channel.",
        f"They also mention background notes about {issue}, but without adding new authoritative facts beyond what was already established earlier in the conversation.",
        f"Previously attempted steps like {attempted[0]} and {attempted[1]} are mentioned indirectly among many irrelevant details, meeting notes, and duplicate summaries.",
        "Additional filler includes repeated references to design reviews, internal docs, analytics discussions, migration notes, environment cleanups, and stakeholder updates.",
    ]
    sentence = " ".join(fragments)
    return " ".join(sentence for _ in range(80))


def _make_ack_block(*, issue: str, block_index: int) -> str:
    sentence = (
        f"Ack block {block_index}: the assistant acknowledges the background detail about {issue} "
        "and repeats that it is collecting context, but it does not add any new ground-truth facts."
    )
    return " ".join(sentence for _ in range(40))


def write_dataset(rows: list[BenchmarkExample], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row.to_dict(), sort_keys=True) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a deterministic long-context benchmark dataset.")
    parser.add_argument("--examples", type=int, default=25, help="Number of examples to generate.")
    parser.add_argument("--seed", type=int, default=7, help="Base RNG seed.")
    parser.add_argument(
        "--min-tokens",
        type=int,
        default=DEFAULT_MIN_TOKENS,
        help="Minimum estimated transcript tokens per example.",
    )
    parser.add_argument(
        "--context-tier",
        type=str,
        default=DEFAULT_CONTEXT_TIER,
        help="Label stored in metadata to describe the context-budget tier.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/sample_v1.jsonl"),
        help="Output JSONL file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = generate_dataset(
        examples=args.examples,
        seed=args.seed,
        min_tokens=args.min_tokens,
        context_tier=args.context_tier,
    )
    write_dataset(rows, args.output)
    print(f"Wrote {len(rows)} examples to {args.output}")


if __name__ == "__main__":
    main()
