from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .schema import BenchmarkExample


@dataclass(frozen=True)
class RuleScore:
    inclusion_pass_rate: float
    must_not_violation_count: int
    passed: bool
    hallucination_flag: bool
    matched_must_include: list[str]
    violated_must_not_include: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def score_response(example: BenchmarkExample, response_text: str) -> RuleScore:
    lowered = response_text.lower()
    must_include = example.gold.must_include
    must_not_include = example.gold.must_not_include

    matched = [needle for needle in must_include if needle.lower() in lowered]
    violated = [needle for needle in must_not_include if needle.lower() in lowered]
    inclusion_pass_rate = 1.0 if not must_include else len(matched) / len(must_include)
    hallucination_flag = bool(violated)
    passed = len(matched) == len(must_include) and not hallucination_flag

    return RuleScore(
        inclusion_pass_rate=round(inclusion_pass_rate, 4),
        must_not_violation_count=len(violated),
        passed=passed,
        hallucination_flag=hallucination_flag,
        matched_must_include=matched,
        violated_must_not_include=violated,
    )
