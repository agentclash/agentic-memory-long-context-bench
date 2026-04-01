from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Turn:
    role: str
    turn_index: int
    kind: str
    text: str
    fact_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Task:
    prompt: str
    requires: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Gold:
    must_include: list[str]
    must_not_include: list[str]
    supporting_fact_ids: list[str]
    stale_fact_ids: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkExample:
    id: str
    seed: int
    scenario_type: str
    difficulty: str
    conversation: list[Turn]
    task: Task
    gold: Gold
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "seed": self.seed,
            "scenario_type": self.scenario_type,
            "difficulty": self.difficulty,
            "conversation": [turn.to_dict() for turn in self.conversation],
            "task": self.task.to_dict(),
            "gold": self.gold.to_dict(),
            "metadata": self.metadata,
        }
