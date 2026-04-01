from __future__ import annotations

import json
from pathlib import Path

from .schema import BenchmarkExample, Gold, Task, Turn


def load_dataset(path: str | Path) -> list[BenchmarkExample]:
    rows: list[BenchmarkExample] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            payload = json.loads(line)
            rows.append(
                BenchmarkExample(
                    id=payload["id"],
                    seed=payload["seed"],
                    scenario_type=payload["scenario_type"],
                    difficulty=payload["difficulty"],
                    conversation=[Turn(**turn) for turn in payload["conversation"]],
                    task=Task(**payload["task"]),
                    gold=Gold(**payload["gold"]),
                    metadata=payload["metadata"],
                )
            )
    return rows
