from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float


PRICING = {
    "gemini-2.5-flash": ModelPricing(input_per_million=0.30, output_per_million=2.50),
    "gemini-2.5-flash-lite": ModelPricing(input_per_million=0.10, output_per_million=0.40),
}


def estimate_cost_usd(model: str, *, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0
    input_cost = (input_tokens / 1_000_000) * pricing.input_per_million
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_million
    return round(input_cost + output_cost, 6)
