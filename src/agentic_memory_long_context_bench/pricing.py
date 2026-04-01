from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_per_million: float
    output_per_million: float
    input_per_million_over_200k: float | None = None
    output_per_million_over_200k: float | None = None


PRICING = {
    "gemini-2.5-flash": ModelPricing(input_per_million=0.30, output_per_million=2.50),
    "gemini-2.5-flash-lite": ModelPricing(input_per_million=0.10, output_per_million=0.40),
    "gemini-2.5-pro": ModelPricing(
        input_per_million=1.25,
        output_per_million=10.00,
        input_per_million_over_200k=2.50,
        output_per_million_over_200k=15.00,
    ),
}


def estimate_cost_usd(model: str, *, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING.get(model)
    if pricing is None:
        return 0.0
    input_rate = pricing.input_per_million
    output_rate = pricing.output_per_million
    if input_tokens > 200_000:
        input_rate = pricing.input_per_million_over_200k or input_rate
        output_rate = pricing.output_per_million_over_200k or output_rate
    input_cost = (input_tokens / 1_000_000) * input_rate
    output_cost = (output_tokens / 1_000_000) * output_rate
    return round(input_cost + output_cost, 6)
