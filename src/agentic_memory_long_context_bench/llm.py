from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .generator import _estimate_text_tokens


class GenerativeModel(Protocol):
    def generate(self, *, prompt: str) -> "ModelResponse":
        ...


@dataclass(frozen=True)
class ModelResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeminiModel:
    def __init__(self, *, model: str):
        from google import genai

        self.model = model
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required to run the online harness. "
                "Export it before invoking run-long-context-harness."
            )
        self._client = genai.Client(api_key=api_key)

    def generate(self, *, prompt: str) -> ModelResponse:
        started = time.perf_counter()
        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        latency_ms = (time.perf_counter() - started) * 1000

        usage = getattr(response, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", None) or _estimate_text_tokens(prompt)
        output_tokens = getattr(usage, "candidates_token_count", None) or _estimate_text_tokens(response.text or "")

        return ModelResponse(
            text=(response.text or "").strip(),
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            latency_ms=round(latency_ms, 2),
            raw={"model": self.model},
        )


class GeminiJudge:
    def __init__(self, *, model: str):
        self._model = GeminiModel(model=model)

    def judge(self, *, prompt: str) -> dict[str, Any]:
        response = self._model.generate(prompt=prompt)
        payload = _extract_json_object(response.text)
        payload["_response"] = response.to_dict()
        return payload


def _extract_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return {
            "overall_score": 0.0,
            "groundedness_score": 0.0,
            "helpfulness_score": 0.0,
            "hallucination": True,
            "reasoning": "Judge response did not contain parseable JSON.",
        }
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return {
            "overall_score": 0.0,
            "groundedness_score": 0.0,
            "helpfulness_score": 0.0,
            "hallucination": True,
            "reasoning": "Judge JSON parse failed.",
        }
