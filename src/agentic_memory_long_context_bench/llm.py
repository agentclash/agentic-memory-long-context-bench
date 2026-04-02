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


class TurnClassifier(Protocol):
    def classify(self, *, role: str, text: str) -> "ClassificationResult":
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


@dataclass(frozen=True)
class ClassificationResult:
    type: str
    field: str | None
    supersedes_description: str | None
    raw: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeminiModel:
    def __init__(
        self,
        *,
        model: str,
        max_retries: int = 5,
        initial_backoff_seconds: float = 2.0,
        backoff_multiplier: float = 2.0,
        max_backoff_seconds: float = 30.0,
    ):
        from google import genai

        self.model = model
        self.max_retries = max_retries
        self.initial_backoff_seconds = initial_backoff_seconds
        self.backoff_multiplier = backoff_multiplier
        self.max_backoff_seconds = max_backoff_seconds
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is required to run the online harness. "
                "Export it before invoking run-long-context-harness."
            )
        self._client = genai.Client(api_key=api_key)

    def generate(self, *, prompt: str) -> ModelResponse:
        attempt = 0
        delay = self.initial_backoff_seconds
        while True:
            attempt += 1
            started = time.perf_counter()
            try:
                response = self._client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )
                latency_ms = (time.perf_counter() - started) * 1000
                break
            except Exception as exc:
                if attempt > self.max_retries or not _is_retryable_exception(exc):
                    raise
                time.sleep(delay)
                delay = min(delay * self.backoff_multiplier, self.max_backoff_seconds)

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
    def __init__(self, *, model: str, **kwargs: Any):
        self._model = GeminiModel(model=model, **kwargs)

    def judge(self, *, prompt: str) -> dict[str, Any]:
        response = self._model.generate(prompt=prompt)
        payload = _extract_json_object(response.text)
        payload["_response"] = response.to_dict()
        return payload


class ClassifierLLM:
    def __init__(self, *, model: str, **kwargs: Any):
        self._model = GeminiModel(model=model, **kwargs)

    def classify(self, *, role: str, text: str) -> ClassificationResult:
        prompt = (
            "Classify a support-conversation message for memory ingestion.\n"
            "Choose exactly one type from: fact, correction, event, procedure, noise.\n"
            "Definitions:\n"
            '- fact: a durable statement about the user such as name, plan, timezone, device, or preference\n'
            '- correction: an update that supersedes an earlier durable fact\n'
            '- event: a temporal issue report, action the user took, or troubleshooting attempt\n'
            '- procedure: a reusable troubleshooting workflow or recipe\n'
            '- noise: acknowledgements, filler, or irrelevant chatter\n\n'
            "Also extract:\n"
            '- field: one of name, plan, timezone, device, preference, general, or null when not applicable\n'
            "- supersedes_description: brief description of the earlier fact being corrected, or null\n\n"
            f"ROLE: {role}\n"
            f"MESSAGE: {text}\n\n"
            'Return strict JSON with keys "type", "field", and "supersedes_description".'
        )
        response = self._model.generate(prompt=prompt)
        payload = _extract_classifier_payload(response.text)
        result_type = str(payload.get("type", "")).strip().lower()
        field = payload.get("field")
        supersedes_description = payload.get("supersedes_description")
        normalized_field = str(field).strip().lower() if isinstance(field, str) and field.strip() else None
        normalized_supersedes = (
            str(supersedes_description).strip()
            if isinstance(supersedes_description, str) and supersedes_description.strip()
            else None
        )
        if result_type not in {"fact", "correction", "event", "procedure", "noise"}:
            result_type = "noise"
        if normalized_field not in {"name", "plan", "timezone", "device", "preference", "general"}:
            normalized_field = None
        return ClassificationResult(
            type=result_type,
            field=normalized_field,
            supersedes_description=normalized_supersedes,
            raw={
                "model": self._model.model,
                "response": response.to_dict(),
                "payload": payload,
                "parse_error": bool(payload.get("_parse_error")),
            },
        )


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


def _extract_classifier_payload(text: str) -> dict[str, Any]:
    payload = _extract_json_object(text)
    if "type" in payload:
        return payload
    return {
        "type": "noise",
        "field": None,
        "supersedes_description": None,
        "_parse_error": True,
        "_raw_text": text[:200],
    }


def _is_retryable_exception(exc: Exception) -> bool:
    text = str(exc).lower()
    retry_markers = [
        "429",
        "resource_exhausted",
        "rate limit",
        "quota",
        "deadline exceeded",
        "timed out",
        "timeout",
        "temporar",
        "unavailable",
        "internal",
        "503",
        "500",
    ]
    return any(marker in text for marker in retry_markers)
