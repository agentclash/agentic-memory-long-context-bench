from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from .generator import _estimate_text_tokens


class GenerativeModel(Protocol):
    def generate(self, *, prompt: str) -> "ModelResponse":
        ...


class JudgeModel(Protocol):
    def judge(self, *, prompt: str) -> dict[str, Any]:
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
                "GEMINI_API_KEY is required to run the online Gemini harness. "
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
            raw={"provider": "gemini", "model": self.model},
        )


class GeminiJudge:
    def __init__(self, *, model: str, **kwargs: Any):
        self._model = GeminiModel(model=model, **kwargs)

    def judge(self, *, prompt: str) -> dict[str, Any]:
        response = self._model.generate(prompt=prompt)
        payload = _extract_json_object(response.text)
        payload["_response"] = response.to_dict()
        return payload


class ClaudeCodeModel:
    def __init__(
        self,
        *,
        model: str,
        claude_binary: str = "claude",
        mcp_config: str | None = None,
        max_budget_usd: float | None = None,
        extra_args: list[str] | None = None,
    ):
        self.model = model
        self.claude_binary = claude_binary
        self.mcp_config = mcp_config
        self.max_budget_usd = max_budget_usd
        self.extra_args = extra_args or []
        if not shutil_which(self.claude_binary):
            raise ValueError(
                f"Claude Code binary '{self.claude_binary}' was not found on PATH. "
                "Install Claude Code first or pass --claude-binary."
            )

    def generate(self, *, prompt: str) -> ModelResponse:
        command = [
            self.claude_binary,
            "-p",
            prompt,
            "--model",
            self.model,
            "--output-format",
            "json",
            "--permission-mode",
            "default",
            "--tools",
            "",
            "--disable-slash-commands",
            "--strict-mcp-config",
        ]
        if self.mcp_config:
            command.extend(["--mcp-config", self.mcp_config])
        if self.max_budget_usd is not None:
            command.extend(["--max-budget-usd", str(self.max_budget_usd)])
        command.extend(self.extra_args)

        started = time.perf_counter()
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=False,
        )
        fallback_latency_ms = round((time.perf_counter() - started) * 1000, 2)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Claude Code headless run failed with exit code {completed.returncode}: "
                f"{completed.stderr.strip() or completed.stdout.strip()}"
            )

        payload = _extract_json_object(completed.stdout)
        result_text = (payload.get("result") or payload.get("text") or "").strip()
        usage = payload.get("usage") or {}
        input_tokens = usage.get("input_tokens") or payload.get("input_tokens") or _estimate_text_tokens(prompt)
        output_tokens = usage.get("output_tokens") or payload.get("output_tokens") or _estimate_text_tokens(result_text)
        latency_ms = payload.get("duration_ms") or fallback_latency_ms
        raw = {
            "provider": "claude_code",
            "model": self.model,
            "cli_payload": payload,
        }
        if "total_cost_usd" in payload:
            raw["total_cost_usd"] = payload["total_cost_usd"]

        return ModelResponse(
            text=result_text,
            input_tokens=int(input_tokens),
            output_tokens=int(output_tokens),
            latency_ms=round(float(latency_ms), 2),
            raw=raw,
        )


class ClaudeCodeJudge:
    def __init__(self, *, model: str, **kwargs: Any):
        self._model = ClaudeCodeModel(model=model, **kwargs)

    def judge(self, *, prompt: str) -> dict[str, Any]:
        response = self._model.generate(prompt=prompt)
        payload = _extract_json_object(response.text)
        payload["_response"] = response.to_dict()
        if "total_cost_usd" in response.raw:
            payload["_response"]["raw"]["total_cost_usd"] = response.raw["total_cost_usd"]
        return payload


def create_model(
    *,
    backend: str,
    model: str,
    max_retries: int,
    initial_backoff_seconds: float,
    backoff_multiplier: float,
    max_backoff_seconds: float,
    claude_binary: str,
    claude_mcp_config: str | None,
    claude_max_budget_usd: float | None,
) -> GenerativeModel:
    if backend == "gemini":
        return GeminiModel(
            model=model,
            max_retries=max_retries,
            initial_backoff_seconds=initial_backoff_seconds,
            backoff_multiplier=backoff_multiplier,
            max_backoff_seconds=max_backoff_seconds,
        )
    if backend == "claude_code":
        return ClaudeCodeModel(
            model=model,
            claude_binary=claude_binary,
            mcp_config=claude_mcp_config,
            max_budget_usd=claude_max_budget_usd,
        )
    raise ValueError(f"Unsupported backend '{backend}'")


def create_judge(
    *,
    backend: str,
    model: str,
    max_retries: int,
    initial_backoff_seconds: float,
    backoff_multiplier: float,
    max_backoff_seconds: float,
    claude_binary: str,
    claude_mcp_config: str | None,
    claude_max_budget_usd: float | None,
) -> JudgeModel:
    if backend == "gemini":
        return GeminiJudge(
            model=model,
            max_retries=max_retries,
            initial_backoff_seconds=initial_backoff_seconds,
            backoff_multiplier=backoff_multiplier,
            max_backoff_seconds=max_backoff_seconds,
        )
    if backend == "claude_code":
        return ClaudeCodeJudge(
            model=model,
            claude_binary=claude_binary,
            mcp_config=claude_mcp_config,
            max_budget_usd=claude_max_budget_usd,
        )
    raise ValueError(f"Unsupported backend '{backend}'")


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


def shutil_which(binary: str) -> str | None:
    try:
        import shutil

        return shutil.which(binary)
    except Exception:
        return None
