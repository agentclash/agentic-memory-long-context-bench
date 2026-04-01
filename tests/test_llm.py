import subprocess

from agentic_memory_long_context_bench.llm import ClaudeCodeModel, create_model


def test_create_model_supports_claude_code(monkeypatch):
    monkeypatch.setattr("agentic_memory_long_context_bench.llm.shutil_which", lambda _: "/usr/bin/claude")
    model = create_model(
        backend="claude_code",
        model="sonnet",
        max_retries=5,
        initial_backoff_seconds=2.0,
        backoff_multiplier=2.0,
        max_backoff_seconds=30.0,
        claude_binary="claude",
        claude_mcp_config=None,
        claude_max_budget_usd=None,
    )
    assert isinstance(model, ClaudeCodeModel)


def test_claude_code_model_parses_headless_json(monkeypatch):
    captured = {}

    def fake_run(command, capture_output, text, check):
        captured["command"] = command
        return subprocess.CompletedProcess(
            args=command,
            returncode=0,
            stdout='{"type":"result","subtype":"success","total_cost_usd":0.0123,"duration_ms":1234,"result":"hello world","usage":{"input_tokens":111,"output_tokens":22}}',
            stderr="",
        )

    monkeypatch.setattr("agentic_memory_long_context_bench.llm.shutil_which", lambda _: "/usr/bin/claude")
    monkeypatch.setattr("agentic_memory_long_context_bench.llm.subprocess.run", fake_run)

    model = ClaudeCodeModel(
        model="sonnet",
        claude_binary="claude",
        mcp_config="/tmp/test-mcp.json",
        max_budget_usd=1.5,
    )
    response = model.generate(prompt="test prompt")

    assert response.text == "hello world"
    assert response.input_tokens == 111
    assert response.output_tokens == 22
    assert response.latency_ms == 1234.0
    assert response.raw["provider"] == "claude_code"
    assert response.raw["total_cost_usd"] == 0.0123
    assert "--model" in captured["command"]
    assert "--mcp-config" in captured["command"]
    assert "--tools" in captured["command"]
