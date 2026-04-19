"""
Tests for S20.2.1 — AIClient (mocha-python-ai).

Story: S20.2.1 | Sprint: S3 | Epic: E20 — Library Extraction

Covers:
  AC1: ModelConfig dataclass has required fields
  AC2: AIClient has chat_with_tools() as its only public method
  AC3: 3 retry attempts made; attempt 3 uses fallback_model
  AC7: primary model times out on first 2 attempts → fallback model used → fallback=True
  AC8: openai is imported lazily (no top-level import in ai_client.py module)
  AC9: result dict always contains tool_calls, raw_output, prompt_tokens,
       completion_tokens, fallback keys
"""

import inspect
from dataclasses import fields
from unittest.mock import MagicMock, patch, call

import pytest

from mocha_python_ai import AIClient, ModelConfig
from mocha_python_ai.ai_client import _RETRY_DELAYS


# ── Helpers ───────────────────────────────────────────────────────────────────

def _config(primary: str = "qwen3-32b", fallback: str = "llama-3.3-70b") -> ModelConfig:
    return ModelConfig(
        provider="openai_compat",
        primary_model=primary,
        fallback_model=fallback,
        api_key="test-key",
        base_url="https://api.groq.com/openai/v1",
        request_timeout=5,
    )


def _make_response(tool_name: str = "hold", args: dict = None) -> MagicMock:
    """Build a mock OpenAI chat completion response with a single tool call."""
    tc = MagicMock()
    tc.function.name = tool_name
    tc.function.arguments = args or {}

    msg = MagicMock()
    msg.content = ""
    msg.tool_calls = [tc]

    choice = MagicMock()
    choice.message = msg

    usage = MagicMock()
    usage.prompt_tokens = 100
    usage.completion_tokens = 50

    resp = MagicMock()
    resp.choices = [choice]
    resp.usage = usage
    return resp


# ── AC1: ModelConfig fields ───────────────────────────────────────────────────

class TestModelConfig:

    def test_required_fields_present(self):
        """AC1: ModelConfig has all required fields."""
        field_names = {f.name for f in fields(ModelConfig)}
        for required in ("provider", "primary_model", "fallback_model",
                         "api_key", "base_url", "disable_thinking",
                         "request_timeout", "log_dir", "component"):
            assert required in field_names, f"ModelConfig missing field: {required}"

    def test_defaults(self):
        """AC1: Optional fields have sensible defaults."""
        cfg = ModelConfig(
            provider="openai_compat",
            primary_model="model-a",
            fallback_model="model-b",
            api_key="key",
            base_url="http://localhost",
        )
        assert cfg.disable_thinking is True
        assert cfg.request_timeout == 30
        assert cfg.log_dir == "/logs"


# ── AC2: public API surface ───────────────────────────────────────────────────

class TestAIClientAPI:

    def test_chat_with_tools_is_callable(self):
        """AC2: chat_with_tools() method exists."""
        client = AIClient(_config())
        assert callable(client.chat_with_tools)

    def test_private_methods_not_exposed_as_main_interface(self):
        """AC2 spirit: internal call helpers are private (_call, _call_openai_compat)."""
        client = AIClient(_config())
        # _call is private; callers should use chat_with_tools
        assert not hasattr(client, "call"), "Should not have a public 'call' method"


# ── AC8: lazy openai import ───────────────────────────────────────────────────

class TestLazyImport:

    def test_no_top_level_openai_import(self):
        """AC8: openai is NOT imported at module top-level in ai_client.py."""
        import mocha_python_ai.ai_client as mod
        src = inspect.getsource(mod)
        # The import must be inside a function, not at the top level
        lines = src.splitlines()
        top_level_import = any(
            line.startswith("import openai") or line.startswith("from openai")
            for line in lines
        )
        assert not top_level_import, (
            "openai is imported at the top level of ai_client.py (violates AC8)"
        )


# ── AC3 + AC7: retry and fallback ────────────────────────────────────────────

class TestRetryAndFallback:

    def _make_client_with_mock(self):
        client = AIClient(_config())
        return client

    def test_success_on_first_attempt_returns_fallback_false(self):
        """AC3: Successful first call → fallback=False in result."""
        client = self._make_client_with_mock()
        mock_resp = _make_response("propose_buy", {"pair": "ETH/USD", "amount_usd": 100.0})

        with patch.object(client, "_get_openai_client") as mock_get:
            mock_oc = MagicMock()
            mock_oc.chat.completions.create.return_value = mock_resp
            mock_get.return_value = mock_oc

            result = client.chat_with_tools(
                messages=[{"role": "user", "content": "trade"}],
                tools=[{"type": "function", "function": {"name": "propose_buy"}}],
            )

        assert result["fallback"] is False

    def test_primary_timeout_uses_fallback_on_third_attempt(self):
        """
        AC7: first 2 attempts raise timeout; 3rd attempt uses fallback_model.
        Result has fallback=True.
        """
        client = self._make_client_with_mock()
        mock_resp = _make_response("hold", {})
        call_count = [0]

        def _side_effect(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] < len(_RETRY_DELAYS):
                raise TimeoutError("Request timed out")
            return mock_resp

        with patch.object(client, "_get_openai_client") as mock_get, \
             patch("time.sleep"):  # Speed up retry delays in test
            mock_oc = MagicMock()
            mock_oc.chat.completions.create.side_effect = _side_effect
            mock_get.return_value = mock_oc

            result = client.chat_with_tools(
                messages=[{"role": "user", "content": "trade"}],
                tools=[],
            )

        assert result["fallback"] is True, (
            "Expected fallback=True when primary model timed out on all but last attempt"
        )

    def test_all_attempts_fail_raises_runtime_error(self):
        """AC3: If all 3 attempts fail, RuntimeError is raised."""
        client = self._make_client_with_mock()

        with patch.object(client, "_get_openai_client") as mock_get, \
             patch("time.sleep"):
            mock_oc = MagicMock()
            mock_oc.chat.completions.create.side_effect = RuntimeError("API error")
            mock_get.return_value = mock_oc

            with pytest.raises(RuntimeError, match="all .* attempts failed"):
                client.chat_with_tools([], [])


# ── AC9: result dict structure ────────────────────────────────────────────────

class TestResultStructure:

    def test_result_has_all_required_keys(self):
        """AC9: Result dict always contains the 5 required keys."""
        client = AIClient(_config())
        mock_resp = _make_response("hold", {})

        with patch.object(client, "_get_openai_client") as mock_get:
            mock_oc = MagicMock()
            mock_oc.chat.completions.create.return_value = mock_resp
            mock_get.return_value = mock_oc

            result = client.chat_with_tools([], [])

        for key in ("tool_calls", "raw_output", "prompt_tokens", "completion_tokens", "fallback"):
            assert key in result, f"Key '{key}' missing from result dict"

    def test_tool_calls_is_list(self):
        """AC9: tool_calls is always a list (possibly empty)."""
        client = AIClient(_config())
        mock_resp = _make_response("hold", {"pair": "BTC/USD"})

        with patch.object(client, "_get_openai_client") as mock_get:
            mock_oc = MagicMock()
            mock_oc.chat.completions.create.return_value = mock_resp
            mock_get.return_value = mock_oc

            result = client.chat_with_tools([], [])

        assert isinstance(result["tool_calls"], list)

    def test_think_tags_stripped_from_raw_output(self):
        """AC5 (ai_client): <think>…</think> blocks stripped from raw_output."""
        client = AIClient(_config())

        mock_msg = MagicMock()
        mock_msg.content = "<think>internal reasoning</think>Final answer."
        mock_msg.tool_calls = []

        mock_choice = MagicMock()
        mock_choice.message = mock_msg

        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 50
        mock_usage.completion_tokens = 20

        mock_resp = MagicMock()
        mock_resp.choices = [mock_choice]
        mock_resp.usage = mock_usage

        with patch.object(client, "_get_openai_client") as mock_get:
            mock_oc = MagicMock()
            mock_oc.chat.completions.create.return_value = mock_resp
            mock_get.return_value = mock_oc

            result = client.chat_with_tools([], [])

        assert "<think>" not in result["raw_output"]
        assert "Final answer." in result["raw_output"]
