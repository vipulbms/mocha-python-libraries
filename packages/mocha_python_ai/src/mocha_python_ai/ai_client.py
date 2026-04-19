"""
AIClient — single abstraction over all LLM providers (Groq/OpenAI-compat and Ollama).

Story: S20.2.1 (Sprint S3)
No agent module should import groq, openai, or ollama directly after this
story is complete — all LLM interaction goes through AIClient.chat_with_tools().

Usage::

    from mocha_python_ai import AIClient, ModelConfig

    config = ModelConfig(
        provider="openai_compat",
        primary_model="qwen/qwen3-32b",
        fallback_model="meta-llama/llama-3.3-70b-versatile",
        api_key=os.environ["GROQ_API_KEY"],
        base_url="https://api.groq.com/openai/v1",
    )
    client = AIClient(config)
    result = client.chat_with_tools(messages, tools, persona_params={"temperature": 0.1})
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

# Exponential backoff delays (seconds) for each retry attempt.
# Attempt 0 = primary model (no sleep before).
# Attempt 1 = primary model again after 1 s.
# Attempt 2 = fallback model after 2 s.
_RETRY_DELAYS = [0, 1, 2]


@dataclass
class ModelConfig:
    """Configuration for a primary + fallback LLM pair."""

    provider: str = "openai_compat"   # "openai_compat" | "ollama"
    primary_model: str = ""
    fallback_model: str = ""
    api_key: str = ""
    base_url: str = ""
    disable_thinking: bool = True     # For qwen3-class models on Groq
    request_timeout: int = 30         # Per-call timeout in seconds
    log_dir: str = "/logs"
    component: str = "AIE"

    # Optional IntegrationLogger instance — injected externally or created lazily
    integration_logger: Any = field(default=None, repr=False)


class AIClient:
    """
    Provider-agnostic LLM client with retry, fallback, and thinking-mode guards.

    chat_with_tools() is the ONLY public method callers should use.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._cfg = config
        self._client = None          # lazy init
        self._intlog = config.integration_logger  # may be None

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        persona_params: dict | None = None,
    ) -> dict:
        """
        Send messages to the LLM with tool definitions.

        Returns::

            {
                "tool_calls":        list[{"name": str, "args": dict}],
                "raw_output":        str,
                "prompt_tokens":     int | None,
                "completion_tokens": int | None,
                "fallback":          bool,
            }

        Retry logic: 3 attempts with exponential backoff.
        Attempt 3 automatically uses the fallback model.
        """
        params = persona_params or {}
        temperature = params.get("temperature", 0.1)
        max_tokens  = params.get("max_tokens", 1024)

        last_err: Exception | None = None
        for attempt_idx, delay in enumerate(_RETRY_DELAYS):
            if delay:
                time.sleep(delay)
            # Use fallback model on the last attempt
            using_fallback = attempt_idx == len(_RETRY_DELAYS) - 1
            model = self._cfg.fallback_model if using_fallback else self._cfg.primary_model
            try:
                result = self._call(model, messages, tools, temperature, max_tokens)
                result["fallback"] = using_fallback
                return result
            except Exception as exc:
                last_err = exc
                logger.warning(
                    "[AIClient] Attempt %d/%d failed (model=%s): %s",
                    attempt_idx + 1, len(_RETRY_DELAYS), model, exc,
                )
        # All attempts exhausted
        raise RuntimeError(
            f"AIClient: all {len(_RETRY_DELAYS)} attempts failed. Last error: {last_err}"
        ) from last_err

    # ─────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────

    def _call(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        start = time.time()
        try:
            if self._cfg.provider == "openai_compat":
                result = self._call_openai_compat(model, messages, tools, temperature, max_tokens)
            else:
                result = self._call_ollama(model, messages, tools, temperature)
        except Exception as exc:
            duration_ms = (time.time() - start) * 1000
            self._log_call(model, "chat_with_tools", status="error",
                           duration_ms=duration_ms, error=str(exc))
            raise
        duration_ms = (time.time() - start) * 1000
        self._log_call(
            model, "chat_with_tools",
            status="ok",
            duration_ms=duration_ms,
            prompt_tokens=result.get("prompt_tokens"),
            completion_tokens=result.get("completion_tokens"),
        )
        return result

    def _call_openai_compat(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        client = self._get_openai_client()
        kwargs: dict[str, Any] = dict(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="required",
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=self._cfg.request_timeout,
        )
        # qwen3 thinking-mode guard (AC4): reasoning_effort=none + reasoning_format=hidden
        if self._cfg.disable_thinking and "qwen3" in model.lower():
            kwargs["extra_body"] = {
                "reasoning_effort": "none",
                "reasoning_format": "hidden",
            }

        response = client.chat.completions.create(**kwargs)
        choice = response.choices[0]
        msg = choice.message

        raw_output = msg.content or ""
        # AC5 — strip <think>…</think> blocks (defence-in-depth)
        raw_output = _THINK_RE.sub("", raw_output).strip()

        tool_calls = self._normalise_openai_tool_calls(msg.tool_calls or [])

        return {
            "tool_calls":        tool_calls,
            "raw_output":        raw_output,
            "prompt_tokens":     getattr(response.usage, "prompt_tokens", None),
            "completion_tokens": getattr(response.usage, "completion_tokens", None),
        }

    def _call_ollama(
        self,
        model: str,
        messages: list[dict],
        tools: list[dict],
        temperature: float,
    ) -> dict:
        try:
            import ollama  # optional dep
        except ImportError as exc:
            raise ImportError(
                "ollama package is not installed. Install it with: pip install ollama"
            ) from exc

        client = ollama.Client()
        response = client.chat(
            model=model,
            messages=messages,
            tools=tools,
            options={"temperature": temperature},
        )
        msg = response.message
        raw_output = msg.content or ""
        raw_output = _THINK_RE.sub("", raw_output).strip()

        tool_calls = []
        for tc in (msg.tool_calls or []):
            tool_calls.append({
                "name": tc.function.name,
                "args": dict(tc.function.arguments) if tc.function.arguments else {},
            })

        return {
            "tool_calls":        tool_calls,
            "raw_output":        raw_output,
            "prompt_tokens":     getattr(response, "prompt_eval_count", None),
            "completion_tokens": getattr(response, "eval_count", None),
        }

    def _normalise_openai_tool_calls(self, raw_calls: list) -> list[dict]:
        """Convert OpenAI SDK ToolCall objects to plain dicts."""
        result = []
        for tc in raw_calls:
            args = tc.function.arguments
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except Exception:
                    args = {}
            result.append({"name": tc.function.name, "args": dict(args)})
        return result

    def _get_openai_client(self):
        if self._client is None:
            from openai import OpenAI  # AC8-compatible: only AIClient imports openai
            self._client = OpenAI(
                api_key=self._cfg.api_key,
                base_url=self._cfg.base_url or None,
            )
        return self._client

    def _log_call(
        self,
        model: str,
        operation: str,
        status: str = "ok",
        duration_ms: float | None = None,
        error: str = "",
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        """Log via IntegrationLogger if one is configured; else fallback to stdlib."""
        if self._intlog is not None:
            try:
                self._intlog.write(
                    service=f"LLM:{model}",
                    operation=operation,
                    request_summary=f"p_tokens={prompt_tokens} c_tokens={completion_tokens}",
                    response_status=None,
                    duration_ms=duration_ms,
                    status=status,
                    error_detail=error,
                )
            except Exception as log_err:
                logger.debug("[AIClient] IntegrationLogger write failed: %s", log_err)
        else:
            logger.debug(
                "[AIClient] model=%s op=%s status=%s duration=%.0fms "
                "prompt_tokens=%s completion_tokens=%s error=%s",
                model, operation, status,
                duration_ms or 0,
                prompt_tokens, completion_tokens, error,
            )
