"""
Author: Tester
Story: S20.1.2
Sprint: S2
Description: Tests for IntegrationLogger and @log_integration decorator.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time

import pytest

from mocha_python_logging.integration_logger import (
    IntegrationLogger,
    _redact,
    cycle_id_var,
    log_integration,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_logger(tmp_path) -> tuple[IntegrationLogger, str]:
    log_dir = str(tmp_path / "logs")
    lg = IntegrationLogger(log_dir=log_dir, component="TEST")
    return lg, str(tmp_path / "logs" / "integration.log")


def _read_records(log_path: str) -> list[dict]:
    if not os.path.exists(log_path):
        return []
    with open(log_path) as f:
        return [json.loads(line) for line in f if line.strip()]


# ---------------------------------------------------------------------------
# AC1: Package importable, IntegrationLogger and log_integration present
# ---------------------------------------------------------------------------

class TestImports:
    def test_integration_logger_importable(self):
        from mocha_python_logging import IntegrationLogger as IL  # noqa: F401
        assert IL is not None

    def test_log_integration_importable(self):
        from mocha_python_logging import log_integration as li  # noqa: F401
        assert li is not None


# ---------------------------------------------------------------------------
# AC2 / AC6: Record fields and direct write()
# ---------------------------------------------------------------------------

class TestDirectWrite:
    def test_write_creates_log_file(self, tmp_path):
        lg, log_path = _make_logger(tmp_path)
        lg.write(service="TEST_SVC", operation="ping", duration_ms=12.5)
        records = _read_records(log_path)
        assert len(records) == 1

    def test_required_fields_present(self, tmp_path):
        lg, log_path = _make_logger(tmp_path)
        lg.write(
            service="GROQ",
            operation="chat_completions",
            request_summary='{"model": "qwen3"}',
            response_status=200,
            duration_ms=310.7,
            status="ok",
        )
        rec = _read_records(log_path)[0]
        for field in ("timestamp", "component", "service", "operation",
                      "request_summary", "response_status", "duration_ms",
                      "status", "error_detail", "cycle_id"):
            assert field in rec, f"Missing field: {field}"

    def test_component_set(self, tmp_path):
        lg, log_path = _make_logger(tmp_path)
        lg.write(service="KRAKEN", operation="get_balance")
        rec = _read_records(log_path)[0]
        assert rec["component"] == "TEST"

    def test_cycle_id_from_contextvar(self, tmp_path):
        lg, log_path = _make_logger(tmp_path)
        token = cycle_id_var.set("CYCLE-99")
        try:
            lg.write(service="TELEGRAM", operation="send_message")
        finally:
            cycle_id_var.reset(token)
        rec = _read_records(log_path)[0]
        assert rec["cycle_id"] == "CYCLE-99"

    def test_cycle_id_explicit_overrides_contextvar(self, tmp_path):
        lg, log_path = _make_logger(tmp_path)
        token = cycle_id_var.set("CYCLE-99")
        try:
            lg.write(service="COINGLASS", operation="mvrv", cycle_id="EXPLICIT-1")
        finally:
            cycle_id_var.reset(token)
        rec = _read_records(log_path)[0]
        assert rec["cycle_id"] == "EXPLICIT-1"


# ---------------------------------------------------------------------------
# AC4: Redaction
# ---------------------------------------------------------------------------

class TestRedaction:
    def test_redact_api_key_in_dict(self):
        result = _redact({"api_key": "supersecret", "url": "https://example.com"})
        assert result["api_key"] == "[REDACTED]"
        assert result["url"] == "https://example.com"

    def test_redact_authorization_header(self):
        result = _redact({"Authorization": "Bearer tok123", "Content-Type": "application/json"})
        assert result["Authorization"] == "[REDACTED]"

    def test_redact_nested(self):
        result = _redact({"headers": {"api_key": "abc", "accept": "json"}})
        assert result["headers"]["api_key"] == "[REDACTED]"
        assert result["headers"]["accept"] == "json"

    def test_redact_in_list(self):
        result = _redact([{"token": "xyz"}, {"name": "foo"}])
        assert result[0]["token"] == "[REDACTED]"
        assert result[1]["name"] == "foo"

    def test_non_sensitive_key_not_redacted(self):
        result = _redact({"model": "qwen3-32b", "temperature": 0.7})
        assert result["model"] == "qwen3-32b"


# ---------------------------------------------------------------------------
# AC5 + sync decorator
# ---------------------------------------------------------------------------

class TestSyncDecorator:
    def test_sync_function_logged(self, tmp_path):
        lg, log_path = _make_logger(tmp_path)

        @lg.decorator("KRAKEN", "get_balance")
        def call_kraken():
            return {"balance": 1000}

        call_kraken()
        records = _read_records(log_path)
        assert len(records) == 1
        assert records[0]["service"] == "KRAKEN"
        assert records[0]["operation"] == "get_balance"

    def test_sync_duration_ms_populated(self, tmp_path):
        lg, log_path = _make_logger(tmp_path)

        @lg.decorator("KRAKEN", "get_balance")
        def slow_call():
            time.sleep(0.05)

        slow_call()
        rec = _read_records(log_path)[0]
        assert rec["duration_ms"] is not None
        assert rec["duration_ms"] >= 40  # at least 40 ms

    def test_sync_error_logged(self, tmp_path):
        lg, log_path = _make_logger(tmp_path)

        @lg.decorator("KRAKEN", "place_order")
        def failing_call():
            raise ValueError("order rejected")

        with pytest.raises(ValueError):
            failing_call()

        rec = _read_records(log_path)[0]
        assert rec["status"] == "error"
        assert "order rejected" in rec["error_detail"]

    def test_sync_status_code_captured(self, tmp_path):
        lg, log_path = _make_logger(tmp_path)

        class FakeResponse:
            status_code = 200

        @lg.decorator("COINGECKO", "get_global")
        def fetch():
            return FakeResponse()

        fetch()
        rec = _read_records(log_path)[0]
        assert rec["response_status"] == 200


# ---------------------------------------------------------------------------
# AC5 + async decorator
# ---------------------------------------------------------------------------

class TestAsyncDecorator:
    def test_async_function_logged(self, tmp_path):
        lg, log_path = _make_logger(tmp_path)

        @lg.decorator("COINGECKO", "get_global")
        async def async_fetch():
            await asyncio.sleep(0)
            return {"data": {}}

        asyncio.run(async_fetch())
        records = _read_records(log_path)
        assert len(records) == 1
        assert records[0]["service"] == "COINGECKO"

    def test_async_duration_ms_populated(self, tmp_path):
        lg, log_path = _make_logger(tmp_path)

        @lg.decorator("GROQ", "chat_completions")
        async def async_call():
            await asyncio.sleep(0.05)

        asyncio.run(async_call())
        rec = _read_records(log_path)[0]
        assert rec["duration_ms"] >= 40

    def test_async_error_logged(self, tmp_path):
        lg, log_path = _make_logger(tmp_path)

        @lg.decorator("COINGLASS", "mvrv")
        async def failing_async():
            raise ConnectionError("timeout")

        with pytest.raises(ConnectionError):
            asyncio.run(failing_async())

        rec = _read_records(log_path)[0]
        assert rec["status"] == "error"
        assert "timeout" in rec["error_detail"]


# ---------------------------------------------------------------------------
# module-level decorator
# ---------------------------------------------------------------------------

class TestModuleLevelDecorator:
    def test_log_integration_decorator_works(self, tmp_path):
        """log_integration with explicit logger writes to correct file."""
        log_dir = str(tmp_path / "module_logs")
        custom_logger = IntegrationLogger(log_dir=log_dir, component="MODULE")

        @log_integration("TELEGRAM", "send_message", logger=custom_logger)
        def send():
            return True

        send()
        log_path = str(tmp_path / "module_logs" / "integration.log")
        records = _read_records(log_path)
        assert records[0]["service"] == "TELEGRAM"
