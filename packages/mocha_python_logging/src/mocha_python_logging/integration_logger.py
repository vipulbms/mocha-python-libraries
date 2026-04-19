"""
Author: Python Developer
Story: S20.1.2
Sprint: S2
Description: IntegrationLogger and @log_integration decorator.

Every outbound network call (Groq, Kraken, CoinGecko, CoinGlass, Telegram)
is wrapped with @log_integration so latency and status are logged without
manual instrumentation at each call site.

Features (S20.1.2 ACs):
- JSON-lines output to /logs/integration.log (rotating 100 MB × 5)
- API key / Authorization header values redacted to [REDACTED]
- duration_ms captured for both sync and async decorated functions
- Required fields: timestamp, component, service, operation, request_summary,
  response_status, duration_ms, status, error_detail, cycle_id
- cycle_id injected via contextvars (same pattern as request_id in trading_agent.py)
"""
from __future__ import annotations

import asyncio
import contextvars
import functools
import json
import logging
import logging.handlers
import re
import time
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar

F = TypeVar("F", bound=Callable[..., Any])

# ── ContextVar for cycle_id propagation ──────────────────────────────────────
# Callers set this before starting an integration call so every log record is
# traceable to the trading cycle that triggered the request.
cycle_id_var: contextvars.ContextVar[str] = contextvars.ContextVar(
    "integration_cycle_id", default=""
)

# ── Sensitive key patterns ────────────────────────────────────────────────────
_REDACT_KEYS_RE = re.compile(
    r"(api[_-]?key|secret|token|authorization|password|x-api-key)",
    re.IGNORECASE,
)


def _redact(value: Any) -> Any:
    """
    Redact sensitive string values from a dict, list, or scalar.
    Walks nested dicts/lists.  Strings matching known sensitive keys
    are replaced with ``[REDACTED]``.
    """
    if isinstance(value, dict):
        return {
            k: "[REDACTED]" if _REDACT_KEYS_RE.search(str(k)) else _redact(v)
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def _safe_summary(obj: Any, max_len: int = 500) -> str:
    """Serialise *obj* to a truncated JSON string for the request_summary field."""
    try:
        raw = json.dumps(_redact(obj), default=str)
    except Exception:
        raw = str(obj)[:max_len]
    return raw[:max_len]


class IntegrationLogger:
    """
    Writes outbound integration events as JSON-lines to a rotating log file.

    Parameters
    ----------
    log_dir : str
        Directory where ``integration.log`` is written.  Created if absent.
    component : str
        Name of the calling component (e.g. ``"QSA"``, ``"AIE"``).
    max_bytes : int
        Maximum size per log file before rotation.  Default 100 MB.
    backup_count : int
        Number of rotated backup files retained.  Default 5.
    """

    def __init__(
        self,
        log_dir: str = "/logs",
        component: str = "",
        max_bytes: int = 100 * 1024 * 1024,
        backup_count: int = 5,
    ) -> None:
        self._component = component
        self._logger = self._setup_logger(log_dir, max_bytes, backup_count)

    # ── logger setup ─────────────────────────────────────────────────────────

    @staticmethod
    def _setup_logger(
        log_dir: str, max_bytes: int, backup_count: int
    ) -> logging.Logger:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        log_path = str(Path(log_dir) / "integration.log")
        handler = logging.handlers.RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        lg = logging.getLogger(f"mocha_integration_{log_dir}")
        lg.setLevel(logging.DEBUG)
        if not lg.handlers:
            lg.addHandler(handler)
        lg.propagate = False
        return lg

    # ── public write method ───────────────────────────────────────────────────

    def write(
        self,
        service: str,
        operation: str,
        request_summary: str = "",
        response_status: Optional[int] = None,
        duration_ms: Optional[float] = None,
        status: str = "ok",
        error_detail: str = "",
        cycle_id: str = "",
    ) -> None:
        """Write one integration log record."""
        record = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "component": self._component,
            "service": service,
            "operation": operation,
            "request_summary": request_summary,
            "response_status": response_status,
            "duration_ms": round(duration_ms, 2) if duration_ms is not None else None,
            "status": status,
            "error_detail": error_detail,
            "cycle_id": cycle_id or cycle_id_var.get(""),
        }
        self._logger.info(json.dumps(record))

    # ── decorator factory ─────────────────────────────────────────────────────

    def decorator(
        self,
        service: str,
        operation: str,
        request_arg: Optional[str] = None,
    ) -> Callable[[F], F]:
        """
        Return a decorator that logs latency and status for the wrapped function.

        Parameters
        ----------
        service : str
            Name of the external service (e.g. ``"GROQ"``, ``"COINGECKO"``).
        operation : str
            Logical operation name (e.g. ``"chat_completions"``, ``"get_global"``).
        request_arg : str | None
            Name of a keyword argument whose value should be included in
            ``request_summary`` (after redaction).  If ``None``, the summary
            is left empty.
        """

        def outer(fn: F) -> F:
            if asyncio.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                    req_summary = _safe_summary(kwargs.get(request_arg)) if request_arg else ""
                    t0 = time.perf_counter()
                    status = "ok"
                    error_detail = ""
                    response_status = None
                    try:
                        result = await fn(*args, **kwargs)
                        if hasattr(result, "status"):
                            response_status = int(result.status)
                        elif hasattr(result, "status_code"):
                            response_status = int(result.status_code)
                        return result
                    except Exception as exc:
                        status = "error"
                        error_detail = str(exc)
                        raise
                    finally:
                        duration_ms = (time.perf_counter() - t0) * 1000
                        self.write(
                            service=service,
                            operation=operation,
                            request_summary=req_summary,
                            response_status=response_status,
                            duration_ms=duration_ms,
                            status=status,
                            error_detail=error_detail,
                        )

                return async_wrapper  # type: ignore[return-value]

            else:

                @functools.wraps(fn)
                def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
                    req_summary = _safe_summary(kwargs.get(request_arg)) if request_arg else ""
                    t0 = time.perf_counter()
                    status = "ok"
                    error_detail = ""
                    response_status = None
                    try:
                        result = fn(*args, **kwargs)
                        if hasattr(result, "status_code"):
                            response_status = int(result.status_code)
                        return result
                    except Exception as exc:
                        status = "error"
                        error_detail = str(exc)
                        raise
                    finally:
                        duration_ms = (time.perf_counter() - t0) * 1000
                        self.write(
                            service=service,
                            operation=operation,
                            request_summary=req_summary,
                            response_status=response_status,
                            duration_ms=duration_ms,
                            status=status,
                            error_detail=error_detail,
                        )

                return sync_wrapper  # type: ignore[return-value]

        return outer


# ── module-level default instance ─────────────────────────────────────────────
# Creating a default instance allows the decorator to be used without
# explicitly constructing an IntegrationLogger (matches the v2 usage pattern).
_default_logger: Optional[IntegrationLogger] = None


def _get_default_logger() -> IntegrationLogger:
    global _default_logger
    if _default_logger is None:
        _default_logger = IntegrationLogger()
    return _default_logger


def log_integration(
    service: str,
    operation: str,
    request_arg: Optional[str] = None,
    logger: Optional[IntegrationLogger] = None,
) -> Callable[[F], F]:
    """
    Module-level decorator factory.  Uses the default ``IntegrationLogger`` if
    none is provided.

    Usage::

        @log_integration("COINGECKO", "get_global")
        async def fetch_btc_dominance(...): ...

        @log_integration("GROQ", "chat_completions", logger=my_logger)
        def call_groq(...): ...
    """
    lg = logger or _get_default_logger()
    return lg.decorator(service, operation, request_arg)
