"""
Author: Tester
Story: S20.1.1
Sprint: S2
Description: Tests for AuditLogger — concurrent writes, timeout, all 8 methods.
"""
from __future__ import annotations

import sqlite3
import threading
import time
import uuid
from pathlib import Path

import pytest

from mocha_python_audit import AuditLogger


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / f"test_audit_{uuid.uuid4().hex[:8]}.db")


@pytest.fixture()
def audit(db_path: str) -> AuditLogger:
    return AuditLogger(db_path, component="TEST")


# ── Schema setup ──────────────────────────────────────────────────────────────

def test_schema_created(db_path: str) -> None:
    AuditLogger(db_path, component="X")
    conn = sqlite3.connect(db_path)
    tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    conn.close()
    assert "audit_events" in tables


def test_second_init_is_idempotent(db_path: str) -> None:
    AuditLogger(db_path, component="X")
    AuditLogger(db_path, component="Y")  # must not raise


# ── All 8 interface methods ───────────────────────────────────────────────────

def test_log_cycle(audit: AuditLogger) -> None:
    audit.log_cycle("cycle-1", persona="conservative", playbook="ranging", regime="stable")
    rows = audit.get_events(event_type="cycle_start")
    assert len(rows) == 1
    assert rows[0]["cycle_id"] == "cycle-1"
    assert rows[0]["component"] == "TEST"


def test_log_signal(audit: AuditLogger) -> None:
    audit.log_signal("c1", "BTC/USD", "BUY", 7, ["RSI oversold"], persona="medium")
    rows = audit.get_events(event_type="signal")
    assert rows[0]["pair"] == "BTC/USD"
    import json
    payload = json.loads(rows[0]["payload"])
    assert payload["score"] == 7
    assert payload["direction"] == "BUY"


def test_log_trade(audit: AuditLogger) -> None:
    audit.log_trade("c1", "ETH/USD", "BUY", 2500.0, 0.02, 50.0, persona="high")
    rows = audit.get_events(event_type="trade")
    assert len(rows) == 1
    assert rows[0]["pair"] == "ETH/USD"


def test_log_balance_snapshot(audit: AuditLogger) -> None:
    audit.log_balance_snapshot("c1", 1000.0, 800.0, 200.0)
    rows = audit.get_events(event_type="balance_snapshot")
    assert len(rows) == 1


def test_log_error(audit: AuditLogger) -> None:
    audit.log_error("c1", "TimeoutError", "Groq timed out", pair="BTC/USD")
    rows = audit.get_events(event_type="error")
    assert len(rows) == 1
    assert rows[0]["pair"] == "BTC/USD"


def test_log_circuit_breaker(audit: AuditLogger) -> None:
    audit.log_circuit_breaker("c1", "open", "consecutive stops", pause_until=time.time() + 3600)
    rows = audit.get_events(event_type="circuit_breaker")
    assert len(rows) == 1


def test_log_fulfillment(audit: AuditLogger) -> None:
    audit.log_fulfillment("c1", "SOL/USD", "order-123", "limit", "filled")
    rows = audit.get_events(event_type="fulfillment")
    assert len(rows) == 1


def test_log_agent_card(audit: AuditLogger) -> None:
    audit.log_agent_card("QSA", "1.0.0", ["signal_compute", "feed_heartbeat"])
    rows = audit.get_events(event_type="agent_card")
    assert len(rows) == 1


# ── Filtering ─────────────────────────────────────────────────────────────────

def test_get_events_filters_by_type(audit: AuditLogger) -> None:
    audit.log_cycle("c1", persona="conservative")
    audit.log_signal("c1", "ETH/USD", "BUY", 6, [])
    cycle_rows = audit.get_events(event_type="cycle_start")
    signal_rows = audit.get_events(event_type="signal")
    assert len(cycle_rows) == 1
    assert len(signal_rows) == 1


def test_get_events_filters_by_component(db_path: str) -> None:
    qsa = AuditLogger(db_path, component="QSA")
    rom = AuditLogger(db_path, component="ROM")
    qsa.log_cycle("c1")
    rom.log_cycle("c2")
    qsa_rows = qsa.get_events(component="QSA")
    assert len(qsa_rows) == 1
    assert qsa_rows[0]["component"] == "QSA"


# ── Concurrency test (AC4) ────────────────────────────────────────────────────

def test_concurrent_writes(db_path: str) -> None:
    """5 threads call log_signal simultaneously — no corruption, all 5 visible."""
    audit = AuditLogger(db_path, component="CONCURRENT")
    errors: list[Exception] = []

    def write(i: int) -> None:
        try:
            audit.log_signal(f"cycle-{i}", f"P{i}/USD", "BUY", i, [f"reason-{i}"])
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=write, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"Concurrent write errors: {errors}"
    rows = audit.get_events(event_type="signal", limit=10)
    assert len(rows) == 5, f"Expected 5 records, got {len(rows)}"


# ── Timeout test (AC5) ────────────────────────────────────────────────────────

def test_write_timeout_raises(tmp_path) -> None:
    """A locked DB raises OperationalError within the timeout window."""
    db_path = str(tmp_path / "lock_test.db")
    # Construct AuditLogger first so schema is created before any lock
    audit = AuditLogger(db_path, component="TIMEOUT", timeout=0.1)

    # Now hold an exclusive lock so the next write times out
    raw = sqlite3.connect(db_path)
    raw.execute("BEGIN EXCLUSIVE")

    with pytest.raises(sqlite3.OperationalError):
        audit.log_cycle("c-timeout")

    raw.rollback()
    raw.close()
