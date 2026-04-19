"""
Author: Python Developer
Story: S20.1.1
Sprint: S2
Description: AuditLogger — thread-safe shared audit trail for all Kryptos agents.

Design principles (S20.1.1 ACs):
- All audit writes go through this class — no raw SQL outside it.
- 8 interface methods covering all event types.
- Thread-safe: uses a threading.Lock around all DB writes.
- 500 ms write timeout — propagates exception cleanly on locked DB.
- component tag present on every record.
- DB connection injected at construction time (no project-specific imports).
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)

# DDL: audit_events table (§12.3 Architecture-Design-v3)
_AUDIT_EVENTS_DDL = """
CREATE TABLE IF NOT EXISTS audit_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type      TEXT    NOT NULL,
    component       TEXT    NOT NULL DEFAULT '',
    cycle_id        TEXT    NOT NULL DEFAULT '',
    persona         TEXT    NOT NULL DEFAULT '',
    pair            TEXT    NOT NULL DEFAULT '',
    payload         TEXT    NOT NULL DEFAULT '{}',
    created_at      REAL    NOT NULL
)
"""

_CREATE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_ae_event_type ON audit_events(event_type)",
    "CREATE INDEX IF NOT EXISTS idx_ae_cycle_id   ON audit_events(cycle_id)",
    "CREATE INDEX IF NOT EXISTS idx_ae_component  ON audit_events(component)",
    "CREATE INDEX IF NOT EXISTS idx_ae_created_at ON audit_events(created_at)",
]


class AuditLogger:
    """
    Thread-safe audit trail writer for Kryptos agents.

    All eight event types are written to the ``audit_events`` table in the
    injected SQLite database.  No other tables are touched.

    Parameters
    ----------
    db_path : str
        Absolute or relative path to the SQLite database file.
    component : str
        Name of the calling component (e.g. ``"QSA"``, ``"AIE"``, ``"ROM"``).
        Written to every record via the ``component`` column.
    timeout : float
        SQLite busy-timeout in seconds.  Default ``0.5`` (500 ms).
        ``sqlite3.OperationalError`` is raised (not swallowed) on timeout so
        the caller can handle it appropriately.
    """

    def __init__(self, db_path: str, component: str = "", timeout: float = 0.5) -> None:
        self._db_path = db_path
        self._component = component
        self._timeout = timeout
        self._lock = threading.Lock()
        self._ensure_schema()

    # ── schema initialisation ────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        """Create ``audit_events`` table and indexes if they do not exist."""
        with self._connect() as conn:
            conn.execute(_AUDIT_EVENTS_DDL)
            for stmt in _CREATE_INDEXES:
                conn.execute(stmt)
            conn.commit()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, timeout=self._timeout)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    # ── private write helper ─────────────────────────────────────────────────

    def _write(
        self,
        event_type: str,
        cycle_id: str = "",
        persona: str = "",
        pair: str = "",
        payload: Optional[dict[str, Any]] = None,
    ) -> None:
        """Insert one row into ``audit_events`` under a mutex."""
        payload_json = json.dumps(payload or {}, default=str)
        now = time.time()
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO audit_events
                        (event_type, component, cycle_id, persona, pair, payload, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (event_type, self._component, cycle_id, persona, pair, payload_json, now),
                )
                conn.commit()

    # ── public interface (8 methods, AC1 of S20.1.1) ─────────────────────────

    def log_cycle(
        self,
        cycle_id: str,
        persona: str = "",
        playbook: str = "",
        regime: str = "",
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record the start of a trading cycle."""
        payload = {"playbook": playbook, "regime": regime, **(extra or {})}
        self._write("cycle_start", cycle_id=cycle_id, persona=persona, payload=payload)

    def log_signal(
        self,
        cycle_id: str,
        pair: str,
        direction: str,
        score: int,
        reasons: list[str],
        persona: str = "",
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a signal evaluation result for a pair."""
        payload: dict[str, Any] = {
            "direction": direction,
            "score": score,
            "reasons": reasons,
            **(extra or {}),
        }
        self._write("signal", cycle_id=cycle_id, persona=persona, pair=pair, payload=payload)

    def log_trade(
        self,
        cycle_id: str,
        pair: str,
        side: str,
        price: float,
        volume: float,
        usd_value: float,
        exit_reason: str = "",
        pnl_usd: Optional[float] = None,
        pnl_pct: Optional[float] = None,
        persona: str = "",
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a trade execution (entry or exit)."""
        payload: dict[str, Any] = {
            "side": side,
            "price": price,
            "volume": volume,
            "usd_value": usd_value,
            "exit_reason": exit_reason,
            "pnl_usd": pnl_usd,
            "pnl_pct": pnl_pct,
            **(extra or {}),
        }
        self._write("trade", cycle_id=cycle_id, persona=persona, pair=pair, payload=payload)

    def log_balance_snapshot(
        self,
        cycle_id: str,
        total_usd: float,
        cash_usd: float,
        positions_usd: float,
        persona: str = "",
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a portfolio balance snapshot."""
        payload: dict[str, Any] = {
            "total_usd": total_usd,
            "cash_usd": cash_usd,
            "positions_usd": positions_usd,
            **(extra or {}),
        }
        self._write("balance_snapshot", cycle_id=cycle_id, persona=persona, payload=payload)

    def log_error(
        self,
        cycle_id: str,
        error_type: str,
        message: str,
        persona: str = "",
        pair: str = "",
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record an error or exception event."""
        payload: dict[str, Any] = {
            "error_type": error_type,
            "message": message,
            **(extra or {}),
        }
        self._write("error", cycle_id=cycle_id, persona=persona, pair=pair, payload=payload)
        logger.error("[AUDIT] %s: %s (cycle=%s pair=%s)", error_type, message, cycle_id, pair)

    def log_circuit_breaker(
        self,
        cycle_id: str,
        state: str,
        reason: str,
        pause_until: Optional[float] = None,
        persona: str = "",
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a circuit breaker state change (open / close)."""
        payload: dict[str, Any] = {
            "state": state,
            "reason": reason,
            "pause_until": pause_until,
            **(extra or {}),
        }
        self._write("circuit_breaker", cycle_id=cycle_id, persona=persona, payload=payload)

    def log_fulfillment(
        self,
        cycle_id: str,
        pair: str,
        order_id: str,
        order_type: str,
        status: str,
        persona: str = "",
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record a fulfillment audit event (FulfillmentService order lifecycle)."""
        payload: dict[str, Any] = {
            "order_id": order_id,
            "order_type": order_type,
            "status": status,
            **(extra or {}),
        }
        self._write("fulfillment", cycle_id=cycle_id, persona=persona, pair=pair, payload=payload)

    def log_agent_card(
        self,
        agent_name: str,
        version: str,
        capabilities: list[str],
        cycle_id: str = "",
        persona: str = "",
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        """Record an agent registration / heartbeat card (A2A protocol)."""
        payload: dict[str, Any] = {
            "agent_name": agent_name,
            "version": version,
            "capabilities": capabilities,
            **(extra or {}),
        }
        self._write("agent_card", cycle_id=cycle_id, persona=persona, payload=payload)

    # ── query helpers (read-only) ─────────────────────────────────────────────

    def get_events(
        self,
        event_type: Optional[str] = None,
        cycle_id: Optional[str] = None,
        component: Optional[str] = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        Return recent audit events as a list of dicts.
        All filters are optional AND-combined.
        """
        clauses: list[str] = []
        params: list[Any] = []
        if event_type:
            clauses.append("event_type = ?")
            params.append(event_type)
        if cycle_id:
            clauses.append("cycle_id = ?")
            params.append(cycle_id)
        if component:
            clauses.append("component = ?")
            params.append(component)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM audit_events {where} ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql, params).fetchall()
            return [dict(r) for r in rows]
