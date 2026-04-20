"""
mocha-python-agent — Agent infrastructure for the Kryptos multi-agent mesh.

Provides AgentCard (descriptor), AgentBootstrap (lifecycle + DB registration),
and helpers for agent discovery.

Story: S20.3.1
"""

import json
import logging
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_REGISTRY_SCHEMA = """
CREATE TABLE IF NOT EXISTS agent_registry (
    agent_id       TEXT PRIMARY KEY,
    name           TEXT NOT NULL,
    version        TEXT NOT NULL,
    status         TEXT NOT NULL DEFAULT 'stopped',
    last_heartbeat TEXT,
    registered_at  TEXT NOT NULL,
    metadata_json  TEXT
);
"""


# ──────────────────────────────────────────────────────────────
# AgentCard — immutable descriptor
# ──────────────────────────────────────────────────────────────

@dataclass
class AgentCard:
    """
    Immutable identity card for a deployed agent instance.

    Fields:
        agent_id:     UUID4 assigned at construction time (auto-generated when omitted).
        name:         Human-readable agent name, e.g. "AIE", "QSA", "ROM".
        version:      Semantic version string, e.g. "1.0.0".
        capabilities: Declared capability tags, e.g. ["llm-decision", "risk-managed"].
        status:       Runtime status: "running" | "stopped" | "error".
        host:         Bind address (always "127.0.0.1" for local agents).
        port:         Port the agent's HTTP health endpoint listens on (0 if headless).
        started_at:   ISO-8601 UTC timestamp when start() was called (empty before start).
    """

    name: str
    version: str
    capabilities: list[str] = field(default_factory=list)
    status: str = "stopped"
    host: str = "127.0.0.1"
    port: int = 0
    started_at: str = ""
    agent_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_row(cls, row: dict) -> "AgentCard":
        meta = json.loads(row.get("metadata_json") or "{}")
        return cls(
            agent_id=row["agent_id"],
            name=row["name"],
            version=row["version"],
            capabilities=meta.get("capabilities", []),
            status=row["status"],
            host=meta.get("host", "127.0.0.1"),
            port=meta.get("port", 0),
            started_at=meta.get("started_at", ""),
        )


# ──────────────────────────────────────────────────────────────
# AgentBootstrap — lifecycle management
# ──────────────────────────────────────────────────────────────

class AgentBootstrap:
    """
    Manages agent lifecycle (start / stop / heartbeat) and persists
    registration records to an SQLite `agent_registry` table.

    Usage::

        card = AgentCard(name="AIE", version="3.0.0", capabilities=["llm-decision"])
        bootstrap = AgentBootstrap(db_path="data/paper_trading.db", card=card)
        bootstrap.start()
        # ... agent loop ...
        bootstrap.stop()
    """

    def __init__(self, db_path: str, card: AgentCard) -> None:
        self._db_path = db_path
        self._card = card
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_schema()

    # ── Internal helpers ────────────────────────────────────────

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _ensure_schema(self) -> None:
        conn = self._get_conn()
        conn.executescript(_REGISTRY_SCHEMA)
        conn.commit()
        conn.close()

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    # ── Lifecycle ──────────────────────────────────────────────

    def start(self) -> None:
        """
        Register (or upsert) this agent as running in the registry.
        Sets status='running', records started_at, and writes metadata_json.
        """
        self._card.status = "running"
        self._card.started_at = self._now_iso()
        metadata = {
            "capabilities": self._card.capabilities,
            "host": self._card.host,
            "port": self._card.port,
            "started_at": self._card.started_at,
        }
        conn = self._get_conn()
        conn.execute(
            """
            INSERT INTO agent_registry
                (agent_id, name, version, status, last_heartbeat, registered_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(agent_id) DO UPDATE SET
                status         = excluded.status,
                last_heartbeat = excluded.last_heartbeat,
                metadata_json  = excluded.metadata_json
            """,
            (
                self._card.agent_id,
                self._card.name,
                self._card.version,
                "running",
                self._now_iso(),
                self._now_iso(),
                json.dumps(metadata),
            ),
        )
        conn.commit()
        conn.close()
        logger.info(
            "[AgentBootstrap] %s v%s started — id=%s",
            self._card.name, self._card.version, self._card.agent_id,
        )

    def stop(self) -> None:
        """Update agent status to 'stopped' in the registry."""
        self._card.status = "stopped"
        conn = self._get_conn()
        conn.execute(
            "UPDATE agent_registry SET status=?, last_heartbeat=? WHERE agent_id=?",
            ("stopped", self._now_iso(), self._card.agent_id),
        )
        conn.commit()
        conn.close()
        logger.info("[AgentBootstrap] %s stopped", self._card.name)

    def heartbeat(self) -> None:
        """Stamp last_heartbeat with the current UTC time."""
        now = self._now_iso()
        conn = self._get_conn()
        conn.execute(
            "UPDATE agent_registry SET last_heartbeat=? WHERE agent_id=?",
            (now, self._card.agent_id),
        )
        conn.commit()
        conn.close()

    # ── Discovery ──────────────────────────────────────────────

    @staticmethod
    def get_live_agents(db_path: str, stale_secs: int = 120) -> list[AgentCard]:
        """
        Return all agents with status='running' whose last_heartbeat is within
        the last stale_secs seconds. Agents that have crashed without calling
        stop() are excluded once their heartbeat goes stale.

        Args:
            db_path:    Path to the SQLite database file.
            stale_secs: Heartbeat freshness window in seconds (default 120).

        Returns:
            List of AgentCard instances for live agents.
        """
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL;")
            cutoff = datetime.fromtimestamp(
                time.time() - stale_secs, tz=timezone.utc
            ).isoformat()
            rows = conn.execute(
                "SELECT * FROM agent_registry WHERE status='running' AND last_heartbeat >= ?",
                (cutoff,),
            ).fetchall()
            conn.close()
            return [AgentCard.from_row(dict(r)) for r in rows]
        except Exception as e:
            logger.warning("[AgentBootstrap] get_live_agents failed: %s", e)
            return []


__all__ = ["AgentCard", "AgentBootstrap"]

