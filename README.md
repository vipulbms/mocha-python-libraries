# mocha-python-libraries

Shared Python library monorepo for the Kryptos v3 multi-agent trading platform.

## Packages

| Package | Install | Purpose |
|---|---|---|
| **mocha-python-audit** | `pip install -e ./packages/mocha_python_audit` | Structured audit trail (SQLite WAL, 8 event types) |
| **mocha-python-logging** | `pip install -e ./packages/mocha_python_logging` | Outbound integration logging (JSON-lines, rotation, redaction) |
| **mocha-python-ai** | `pip install -e ./packages/mocha_python_ai` | LLM client helpers — *Sprint S4 scope* |
| **mocha-python-agent** | `pip install -e ./packages/mocha_python_agent` | A2A protocol, agent cards, task bus — *Sprint S4 scope* |

## Repository layout

```
packages/
  mocha_python_audit/       # Sprint S2
  mocha_python_logging/     # Sprint S2
  mocha_python_ai/          # Sprint S4 scaffold
  mocha_python_agent/       # Sprint S4 scaffold
.github/workflows/ci.yml    # per-package matrix CI
```

## Quick start

```bash
pip install -e "packages/mocha_python_audit[dev]" && pytest packages/mocha_python_audit/tests/ -v
pip install -e "packages/mocha_python_logging[dev]" && pytest packages/mocha_python_logging/tests/ -v
```

## Integration in Kryptos

Consumed by the Kryptos agent starting in Sprint S4 (`release/2.0.0`).

