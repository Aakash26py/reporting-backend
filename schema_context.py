# schema_context.py
import sqlite3, json, os, time, logging
from typing import Dict, List, Tuple, Any
from functools import lru_cache

log = logging.getLogger("schema")

DB_PATH = "chat_history.db"
CACHE_TTL = 86400  # 1 day (in seconds)


def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _fetch_schema() -> Dict[str, List[str]]:
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [r[0] for r in cur.fetchall()]
    schema = {}
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        schema[t] = [r[1] for r in cur.fetchall()]
    conn.close()
    return schema


def _fetch_descriptions() -> Dict[str, Dict[str, str]]:
    """Fetch table/column descriptions from DB if exists."""
    conn = get_db(); cur = conn.cursor()
    desc = {}
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='schema_descriptions'")
        if cur.fetchone():
            cur.execute("SELECT table_name, column_name, description FROM schema_descriptions")
            for t, c, d in cur.fetchall():
                desc.setdefault(t, {})[c] = d
    except Exception as e:
        log.warning("schema_descriptions fetch failed: %s", e)
    finally:
        conn.close()
    return desc


def _fetch_enum_values() -> Dict[str, Dict[str, List[str]]]:
    """Collect enum-like distinct values for each table.column."""
    conn = get_db(); cur = conn.cursor()
    enum_map = {}
    schema = _fetch_schema()
    for table, cols in schema.items():
        for col in cols:
            try:
                cur.execute(f"SELECT COUNT(DISTINCT {col}) FROM {table}")
                count = cur.fetchone()[0]
                if 1 < count <= 50:  # treat as enum-like
                    cur.execute(f"SELECT DISTINCT {col} FROM {table} LIMIT 50")
                    values = [r[0] for r in cur.fetchall() if r[0] is not None]
                    if values:
                        enum_map.setdefault(table, {})[col] = values
            except Exception:
                continue
    conn.close()
    return enum_map


_cached_data: Tuple[float, Dict[str, Any]] = (0, {})


def _refresh_cache() -> Dict[str, Any]:
    schema = _fetch_schema()
    desc = _fetch_descriptions()
    enums = _fetch_enum_values()

    merged = {
        "schema": schema,
        "descriptions": desc,
        "enums": enums,
        "schema_text": build_schema_text(schema, desc),
    }

    global _cached_data
    _cached_data = (time.time(), merged)
    return merged


def _get_cached() -> Dict[str, Any]:
    ts, data = _cached_data
    if time.time() - ts > CACHE_TTL or not data:
        return _refresh_cache()
    return data


def get_schema_context() -> Dict[str, Any]:
    """Returns schema + descriptions + enums (cached for 24h)."""
    return _get_cached()


def build_schema_text(schema: Dict[str, List[str]], desc: Dict[str, Dict[str, str]]) -> str:
    """Pretty-print schema with optional descriptions for LLM context."""
    lines = []
    for table, cols in schema.items():
        lines.append(f"{table}:")
        for col in cols:
            d = desc.get(table, {}).get(col)
            if d:
                lines.append(f"  - {col}: {d}")
            else:
                lines.append(f"  - {col}")
        lines.append("")  # spacer
    return "\n".join(lines)

