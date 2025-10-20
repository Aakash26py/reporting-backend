# sql_guard.py
# Minimal, safe SELECT validator & executor for SQLite (unchanged core idea).
import re, sqlite3
from typing import Dict, List, Tuple, Optional, Any

QUAL_COL_RE = re.compile(r"(?i)\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")

def get_schema_map(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [r[0] for r in cur.fetchall()]
    schema = {}
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        schema[t.lower()] = [c[1].lower() for c in cur.fetchall()]
    return schema

def strip_code_fences(sql: str) -> str:
    return (sql or "").replace("```sql","").replace("```","").strip()

def is_select_like(sql: str) -> bool:
    return bool(re.match(r"(?is)^\s*select\b", sql or ""))

def try_explain(conn: sqlite3.Connection, sql: str, params: Optional[List[Any]] = None) -> Optional[str]:
    try:
        cur = conn.cursor()
        cur.execute(f"EXPLAIN QUERY PLAN {sql}", params or [])
        _ = cur.fetchall()
        return None
    except Exception as e:
        return str(e)

def dry_run(conn: sqlite3.Connection, sql: str, params: Optional[List[Any]] = None) -> Optional[str]:
    try:
        cur = conn.cursor()
        test_sql = sql if re.search(r"(?is)\blimit\b", sql) else (sql + " LIMIT 1")
        cur.execute(test_sql, params or [])
        _ = cur.fetchall()
        return None
    except Exception as e:
        return str(e)

def guarded_select(
    candidate_sql: str,
    params: Optional[List[Any]],
    conn: sqlite3.Connection,
    max_repair_loops: int = 1,
    planner_callback = None,
) -> Tuple[Optional[List[Dict[str, Any]]], str, Optional[str]]:
    """
    Light guard:
      1) ensure SELECT
      2) EXPLAIN check
      3) dry-run
      4) execute
    No heavy rewrites; just basic normalization.
    """
    sql = strip_code_fences(candidate_sql)
    if not is_select_like(sql):
        return None, sql, "Not a SELECT statement."

    # basic clean
    sql = sql.rstrip(";")

    # validate
    err = try_explain(conn, sql, params or [])
    if err:
        return None, sql, f"EXPLAIN failed: {err}"

    err2 = dry_run(conn, sql, params or [])
    if err2:
        return None, sql, f"dry-run failed: {err2}"

    # run fully
    try:
        cur = conn.cursor()
        cur.execute(sql, params or [])
        rows = cur.fetchall()
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in rows], sql, None
    except Exception as e:
        return None, sql, f"execution failed: {e}"
