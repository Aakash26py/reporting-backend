# sql_guard.py
# SQLite-safe verification, normalization, and guarded execution for SELECT statements.
from __future__ import annotations
import re
import sqlite3
from typing import Dict, List, Tuple, Optional, Any

# ----------------------------- Toggle debug ------------------------------------
DEBUG = False
def _dbg(label: str, text: str):
    if DEBUG:
        print(f"\n[sql_guard:{label}]\n{text}\n")

# ----------------------------- Schema Introspection -----------------------------

def get_schema_map(conn: sqlite3.Connection) -> Dict[str, List[str]]:
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [r[0] for r in cur.fetchall()]
    schema: Dict[str, List[str]] = {}
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = [c[1].lower() for c in cur.fetchall()]
        schema[t.lower()] = cols
    return schema

# ----------------------------- Helpers -----------------------------------------

_SQL_NAME = r"[A-Za-z_][A-Za-z0-9_]*"
QUAL_COL_RE = re.compile(rf"(?i)\b({_SQL_NAME})\.({_SQL_NAME})\b")
STRING_LIT_RE = re.compile(r"(?s)'([^']|'')*'")
LINE_COMMENT_RE = re.compile(r"--[^\n]*")
BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)

SQL_KEYWORDS = {
    "SELECT","FROM","JOIN","ON","WHERE","AND","OR","NOT","IN","AS",
    "GROUP","BY","ORDER","LIMIT","DESC","ASC","HAVING","DISTINCT",
    "CASE","WHEN","THEN","ELSE","END","LIKE","IS","NULL","BETWEEN",
    "COUNT","SUM","AVG","MIN","MAX","CAST","COALESCE","SUBSTR","STRFTIME"
}

def strip_code_fences(sql: str) -> str:
    sql = (sql or "").strip()
    sql = sql.replace("```sql", "").replace("```", "").strip()
    return sql

def remove_trailing_semicolon(sql: str) -> str:
    return (sql or "").rstrip().rstrip(";")

def is_select_like(sql: str) -> bool:
    return bool(re.match(r"(?is)^\s*select\b", sql or ""))

def _strip_comments_keep_spans(sql: str) -> Tuple[str, List[str]]:
    sentinels: List[str] = []
    def stash(m: re.Match) -> str:
        sentinels.append(m.group(0))
        return f"__HIDE_{len(sentinels)-1}__"
    s = BLOCK_COMMENT_RE.sub(stash, sql)
    s = LINE_COMMENT_RE.sub(stash, s)
    s = STRING_LIT_RE.sub(stash, s)
    return s, sentinels

def _restore_sentinels(s: str, sentinels: List[str]) -> str:
    def putback(m: re.Match) -> str:
        idx = int(m.group(1))
        return sentinels[idx]
    return re.sub(r"__HIDE_(\d+)__", putback, s)

def find_tables_in_from(sql: str) -> List[str]:
    from_match = re.search(r"(?is)\bfrom\b(.*?)(\bwhere\b|\bgroup\b|\border\b|\bhaving\b|\blimit\b|$)", sql or "")
    if not from_match:
        return []
    from_block = from_match.group(1)
    parts = re.split(r"(?is)\bjoin\b|,", from_block)
    tables = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.match(rf"(?is)({_SQL_NAME})(?:\s+(?:as\s+)?({_SQL_NAME}))?", p)
        if m:
            tbl = m.group(1)
            tables.append(tbl)
    return [t.lower() for t in tables]

def extract_alias_map(sql: str) -> Dict[str, str]:
    """alias->table map"""
    alias_map: Dict[str,str] = {}
    for m in re.finditer(rf"(?is)\b(from|join)\s+({_SQL_NAME})(?:\s+(?:as\s+)?({_SQL_NAME}))?", sql or ""):
        table = m.group(2)
        alias = m.group(3)
        if alias:
            alias_map[alias.lower()] = table.lower()
    return alias_map

def invert_alias_map(alias_to_table: Dict[str,str]) -> Dict[str,str]:
    """table->alias map (first alias wins)"""
    table_to_alias: Dict[str, str] = {}
    for a, t in alias_to_table.items():
        table_to_alias.setdefault(t, a)
    return table_to_alias

# ----------------------------- Lint & Qualify ----------------------------------

def lint_unknown_tables_and_columns(
    sql: str, schema: Dict[str, List[str]]
) -> Tuple[List[str], List[str]]:
    unknown_tables: List[str] = []
    for t in find_tables_in_from(sql):
        if t not in schema:
            unknown_tables.append(t)

    unknown_cols: List[str] = []
    for t, c in QUAL_COL_RE.findall(sql):
        tl = t.lower(); cl = c.lower()
        if tl in schema and cl not in schema[tl]:
            unknown_cols.append(f"{t}.{c}")
    return unknown_tables, unknown_cols

def qualify_unqualified_columns(sql: str, schema: Dict[str, List[str]]) -> str:
    """
    Qualify bare column names using tables present in FROM/JOIN.
    Idempotent. Protect SELECT aliases (AS alias_name) so we don't qualify them.
    """
    original = sql
    protected, sentinels = _strip_comments_keep_spans(sql)

    alias_names: List[str] = []
    def alias_stash(m: re.Match) -> str:
        alias_names.append(m.group(1))
        return f" AS __ALIAS_{len(alias_names)-1}__"
    protected = re.sub(r"(?is)\bas\s+([A-Za-z_][A-Za-z0-9_]*)\b", alias_stash, protected)

    alias_to_table = extract_alias_map(protected)
    table_to_alias = invert_alias_map(alias_to_table)
    from_tables = find_tables_in_from(protected)
    known_aliases = set(alias_to_table.keys())
    known_tables = set(from_tables)

    col_to_single_table: Dict[str, str] = {}
    for t in from_tables:
        for col in schema.get(t, []):
            if col not in col_to_single_table:
                col_to_single_table[col] = t
            elif col_to_single_table[col] != t:
                col_to_single_table[col] = "__AMBIG__"

    ident_re = re.compile(rf"(?i)(?<!\.)\b([A-Za-z_][A-Za-z0-9_]*)\b(?!\s*\.)")

    def repl(m: re.Match) -> str:
        word = m.group(1)
        U = word.upper(); lw = word.lower()
        if U in SQL_KEYWORDS or lw in known_aliases or lw in known_tables:
            return word
        if re.match(r"__ALIAS_\d+__", word):
            return word
        owner = col_to_single_table.get(lw)
        if not owner or owner == "__AMBIG__":
            return word
        alias = table_to_alias.get(owner)
        prefix = alias or owner
        return f"{prefix}.{word}"

    qualified = ident_re.sub(repl, protected)
    qualified = re.sub(r" AS __ALIAS_(\d+)__", lambda m: f" AS {alias_names[int(m.group(1))]}", qualified)

    out = _restore_sentinels(qualified, sentinels)
    _dbg("qualify_unqualified_columns", f"before:\n{original}\n\nafter:\n{out}")
    return out

def map_table_to_alias(sql: str) -> str:
    original = sql
    protected, sentinels = _strip_comments_keep_spans(sql)
    alias_to_table = extract_alias_map(protected)
    if not alias_to_table:
        return sql
    table_to_alias = invert_alias_map(alias_to_table)
    def repl(m: re.Match) -> str:
        left, col = m.group(1), m.group(2)
        ll = left.lower()
        if ll in alias_to_table:
            return m.group(0)
        alias = table_to_alias.get(ll)
        if alias:
            return f"{alias}.{col}"
        return m.group(0)
    remapped = QUAL_COL_RE.sub(repl, protected)
    out = _restore_sentinels(remapped, sentinels)
    _dbg("map_table_to_alias", f"before:\n{original}\n\nafter:\n{out}")
    return out

# ----------------------------- Explain & Dry-run --------------------------------

def try_explain(conn: sqlite3.Connection, sql: str, params: Optional[List[Any]] = None) -> Optional[str]:
    cur = conn.cursor()
    try:
        cur.execute(f"EXPLAIN QUERY PLAN {sql}", params or [])
        _ = cur.fetchall()
        return None
    except Exception as e:
        return str(e)

def _has_limit(sql: str) -> bool:
    return bool(re.search(r"(?is)\blimit\b", sql or ""))

def dry_run_sample(
    conn: sqlite3.Connection,
    sql: str,
    params: Optional[List[Any]] = None
) -> Tuple[Optional[List[sqlite3.Row]], Optional[str]]:
    cur = conn.cursor()
    try:
        sql_to_run = sql if _has_limit(sql) else (sql + " LIMIT 1")
        cur.execute(sql_to_run, params or [])
        _ = cur.fetchall()
        return _, None
    except Exception as e:
        return None, str(e)

# ----------------------------- Repair & Guard (idempotent) ----------------------

def normalize_sql_once(sql: str, conn: sqlite3.Connection, schema: Dict[str, List[str]]) -> str:
    s = map_table_to_alias(sql)
    s = qualify_unqualified_columns(s, schema)
    return s

def repair_sql(
    sql: str,
    conn: sqlite3.Connection,
    schema: Dict[str, List[str]],
    params: Optional[List[Any]] = None
) -> Tuple[str, Optional[str]]:
    if not sql:
        return sql, "empty SQL"
    s = strip_code_fences(sql)
    s = remove_trailing_semicolon(s)
    if not is_select_like(s):
        return s, "non-SELECT statement"

    for _ in range(2):
        new_s = normalize_sql_once(s, conn, schema)
        if new_s == s:
            break
        s = new_s

    alias_to_table = extract_alias_map(s)
    def alias_to_table_repl(m: re.Match) -> str:
        left, col = m.group(1), m.group(2)
        ll = left.lower()
        if ll in alias_to_table:
            return f"{alias_to_table[ll]}.{col}"
        return m.group(0)
    lint_view = QUAL_COL_RE.sub(alias_to_table_repl, s)
    unk_tbls, unk_cols = lint_unknown_tables_and_columns(lint_view, schema)
    if unk_tbls:
        return s, f"unknown tables: {', '.join(sorted(set(unk_tbls)))}"
    if unk_cols:
        return s, f"unknown columns: {', '.join(sorted(set(unk_cols)))}"

    err = try_explain(conn, s, params or [])
    if err:
        return s, f"EXPLAIN failed: {err}"

    _, run_err = dry_run_sample(conn, s, params or [])
    if run_err:
        return s, f"dry-run failed: {run_err}"

    return s, None

def guarded_select(
    candidate_sql: str,
    params: Optional[List[Any]],
    conn: sqlite3.Connection,
    max_repair_loops: int = 1,
    planner_callback = None,
) -> Tuple[Optional[List[Dict[str, Any]]], str, Optional[str]]:
    schema = get_schema_map(conn)
    sql = candidate_sql
    last_err = None

    for _ in range(max_repair_loops + 1):
        fixed, err = repair_sql(sql, conn, schema, params=params)
        _dbg("after_repair", fixed)
        if not err:
            try:
                cur = conn.cursor()
                cur.execute(fixed, params or [])
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                return [dict(zip(cols, r)) for r in rows], fixed, None
            except Exception as e:
                last_err = f"execution failed: {e}"
        else:
            last_err = err

        if planner_callback:
            hint = (
                f"SQLite validation/exec error: {last_err}\n\n"
                f"SQL:\n{sql}\n\n"
                "Return corrected SQL only."
            )
            new_sql = planner_callback(hint)
            if new_sql and new_sql.strip() != sql.strip():
                sql = strip_code_fences(remove_trailing_semicolon(new_sql))
                continue

        if fixed == sql:
            break
        sql = fixed

    return None, sql, last_err
