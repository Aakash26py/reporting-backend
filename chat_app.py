# api_app.py
from flask import Blueprint, request, Response, jsonify
from flask_cors import CORS
import sqlite3, json, logging, traceback
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

from sql_guard import guarded_select
from llm_logic import LlmOrchestrator

# ---------- App / CORS ----------
chat_bp = Blueprint("chat", __name__)
CORS(chat_bp)

# ---------- Config ----------
DB_PATH = "chat_history.db"
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("chat")

# ---------- DB helpers ----------
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    return conn

def ensure_tables():
    conn = get_db(); cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS sessions(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS messages(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id INTEGER NOT NULL,
          role TEXT NOT NULL,
          content TEXT NOT NULL,
          created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS session_kv(
          session_id INTEGER NOT NULL,
          k TEXT NOT NULL,
          v TEXT NOT NULL,
          PRIMARY KEY(session_id,k)
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS llm_traces(
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          session_id INTEGER,
          stage TEXT NOT NULL,
          prompt TEXT,
          response TEXT,
          error TEXT,
          created_at TEXT NOT NULL
        )
    """)

    conn.commit(); conn.close()

ensure_tables()

def create_session() -> int:
    conn = get_db(); cur = conn.cursor()
    cur.execute("INSERT INTO sessions(created_at) VALUES(?)", (datetime.now().isoformat(),))
    sid = cur.lastrowid
    conn.commit(); conn.close()
    log.info("Created new session id=%s", sid)
    return sid

def validate_or_create_sid(sid_in: Optional[int]) -> int:
    if not sid_in:
        return create_session()
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT 1 FROM sessions WHERE id=?", (sid_in,))
    ok = cur.fetchone() is not None
    conn.close()
    if ok:
        return sid_in
    return create_session()

def save_message(session_id: int, role: str, content: str):
    conn = get_db(); cur = conn.cursor()
    cur.execute("INSERT INTO messages(session_id,role,content,created_at) VALUES(?,?,?,?)",
                (session_id, role, content, datetime.now().isoformat()))
    conn.commit(); conn.close()

def set_kv(sid: int, k: str, v: Any):
    conn = get_db(); cur = conn.cursor()
    cur.execute("INSERT OR REPLACE INTO session_kv(session_id,k,v) VALUES(?,?,?)",
                (sid, k, json.dumps(v, ensure_ascii=False)))
    conn.commit(); conn.close()

def get_kv(sid: int, k: str):
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT v FROM session_kv WHERE session_id=? AND k=?", (sid,k))
    row = cur.fetchone(); conn.close()
    return json.loads(row["v"]) if row else None

def get_schema_snapshot() -> Dict[str, List[str]]:
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [r[0] for r in cur.fetchall()]
    schema = {}
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        schema[t] = [r[1] for r in cur.fetchall()]
    conn.close()
    return schema

def log_trace(session_id: Optional[int], stage: str, prompt: Optional[str], response: Optional[str], error: Optional[str]):
    try:
        conn = get_db(); cur = conn.cursor()
        cur.execute("INSERT INTO llm_traces(session_id,stage,prompt,response,error,created_at) VALUES(?,?,?,?,?,?)",
                    (session_id, stage, prompt, response, error, datetime.now().isoformat()))
        conn.commit(); conn.close()
    except Exception:
        log.exception("Failed to log llm trace")

# ---------- Validator wrappers ----------
def validator_explain_and_dryrun(sql: str) -> Tuple[bool, Optional[str], str]:
    """
    Validate SQL via EXPLAIN QUERY PLAN + dry-run sample.
    Returns (ok, err, final_sql).
    """
    conn = get_db()
    rows, final_sql, err = guarded_select(sql, [], conn, max_repair_loops=1, planner_callback=None)
    conn.close()
    if err:
        return False, err, final_sql or sql
    return True, None, final_sql or sql

# ---------- LLM Orchestrator ----------
ORCH = LlmOrchestrator(
    sql_model="qwen2.5:14b-instruct-q4_K_M",
    answer_model="qwen2.5:14b-instruct-q4_K_M",
    sql_temperature=0.1,
    answer_temperature=0.3,
    num_ctx=8192,
)

# ---------- SSE Chat ----------
@chat_bp.route("/stream")
def stream_chat():
    msg = (request.args.get("message") or "").strip()
    sid = validate_or_create_sid(request.args.get("session_id", type=int))
    save_message(sid, "user", msg)

    def sse(token: str) -> str:
        return f"data: {json.dumps({'token': token, 'session_id': sid})}\n\n"

    def generate():
        # let client know session id immediately
        yield f"event: session\ndata: {json.dumps({'session_id': sid})}\n\n"

        # STEP A — Generate SQL (with LLM verifier)
        yield sse("💭 Generating SQL...")
        log.info("[SSE] Step A: Generating SQL for: %r", msg)

        try:
            schema = get_schema_snapshot()
            schema_text = "\n".join([f"{t}: {', '.join(cols)}" for t, cols in schema.items()])
            context = {
                "last_rows": get_kv(sid, "last_rows"),
                "pinned_customer": get_kv(sid, "pinned_customer"),
            }

            sql, err = ORCH.generate_sql_with_retries(
                session_id=sid,
                question=msg,
                schema_text=schema_text,
                context_obj=context,
                validate_fn=validator_explain_and_dryrun,
                trace_fn=lambda stage, p, r, e: log_trace(sid, stage, p, r, e),
            )

            if err:
                log.error("[SSE] Step A failed: %s", err)
                final = f"❌ SQL generation failed after retries.\n{err}"
                save_message(sid, "assistant", final)
                yield sse(final)
                yield "data: [DONE]\n\n"
                return

            log.info("[SSE] Step A OK. SQL:\n%s", sql)
            yield sse(f"🧪 Running SQL...\n\n```sql\n{sql}\n```")

            # STEP B — Execute SQL
            log.info("[SSE] Step B: Executing SQL")
            conn = get_db()
            rows, final_sql, exec_err = guarded_select(sql, [], conn, max_repair_loops=1, planner_callback=None)
            conn.close()
            log_trace(sid, "B-exec", sql, json.dumps(rows or [], ensure_ascii=False), exec_err)

            if exec_err:
                log.error("[SSE] Step B failed: %s", exec_err)
                final = f"❌ SQL failed: {exec_err}\nSQL: {final_sql}"
                save_message(sid, "assistant", final)
                yield sse(final)
                yield "data: [DONE]\n\n"
                return

            log.info("[SSE] Step B OK. Rows=%d", len(rows or []))

            # persist minimal context for follow-ups
            if rows:
                set_kv(sid, "last_rows", rows)
                r0 = rows[0]
                if "customer_id" in r0:
                    set_kv(sid, "pinned_customer", {"customer_id": r0["customer_id"]})

            # STEP C — Draft answer from rows
            yield sse("🧾 Drafting answer...")
            log.info("[SSE] Step C: Drafting answer")

            answer = ORCH.generate_answer_from_rows(
                session_id=sid,
                question=msg,
                sql_used=final_sql,
                rows=rows or [],
                trace_fn=lambda stage, p, r, e: log_trace(sid, stage, p, r, e),
            )
            log.info("[SSE] Step C OK.")

            final = f"{answer}\n\n(SQL: {final_sql} · {len(rows or [])} rows)"
            save_message(sid, "assistant", final)
            yield sse(final)

        except Exception as e:
            tb = traceback.format_exc()
            log_trace(sid, "fatal", None, None, tb)
            log.exception("Fatal in /stream: %s", e)
            yield sse(f"❌ Error: {e}")

        yield "data: [DONE]\n\n"

    resp = Response(generate(), mimetype="text/event-stream")
    resp.headers["Cache-Control"] = "no-cache"
    resp.headers["X-Accel-Buffering"] = "no"
    return resp

# ---------- Extra APIs ----------
@chat_bp.route("/session/new")
def new_session():
    sid = create_session()
    return jsonify({"session_id": sid})

@chat_bp.route("/sessions")
def sessions():
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT id, created_at FROM sessions ORDER BY id DESC")
    data = [{"id": r["id"], "created_at": r["created_at"]} for r in cur.fetchall()]
    conn.close()
    return jsonify(data)

@chat_bp.route("/history/<int:sid>")
def history(sid):
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT role, content, created_at FROM messages WHERE session_id=? ORDER BY id", (sid,))
    rows = [{"role": r["role"], "content": r["content"], "created_at": r["created_at"]} for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)

@chat_bp.route("/traces/<int:sid>")
def traces(sid):
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
      SELECT stage, prompt, response, error, created_at
      FROM llm_traces WHERE session_id=? ORDER BY id
    """, (sid,))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)
