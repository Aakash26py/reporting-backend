from flask import Blueprint, request, Response, jsonify
from flask_cors import CORS
import json, logging, traceback
from datetime import datetime

from sql_guard import guarded_select
from llm_logic import LlmOrchestrator
from schema_context import get_schema_context

chat_bp = Blueprint("chat", __name__)
CORS(chat_bp)

DB_PATH = "chat_history.db"
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("chat")

from dotenv import load_dotenv
import os

load_dotenv()
MODEL = os.getenv("MODEL")



# ---------- Utility DB ops ----------
import sqlite3
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def save_message(session_id, role, content):
    conn = get_db(); cur = conn.cursor()
    cur.execute("INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
                (session_id, role, content, datetime.now().isoformat()))
    conn.commit(); conn.close()

def create_session():
    conn = get_db(); cur = conn.cursor()
    cur.execute("INSERT INTO sessions (created_at) VALUES (?)", (datetime.now().isoformat(),))
    sid = cur.lastrowid
    conn.commit(); conn.close()
    return sid


# ---------- Orchestrator ----------
ORCH = LlmOrchestrator(
    sql_model=MODEL,
    answer_model=MODEL,
    sql_temperature=0.1,
    answer_temperature=0.3,
    num_ctx=8192,
)


# ---------- SSE Chat ----------
@chat_bp.route("/stream")
def stream_chat():
    msg = (request.args.get("message") or "").strip()
    sid = request.args.get("session_id", type=int) or create_session()
    save_message(sid, "user", msg)

    def sse(data: str):
        return f"data: {json.dumps({'token': data, 'session_id': sid})}\n\n"

    def generate():
        yield sse("💭 Generating SQL...")
        try:
            ctx = get_schema_context()
            schema_text = ctx["schema_text"]

            # feed both descriptions + enums
            context = {
                "enums": ctx["enums"],
                "descriptions": ctx["descriptions"]
            }

            sql, err = ORCH.generate_sql_with_retries(
                session_id=sid,
                question=msg,
                schema_text=schema_text,
                context_obj=context,
                validate_fn=lambda q: validator_explain_and_dryrun(q),
                trace_fn=lambda s, p, r, e: log.info("[%s] %s", s, e or r)
            )

            if err:
                yield sse(f"❌ SQL generation failed: {err}")
                yield "data: [DONE]\n\n"
                return

            yield sse(f"🧪 Running SQL...\n\n```sql\n{sql}\n```")
            conn = get_db()
            rows, final_sql, exec_err = guarded_select(sql, [], conn, max_repair_loops=1, planner_callback=None)
            conn.close()

            if exec_err:
                yield sse(f"❌ SQL failed: {exec_err}")
                yield "data: [DONE]\n\n"
                return

            yield sse("🧾 Drafting answer...")
            answer = ORCH.generate_answer_from_rows(
                session_id=sid,
                question=msg,
                sql_used=final_sql,
                rows=rows or [],
                trace_fn=lambda s, p, r, e: log.info("[%s] %s", s, e or r)
            )
            # yield sse(f"{answer}\n\n(SQL: {final_sql} · {len(rows or [])} rows)")
            log.info(f"Final SQL: {final_sql}")
            yield sse(f"{answer}")

        except Exception as e:
            tb = traceback.format_exc()
            yield sse(f"❌ Error: {e}\n{tb}")

        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype="text/event-stream")

@chat_bp.route("/sessions")
def sessions():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT id, created_at FROM sessions ORDER BY id DESC")
    data = [{"id": r[0], "created_at": r[1]} for r in cur.fetchall()]
    conn.close()
    return jsonify(data)



# ---------- Validation ----------
def validator_explain_and_dryrun(sql: str):
    conn = get_db()
    rows, final_sql, err = guarded_select(sql, [], conn, max_repair_loops=1, planner_callback=None)
    conn.close()
    return (err is None, err, final_sql or sql)
