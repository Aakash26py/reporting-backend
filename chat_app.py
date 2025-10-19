# chat_app.py
from flask import Blueprint, request, Response, jsonify
from flask_cors import CORS
import sqlite3, json, re, logging, unicodedata, string, difflib
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from sql_guard import guarded_select

chat_bp = Blueprint("chat", __name__)
CORS(chat_bp)

DB_PATH = "chat_history.db"

# ===================== LOGGING =====================
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("chat_app")

# ===================== DB HELPERS =====================

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def ensure_aux_tables():
    """Create aux tables (idempotent)."""
    conn = get_db()
    cur = conn.cursor()
    # session kv
    cur.execute("""
        CREATE TABLE IF NOT EXISTS session_kv (
            session_id INTEGER NOT NULL,
            k TEXT NOT NULL,
            v TEXT NOT NULL,
            PRIMARY KEY (session_id, k)
        )
    """)
    # generic normalization synonyms table
    cur.execute("""
        CREATE TABLE IF NOT EXISTS normalization_synonyms (
            column_name TEXT NOT NULL,
            variant_normalized TEXT NOT NULL,
            canonical_value TEXT NOT NULL,
            PRIMARY KEY (column_name, variant_normalized)
        )
    """)
    conn.commit()
    conn.close()

ensure_aux_tables()

def create_session():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("INSERT INTO sessions (created_at) VALUES (?)", (datetime.now().isoformat(),))
    sid = cur.lastrowid
    conn.commit(); conn.close()
    return sid

def save_message(session_id, role, content):
    conn = get_db()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO messages (session_id, role, content, created_at) VALUES (?,?,?,?)",
        (session_id, role, content, datetime.now().isoformat()),
    )
    conn.commit(); conn.close()

def get_schema_snapshot() -> Dict[str, List[str]]:
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")
    tables = [r[0] for r in cur.fetchall()]
    schema = {}
    for t in tables:
        cur.execute(f"PRAGMA table_info({t})")
        cols = [row[1] for row in cur.fetchall()]
        schema[t] = cols
    conn.close()
    return schema

def schema_text(schema: Dict[str, List[str]]) -> str:
    return "\n".join([f"{t}: {', '.join(cols)}" for t, cols in schema.items()])

# ---------- session kv ----------
def set_session_kv(session_id: int, k: str, v: Any):
    conn = get_db(); cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO session_kv (session_id,k,v) VALUES (?,?,?)",
        (session_id, k, json.dumps(v, ensure_ascii=False))
    )
    conn.commit(); conn.close()

def get_session_kv(session_id: int, k: str):
    conn = get_db(); cur = conn.cursor()
    cur.execute("SELECT v FROM session_kv WHERE session_id=? AND k=?", (session_id, k))
    row = cur.fetchone(); conn.close()
    return json.loads(row["v"]) if row else None

def get_recent_history(session_id: int, n: int = 8) -> List[Dict[str, str]]:
    """Oldest → newest, last n turns."""
    conn = get_db(); cur = conn.cursor()
    cur.execute("""
        SELECT role, content FROM messages
        WHERE session_id=? ORDER BY id DESC LIMIT ?
    """, (session_id, n))
    rows = cur.fetchall(); conn.close()
    return [{"role": r["role"], "content": r["content"]} for r in rows][::-1]

# ===================== LLM SETUP =====================

llm = ChatOllama(model="llama3", temperature=0.0)
parser = StrOutputParser()

intent_prompt = ChatPromptTemplate.from_messages([
    ("system", "Classify the user request as exactly one token: 'sql' or 'chat'. Return only that token."),
    ("human", "{question}")
])
intent_chain = intent_prompt | llm | parser

plan_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert SQL planner for SQLite. Output STRICT JSON only, with keys:\n"
        "- aggregation: one of [SUM, AVG, MAX, MIN, COUNT, NONE]\n"
        "- metric: one of [amount, interest, *] (* allowed only with COUNT)\n"
        "- table: one of [loans, join_loans_customers]\n"
        "- group_by: one of [none, branch, status, month, customer_name]\n"
        "- filters: array of objects with keys: column (like loans.branch), op ('eq' only), value (string)\n"
        "- top_k: integer or null\n"
        "Rules:\n"
        "• Use ONLY tables/columns in the schema.\n"
        "• For month grouping, use strftime('%Y-%m', loans.disbursed) AS month.\n"
        "• If using customer names, table must be join_loans_customers (JOIN customers on id).\n"
        "Return ONLY JSON, no markdown."
    ),
    ("human", "Schema:\n{schema}\n\nUser question:\n{question}\n\nJSON:")
])
plan_chain = plan_prompt | llm | parser

# --- requirements extraction (model-driven) ---
requirements_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a requirements extractor. Given a database schema and a user question, "
     "infer what must be present in the SQL result columns to answer directly.\n"
     "Return STRICT JSON with keys:\n"
     "- required_columns: array of column names that SHOULD be present in the final SELECT output "
     "  to answer the question (e.g., ['customer_name','amount','id']). Prefer human-readable labels if the question involves a person.\n"
     "- preferred_order: one of ['desc','asc',null]\n"
     "- order_by_hint: best column for ordering (e.g., 'amount' or 'disbursed'), else null\n"
     "- k: integer top-k (1 for singular questions; small number like 5 otherwise; null if not applicable)\n"
     "- filters_hint: array of free-text normalized filters, e.g., ['loans.branch = Mumbai'] or []\n"
     "No commentary. JSON only."
    ),
    ("human", "Schema:\n{schema}\n\nQuestion:\n{question}\n\nJSON:")
])
requirements_chain = requirements_prompt | llm | parser

# --- conversation context resolver (model-driven) ---
context_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a conversation context resolver for SQL QA over a DB. "
     "Use the chat history, the latest question, and the actual last SQL rows (if any) "
     "to resolve references and derive constraints.\n"
     "Return STRICT JSON with keys:\n"
     "- clarified_question: string (rewrite of the latest user question using resolved context)\n"
     "- required_columns: array of output columns needed to answer (prefer human labels if about people)\n"
     "- filters_hint: array of normalized equality filters like 'loans.branch = Mumbai' or 'loans.amount = 12345.67'\n"
     "- order_by_hint: string or null (column to sort by)\n"
     "- preferred_order: 'asc'|'desc'|null\n"
     "- k: integer or null (top-k needed)\n"
     "Rules: Be faithful to the chat; prefer constraints grounded in last_rows. No prose; JSON only."
    ),
    ("human",
     "Schema tables and columns:\n{schema}\n\n"
     "Recent conversation turns (oldest → newest):\n{history}\n\n"
     "Last SQL rows (JSON):\n{last_rows}\n\n"
     "Latest user message:\n{question}\n\n"
     "Return JSON:")
])
context_chain = context_prompt | llm | parser

# --- critique/revision (optional single pass) ---
critique_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are a SQL plan critic. Judge if the SQL result columns would directly answer the question. "
     "Return STRICT JSON with keys: \n"
     "- answers_question: true/false\n"
     "- missing_columns: array of column names that are required but not selected\n"
     "- revision_advice: short text on how to edit the plan (e.g., 'join customers to select c.name as customer_name')."
    ),
    ("human",
     "Question:\n{question}\n\n"
     "Schema:\n{schema}\n\n"
     "Candidate SQL (only):\n{sql}\n\n"
     "Expected/result columns to appear if possible:\n{required_columns}\n\nJSON:")
])
critique_chain = critique_prompt | llm | parser

# ===================== NORMALIZATION (GENERIC, DATA-DRIVEN) =====================

# Map normalized columns to source tables to harvest candidate canonicals for fuzzy matching
COLUMN_CANDIDATE_SOURCES: Dict[str, List[Tuple[str, str]]] = {
    # column -> list of (table, column)
    "loans.branch": [("loans", "branch")],
    "customers.name": [("customers", "name")],
    "customers.segment": [("customers", "segment")],
    "customers.occupation": [("customers", "occupation")],
    "loans.status": [("loans", "status")],
}

PUNCT_TABLE = str.maketrans({c: " " for c in string.punctuation})

def _pre_normalize_text(s: str) -> str:
    # Robust, language-agnostic pre-normalization
    s = unicodedata.normalize("NFKC", s or "")
    s = s.casefold()
    s = s.translate(PUNCT_TABLE)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_filter_column(col: str) -> str:
    """
    Map ambiguous column tokens to fully-qualified ones, without hardcoding domain values.
    """
    col = (col or "").strip()
    if "." in col:
        t, c = col.split(".", 1)
        return f"{t.strip().lower()}.{c.strip().lower()}"
    base = col.lower()
    # Common intents
    if base in {"branch", "city", "location", "branch_name"}:
        return "loans.branch"
    if base in {"name", "customer", "customer_name"}:
        return "customers.name"
    if base in {"segment"}:
        return "customers.segment"
    if base in {"occupation", "job", "role"}:
        return "customers.occupation"
    if base in {"status"}:
        return "loans.status"
    if base in {"amount","interest","disbursed","customer_id","id","age"}:
        return f"loans.{base}" if base != "age" else "customers.age"
    return col.lower()

def normalize_value(column_fq: str, value: Any, conn: sqlite3.Connection) -> Any:
    """
    Generic, data-driven normalization for equality filters.
    - Uses normalization_synonyms table first.
    - Falls back to fuzzy match against distinct values in source tables.
    - Leaves numeric-like values untouched.
    """
    if value is None:
        return value

    # Numeric? return as-is
    if isinstance(value, (int, float)):
        return value
    if isinstance(value, str) and re.fullmatch(r"[+-]?\d+(\.\d+)?", value.strip()):
        return value.strip()

    if not isinstance(value, str):
        return value

    norm = _pre_normalize_text(value)

    # Try synonyms
    cur = conn.cursor()
    cur.execute("""
        SELECT canonical_value
        FROM normalization_synonyms
        WHERE column_name = ? AND variant_normalized = ?
    """, (column_fq, norm))
    row = cur.fetchone()
    if row:
        return row["canonical_value"]

    # Fuzzy fallback: fetch candidates from data
    sources = COLUMN_CANDIDATE_SOURCES.get(column_fq, [])
    candidates: List[str] = []
    for t, c in sources:
        try:
            cur.execute(f"SELECT DISTINCT {c} AS v FROM {t} WHERE {c} IS NOT NULL")
            candidates.extend([r["v"] for r in cur.fetchall() if r["v"]])
        except Exception:
            pass

    # score using difflib (built-in, no external deps)
    if candidates:
        # build normalized map to original canonical
        norm_map = { _pre_normalize_text(x): x for x in candidates if isinstance(x, str) }
        best = difflib.get_close_matches(norm, list(norm_map.keys()), n=1, cutoff=0.86)
        if best:
            return norm_map[best[0]]

    # Default: return original (strip extra spaces, title for human columns)
    # If column is textual, title-case; else keep as cleaned original
    if column_fq in COLUMN_CANDIDATE_SOURCES or column_fq.endswith((".name",".branch",".segment",".occupation",".status")):
        return value.strip()
    return value

# ===================== UTILITIES (FORMATTING & GUARDS) =====================
RUPEE = "₹"
def inr(x):
    try:
        return f"{RUPEE}{float(x):,.2f}"
    except Exception:
        return str(x)

def safe_column(col: str, schema: Dict[str, List[str]]) -> bool:
    if col in ("month", "customer_name"):
        return True
    if "." not in col:
        return any(col in cols for cols in schema.values())
    t, c = col.split(".", 1)
    return t in schema and c in schema[t]

def drop_unknown_filters(plan: Dict[str,Any], schema: Dict[str, List[str]]) -> Dict[str,Any]:
    kept = []
    for f in plan.get("filters", []):
        col = f.get("column","")
        col_full = col if "." in col else f"loans.{col}"
        if safe_column(col_full, schema):
            kept.append(f)
    plan["filters"] = kept
    return plan

def validate_plan(plan: Dict[str, Any], schema: Dict[str, List[str]]) -> Tuple[bool, str]:
    try:
        if plan.get("aggregation") not in ["SUM","AVG","MAX","MIN","COUNT","NONE"]:
            return False, "Bad aggregation"
        if plan.get("metric") not in ["amount","interest","*"]:
            return False, "Bad metric"
        if plan.get("table") not in ["loans","join_loans_customers"]:
            return False, "Bad table"
        if plan.get("group_by") not in ["none","branch","status","month","customer_name"]:
            return False, "Bad group_by"

        for f in plan.get("filters", []):
            if f.get("op") != "eq":
                return False, "Only 'eq' filter supported"
            col = f.get("column","")
            if not safe_column(col if "." in col else f"loans.{col}", schema):
                return False, f"Unknown filter column: {col}"

        if plan["aggregation"] != "COUNT" and plan["metric"] not in ["*", None, ""]:
            metric_col = f"loans.{plan['metric']}" if "." not in plan["metric"] else plan["metric"]
            if not safe_column(metric_col, schema):
                return False, f"Unknown metric column: {metric_col}"

        return True, ""
    except Exception as e:
        return False, f"Plan validation error: {e}"

def simplify_filters(filters: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Prefer stable keys; avoid brittle amount constraints."""
    has_lid = any((f.get("column","").lower() in {"l.id","loans.id"}) for f in filters)
    if has_lid:
        return [f for f in filters if (f.get("column","").lower() in {"l.id","loans.id"})]
    has_cid = any((f.get("column","").lower() in {"l.customer_id","loans.customer_id","c.id","customers.id"}) for f in filters)
    if has_cid:
        kept = []
        for f in filters:
            col = (f.get("column","") or "").lower()
            if col in {"l.customer_id","loans.customer_id","c.id","customers.id"} or col.endswith(".branch"):
                kept.append(f)
        return kept or filters
    return filters

def columns_from_requirements(req: Dict[str, Any], schema: Dict[str, List[str]]) -> List[str]:
    """Pass-through extra columns requested by the model if they exist."""
    out: List[str] = []
    for col in (req.get("required_columns") or []):
        if not isinstance(col, str):
            continue
        if "." not in col:
            continue
        t, c = col.split(".", 1)
        tl, cl = t.strip().lower(), c.strip().lower()
        if tl in {"customers","c"} and "customers" in schema and cl in schema["customers"]:
            out.append(f"c.{cl} AS {cl}")
        elif tl in {"loans","l"} and "loans" in schema and cl in schema["loans"]:
            out.append(f"l.{cl} AS {cl}")
    return out

def coerce_filters_generic(plan: Dict[str, Any], conn: sqlite3.Connection) -> None:
    """Normalize all equality filters generically via synonyms/fuzzy matching."""
    new_filters = []
    for f in plan.get("filters", []):
        col = normalize_filter_column(f.get("column",""))
        val = f.get("value")
        # normalize value using DB-driven synonyms+fuzzy
        val_norm = normalize_value(col, val, conn)
        new_filters.append({"column": col, "op": "eq", "value": val_norm})
    plan["filters"] = new_filters

# --- derive follow-up filters from last_rows (use stable keys)
def derive_followup_filters(latest_msg: str, last_rows: List[Dict[str, Any]]) -> List[str]:
    out: List[str] = []
    if not last_rows:
        return out
    txt = (latest_msg or "").lower()
    if any(k in txt for k in ("age", "how old", "who is", "tell me about", "details")):
        r = last_rows[0]
        cid = r.get("customer_id") or r.get("c.id")
        if cid is not None:
            out.append(f"customers.id = {cid}")
        br = r.get("branch")
        if isinstance(br, str) and br.strip():
            out.append(f"loans.branch = {br.strip()}")
    return out

# ===================== ENFORCE/RECONCILE =====================

def enforce_requirements(plan: Dict[str,Any], req: Dict[str,Any]) -> Dict[str,Any]:
    required = [c.lower() for c in (req.get("required_columns") or []) if isinstance(c, str)]
    want_name = any(k in required for k in ("customer_name","name","c.name"))
    if want_name:
        plan["table"] = "join_loans_customers"
        if plan.get("aggregation","NONE") != "NONE" and plan.get("group_by") not in ("customer_name",):
            plan["aggregation"] = "NONE"
            order_col = (req.get("order_by_hint") or "amount").strip()
            order_dir = (req.get("preferred_order") or "desc").lower()
            plan["_order"] = f"{order_col}_{'desc' if order_dir not in ('asc','desc') else order_dir}"
            plan["top_k"] = plan.get("top_k") or (req.get("k") or 1)
    return plan

def reconcile_filters_and_from(plan: Dict[str, Any]) -> Dict[str, Any]:
    filters = plan.get("filters") or []
    if any((f.get("column","").split(".",1)[0].lower() == "customers") for f in filters):
        plan["table"] = "join_loans_customers"
    return plan

# ===================== SQL COMPILATION & OUTPUT =====================

def compile_sql(plan: Dict[str, Any]) -> Tuple[str, List[Any]]:
    table = plan["table"]
    agg   = plan["aggregation"]
    metric= plan["metric"]
    group = plan["group_by"]
    filters = plan.get("filters", [])
    top_k = plan.get("top_k")
    order_hint = plan.get("_order")

    if table == "join_loans_customers":
        from_sql = "FROM loans l JOIN customers c ON c.id = l.customer_id"
    else:
        from_sql = "FROM loans l"

    select_cols, group_cols = [], []

    if group == "month":
        select_cols.append("strftime('%Y-%m', l.disbursed) AS month")
        group_cols.append("month")
    elif group == "branch":
        select_cols.append("l.branch"); group_cols.append("l.branch")
    elif group == "status":
        select_cols.append("l.status"); group_cols.append("l.status")
    elif group == "customer_name":
        if table != "join_loans_customers":
            from_sql = "FROM loans l JOIN customers c ON c.id = l.customer_id"
        select_cols.append("c.name AS customer_name")
        group_cols.append("customer_name")

    agg_expr = None
    if agg == "COUNT":
        agg_expr = "COUNT(*) AS value"
    elif agg == "NONE":
        agg_expr = None
    else:
        metric_col = f"l.{metric}" if "." not in metric else metric
        agg_expr = f"{agg}({metric_col}) AS value"

    if agg_expr:
        select_sql = "SELECT " + (", ".join(select_cols + [agg_expr]) if select_cols else agg_expr)
    else:
        base_cols = ["l.id","l.branch","l.amount","l.interest","l.status","l.disbursed"]
        if table == "join_loans_customers":
            base_cols.insert(1, "c.id AS customer_id")
            base_cols.insert(2, "c.name AS customer_name")
        extra_selects = plan.get("_extra_selects") or []
        select_sql = "SELECT " + ", ".join(base_cols + select_cols + extra_selects)

    where, params = [], []
    for f in filters:
        col = f.get("column","")
        col = col if "." in col else f"l.{col}"
        where.append(f"{col} = ?")
        params.append(f.get("value"))
    where_sql = f"WHERE {' AND '.join(where)}" if where else ""

    group_sql = f"GROUP BY {', '.join(group_cols)}" if (agg_expr and group_cols) else ""

    order_sql = ""
    if agg_expr and agg in ["SUM","AVG","MAX","MIN","COUNT"]:
        order_sql = "ORDER BY value DESC"
    elif agg == "NONE":
        if order_hint == "amount_desc":
            order_sql = "ORDER BY l.amount DESC"
        elif order_hint == "amount_asc":
            order_sql = "ORDER BY l.amount ASC"
        elif isinstance(order_hint, str) and order_hint.endswith("_desc"):
            col = order_hint[:-5]
            col = col if "." in col else f"l.{col}"
            order_sql = f"ORDER BY {col} DESC"
        elif isinstance(order_hint, str) and order_hint.endswith("_asc"):
            col = order_hint[:-4]
            col = col if "." in col else f"l.{col}"
            order_sql = f"ORDER BY {col} ASC"

    lim = top_k if isinstance(top_k, int) and top_k > 0 else (50 if agg == "NONE" else None)
    limit_sql = f"LIMIT {lim}" if lim else ""

    sql = " ".join([select_sql, from_sql, where_sql, group_sql, order_sql, limit_sql]).strip()
    return sql, params

def format_person_row(r: Dict[str, Any]) -> str:
    name = r.get("customer_name") or r.get("name") or "Unknown"
    parts = [name]
    if r.get("branch"):
        parts.append(f"({r['branch']})")
    line1 = " ".join(parts)
    details = []
    if r.get("amount") is not None:
        details.append(f"Amount {inr(r['amount'])}")
    if r.get("id"):
        details.append(f"Loan ID {r['id']}")
    if r.get("age") is not None:
        try:
            details.append(f"Age {int(r['age'])}")
        except Exception:
            pass
    if r.get("segment"):
        details.append(f"Segment {r['segment']}")
    if r.get("occupation"):
        details.append(f"Occupation {r['occupation']}")
    return line1 + (": " + "; ".join(details) if details else "")

def verbalize(plan: Dict[str,Any], rows: List[Dict[str,Any]]) -> str:
    if not rows:
        return "I couldn’t find matching records for that query."
    if plan.get("aggregation") == "NONE" and len(rows) == 1:
        return format_person_row(rows[0])

    agg = (plan or {}).get("aggregation")
    group = (plan or {}).get("group_by")

    if agg in ["SUM","AVG","MAX","MIN","COUNT"] and group == "none" and "value" in rows[0]:
        val = rows[0]["value"]
        txt = inr(val) if agg != "COUNT" else f"{int(val)}"
        return f"{agg.title()} of {plan['metric']} is {txt}."

    if agg in ["SUM","AVG","COUNT"] and "value" in rows[0]:
        label_key = {"branch":"branch","status":"status","month":"month","customer_name":"customer_name"}.get(group)
        lines = []
        for r in rows[:8]:
            lab = r.get(label_key, "—")
            val = r.get("value")
            display = inr(val) if agg != "COUNT" else f"{int(val)}"
            lines.append(f"- {lab}: {display}")
        header = f"{agg.title()} by {group.replace('_',' ')}:"
        return header + "\n" + "\n".join(lines)

    head = rows[:5]
    return json.dumps(head, ensure_ascii=False, indent=2)

# ===================== GENERIC HELPERS =====================

def make_diverse_plan_prompts(question: str, schema: str, n: int = 4) -> List[Dict[str, Any]]:
    return [{"question": question, "schema": schema} for _ in range(max(1, n))]

def compile_and_score_candidates(
    candidates: List[Tuple[Dict[str,Any], str, List[Any]]],
    req_cols: List[str],
    executor
) -> Tuple[Optional[List[Dict[str,Any]]], Optional[str], Optional[str], Optional[Dict[str,Any]]]:
    best = None
    for plan, sql, params in candidates:
        rows, final_sql, err = executor(sql, params)
        if err:
            score = -10
        else:
            has_cols = 0
            if rows:
                sample = rows[0]
                cols = set(sample.keys())
                for c in req_cols:
                    if isinstance(c, str):
                        alias = c.split(".")[-1]
                        if alias in cols:
                            has_cols += 1
            score = (3 if rows else 0) + has_cols
        if not best or score > best["score"]:
            best = {"rows": rows, "sql": final_sql, "err": err, "score": score, "plan": plan}
    if not best:
        return None, None, "no candidates", None
    return best["rows"], best["sql"], best["err"], best["plan"]

# ===================== HEURISTIC / FALLBACK =====================

def heuristic_plan(question: str) -> Dict[str,Any]:
    q = question.lower()
    m = re.search(r"(?:from|in)\s+([a-z ]+)\b", q)
    branch = m.group(1).strip() if m else None
    return {
        "aggregation": "NONE",
        "metric": "amount",
        "table": "join_loans_customers",
        "group_by": "none",
        "filters": ([{"column":"loans.branch","op":"eq","value":branch}] if branch else []),
        "top_k": 1,
        "_order": "amount_desc"
    }

def fallback_top1_by_branch(question: str) -> Dict[str, Any]:
    return heuristic_plan(question)

# ===================== SSE ROUTE =====================
@chat_bp.route("/stream")
def stream_chat():
    msg = (request.args.get("message") or "").strip()
    sid = request.args.get("session_id", type=int) or create_session()
    save_message(sid, "user", msg)

    def emit(token: str):
        return f"data: {json.dumps({'token': token})}\n\n"

    def generate():
        yield emit("💭 Thinking...")
        try:
            # 0) intent
            intent = intent_chain.invoke({"question": msg}).strip().lower()
            if intent not in ("sql", "chat"):
                intent = "sql" if re.search(
                    r"\b(sum|total|count|avg|average|max|min|top|highest|lowest|most|per|by|who|what|when|age|customer|branch|loan|interest|amount)\b",
                    msg.lower()
                ) else "chat"

            if intent == "chat":
                chat_prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "You are a banking assistant. Never invent facts. "
                     "Only use information explicitly present in the latest SQL query result or from a fresh SQL query. "
                     "If the user asks for details that are not in the database, reply with: "
                     "“I don’t have that in the database.” Be concise and use INR (₹) for currency."
                     ),
                    ("human", "{q}")
                ])
                answer = (chat_prompt | llm | parser).invoke({"q": msg})
                save_message(sid, "assistant", answer)
                yield emit(answer)
                yield "data: [DONE]\n\n"
                return

            # --- Conversational context ---
            last_rows = get_session_kv(sid, "last_rows") or []
            history   = get_recent_history(sid, n=8)

            schema = get_schema_snapshot()
            schema_txt = schema_text(schema)

            log.info("Schema tables: %s", list(schema.keys()))
            log.info("Customers cols: %s", schema.get("customers"))

            # 1) Model-driven context resolution
            ctx_raw = context_chain.invoke({
                "schema": schema_txt,
                "history": json.dumps(history, ensure_ascii=False, indent=2),
                "last_rows": json.dumps(last_rows, ensure_ascii=False, indent=2),
                "question": msg
            }).strip().replace("```json","").replace("```","")
            try:
                ctx = json.loads(ctx_raw[ctx_raw.find("{"): ctx_raw.rfind("}")+1])
            except Exception:
                ctx = {"clarified_question": msg, "required_columns": [], "filters_hint": [], "order_by_hint": None, "preferred_order": None, "k": None}

            effective_question = ctx.get("clarified_question") or msg

            # 2) Requirements (model-driven) + merge with context
            req_raw = requirements_chain.invoke({"schema": schema_txt, "question": effective_question}).strip()
            req_raw = req_raw.replace("```json","").replace("```","")
            try:
                req = json.loads(req_raw[req_raw.find("{"): req_raw.rfind("}")+1])
            except Exception:
                req = {"required_columns": [], "preferred_order": None, "order_by_hint": None, "k": None, "filters_hint": []}

            # Merge
            req["filters_hint"]     = (req.get("filters_hint") or []) + (ctx.get("filters_hint") or [])
            req["required_columns"] = list({*(req.get("required_columns") or []), *(ctx.get("required_columns") or [])})
            req["order_by_hint"]    = req.get("order_by_hint") or ctx.get("order_by_hint")
            req["preferred_order"]  = req.get("preferred_order") or ctx.get("preferred_order")
            req["k"]                = req.get("k") or ctx.get("k")

            # If user asks for age explicitly, ensure it's requested
            if re.search(r"\b(age|how old|years old)\b", (msg or "").lower()):
                need = set(req.get("required_columns") or [])
                need.update(["customers.age", "customers.name"])
                req["required_columns"] = list(need)

            # Follow-up pinning (e.g., customers.id + branch)
            req["filters_hint"] = (req.get("filters_hint") or []) + derive_followup_filters(msg, last_rows)

            required_columns: List[str] = req.get("required_columns") or []
            preferred_order  = req.get("preferred_order")
            order_by_hint    = req.get("order_by_hint")
            top_k_hint       = req.get("k")
            filters_hint     = req.get("filters_hint") or []

            log.info("Context resolved: %s", ctx)
            log.info("Requirements: required_columns=%s order_by=%s preferred_order=%s k=%s",
                     req.get("required_columns"), req.get("order_by_hint"), req.get("preferred_order"), req.get("k"))

            # 3) Build diverse candidate plans
            diverse_specs = make_diverse_plan_prompts(effective_question, schema_txt, n=4)
            plans: List[Tuple[Dict[str,Any], str, List[Any]]] = []

            for _spec in diverse_specs:
                raw_plan = plan_chain.invoke({"schema": schema_txt, "question": effective_question})
                raw_plan = raw_plan.strip().replace("```json","").replace("```","")
                if "{" in raw_plan and "}" in raw_plan:
                    raw_plan = raw_plan[raw_plan.find("{"): raw_plan.rfind("}")+1]
                try:
                    plan = json.loads(raw_plan)
                except Exception:
                    plan = heuristic_plan(effective_question)

                if isinstance(top_k_hint, int):
                    plan["top_k"] = top_k_hint
                if order_by_hint and plan.get("aggregation") == "NONE":
                    if (preferred_order or "").lower() == "desc":
                        plan["_order"] = f"{order_by_hint}_desc"
                    elif (preferred_order or "").lower() == "asc":
                        plan["_order"] = f"{order_by_hint}_asc"

                # Merge filters
                existing = plan.get("filters", [])
                merged: dict[tuple, dict] = {}

                def _norm_col(c: str) -> str:
                    return normalize_filter_column(c)

                for f in existing:
                    col = _norm_col(f.get("column", ""))
                    val = f.get("value", "")
                    merged[(col, val)] = {"column": col, "op": "eq", "value": val}

                for ftxt in filters_hint:
                    if not isinstance(ftxt, str):
                        continue
                    m = re.match(r"\s*([A-Za-z0-9_.]+)\s*=\s*(.+)", ftxt)
                    if not m:
                        continue
                    col = _norm_col(m.group(1))
                    val = m.group(2).strip().strip('"').strip("'")
                    merged[(col, val)] = {"column": col, "op": "eq", "value": val}

                plan["filters"] = list(merged.values())

                # align and reconcile
                plan = enforce_requirements(plan, req)
                plan = reconcile_filters_and_from(plan)

                # GENERIC normalization for all filters (no hardcoded values)
                conn = get_db()
                coerce_filters_generic(plan, conn)
                conn.close()

                # drop unknowns, simplify
                plan = drop_unknown_filters(plan, schema)
                plan["filters"] = simplify_filters(plan.get("filters", []))

                # pass-through extras (e.g., customers.age)
                plan["_extra_selects"] = columns_from_requirements(req, schema)

                ok, _why = validate_plan(plan, schema)
                if not ok:
                    continue

                # prune filters for tables not present in FROM
                from_tables = ["loans"] + (["customers"] if plan.get("table") == "join_loans_customers" else [])
                kept = []
                for f in plan.get("filters") or []:
                    left = f.get("column","")
                    t = left.split(".",1)[0].lower() if "." in left else "loans"
                    if t in from_tables:
                        kept.append(f)
                plan["filters"] = kept

                sql, params = compile_sql(plan)
                log.info("Candidate plan: %s", plan)
                log.info("SQL: %s", sql)
                log.info("Params: %s", params)
                plans.append((plan, sql, params))

            if not plans:
                fb = fallback_top1_by_branch(effective_question)
                conn = get_db()
                coerce_filters_generic(fb, conn)
                conn.close()
                fb = drop_unknown_filters(fb, schema)
                fb["filters"] = simplify_filters(fb.get("filters", []))
                fb["_extra_selects"] = columns_from_requirements(req, schema)
                ok_fb, _ = validate_plan(fb, schema)
                if ok_fb:
                    sql_fb, params_fb = compile_sql(fb)
                    log.info("Fallback plan: %s", fb)
                    log.info("SQL: %s", sql_fb)
                    log.info("Params: %s", params_fb)
                    plans = [(fb, sql_fb, params_fb)]
                else:
                    err = "Could not build a safe plan."
                    save_message(sid, "assistant", err); yield emit(err); yield "data: [DONE]\n\n"; return

            # 4) Execution-guided selection
            def exec_with_guard(sql, params):
                conn = get_db()
                rows, final_sql, err = guarded_select(sql, params, conn, max_repair_loops=1, planner_callback=None)
                conn.close()
                return rows, final_sql, err

            rows, final_sql, err, chosen_plan = compile_and_score_candidates(plans, req.get("required_columns", []), exec_with_guard)
            log.info("Rows: %d; Columns: %s", len(rows or []), (list(rows[0].keys()) if rows else []))

            # 5) Optional critique & single revision if needed
            need_revision = False
            if not err and rows:
                sample_cols = set(rows[0].keys())
                missing = []
                for c in req.get("required_columns") or []:
                    if isinstance(c, str) and (c.split(".")[-1] not in sample_cols):
                        missing.append(c)
                need_revision = len(missing) > 0
            else:
                need_revision = True

            if need_revision:
                crit_raw = critique_chain.invoke({
                    "question": effective_question,
                    "schema": schema_txt,
                    "sql": (final_sql or plans[0][1]),
                    "required_columns": json.dumps(req.get("required_columns") or [], ensure_ascii=False)
                }).strip().replace("```json","").replace("```","")
                try:
                    crit = json.loads(crit_raw[crit_raw.find("{"): crit_raw.rfind("}")+1])
                except Exception:
                    crit = {"answers_question": False, "missing_columns": req.get("required_columns") or [], "revision_advice": ""}

                hint = (
                    f"Revise the plan to ensure columns {crit.get('missing_columns') or req.get('required_columns')} appear in SELECT. "
                    f"{crit.get('revision_advice','')}"
                )
                raw_plan2 = plan_chain.invoke({"schema": schema_txt, "question": f"{effective_question}\n\nHINT: {hint}"})
                raw_plan2 = raw_plan2.strip().replace("```json","").replace("```","")
                try:
                    plan2 = json.loads(raw_plan2[raw_plan2.find("{"): raw_plan2.rfind("}")+1])
                except Exception:
                    plan2 = heuristic_plan(effective_question)

                plan2 = enforce_requirements(plan2, req)
                plan2 = reconcile_filters_and_from(plan2)

                conn = get_db()
                coerce_filters_generic(plan2, conn)
                conn.close()

                plan2 = drop_unknown_filters(plan2, schema)
                plan2["filters"] = simplify_filters(plan2.get("filters", []))
                plan2["_extra_selects"] = columns_from_requirements(req, schema)

                ok2, _ = validate_plan(plan2, schema)
                if ok2:
                    from_tables2 = ["loans"] + (["customers"] if plan2.get("table") == "join_loans_customers" else [])
                    kept2 = []
                    for f in plan2.get("filters") or []:
                        left = f.get("column","")
                        t = left.split(".",1)[0].lower() if "." in left else "loans"
                        if t in from_tables2:
                            kept2.append(f)
                    plan2["filters"] = kept2

                    sql2, params2 = compile_sql(plan2)
                    rows2, final_sql2, err2 = exec_with_guard(sql2, params2)
                    log.info("Revised SQL: %s", final_sql2)
                    log.info("Revised Params: %s", params2)
                    log.info("Revised rows: %d; Columns: %s", len(rows2 or []), (list(rows2[0].keys()) if rows2 else []))
                    if not err2 and rows2:
                        cols2 = set(rows2[0].keys())
                        miss2 = []
                        for c in req.get("required_columns") or []:
                            if isinstance(c, str) and (c.split(".")[-1] not in cols2):
                                miss2.append(c)
                        if len(miss2) == 0 or (not rows):
                            rows, final_sql, err, chosen_plan = rows2, final_sql2, err2, plan2

            # 6) Finalize
            if err:
                log.exception("Execution error: %s", err)
                msg_err = f"❌ SQL failed: {err}\nSQL: {final_sql}\n"
                save_message(sid, "assistant", msg_err); yield emit(msg_err); yield "data: [DONE]\n\n"; return

            if re.search(r"\b(age|how old|years old)\b", (msg or "").lower()):
                has_age = bool(rows and 'age' in rows[0])
                log.warning("Age requested; selected_age_col=%s", has_age)

            answer = verbalize(chosen_plan or {}, rows or [])
            final = f"{answer}\n\n(SQL: {final_sql} · {len(rows or [])} rows)"
            save_message(sid, "assistant", final)

            set_session_kv(sid, "last_rows", rows or [])
            set_session_kv(sid, "last_plan", chosen_plan or {})

            yield emit(final)

        except Exception as e:
            log.exception("Fatal error in stream_chat: %s", e)
            yield emit(f"❌ Error: {e}")
        yield "data: [DONE]\n\n"
    return Response(generate(), mimetype="text/event-stream")

# ===================== SUPPORT ROUTES =====================
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
