# llm_logic.py
import json, re, traceback, logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

log = logging.getLogger("chat")

FENCE_RE = re.compile(r"(?is)```(?:sql)?(.*?)```")

def extract_sql(text: str) -> str:
    """Prefer fenced SQL; else first SELECT...; else raw trimmed."""
    if not text:
        return ""
    m = FENCE_RE.search(text)
    if m:
        return m.group(1).strip().rstrip(";")
    s = text.strip()
    if s.lower().startswith("select"):
        return s.rstrip(";")
    m2 = re.search(r"(?is)\bselect\b.*", s)
    return (m2.group(0) if m2 else s).strip().rstrip(";")

class LlmOrchestrator:
    """
    Encapsulates:
      - Stage A: LLM → SQL (with LLM self-verification + engine validator; up to 4 retries)
      - Stage C: LLM → human answer from rows
    """

    def __init__(
        self,
        sql_model: str = "qwen2.5:14b-instruct-q4_K_M",
        answer_model: str = "qwen2.5:14b-instruct-q4_K_M",
        sql_temperature: float = 0.1,
        answer_temperature: float = 0.3,
        num_ctx: int = 8192,
    ):
        self.llm_sql = ChatOllama(model=sql_model, temperature=sql_temperature, num_ctx=num_ctx)
        self.llm_answer = ChatOllama(model=answer_model, temperature=answer_temperature, num_ctx=num_ctx)
        self.parser = StrOutputParser()

        self.sql_chain = self._build_sql_chain()
        self.verify_chain = self._build_verify_chain()
        self.answer_chain = self._build_answer_chain()

    # ---------- Prompts ----------
    def _build_sql_chain(self):
        sql_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a SQL expert for SQLite. You receive a database schema, recent context (may be empty), "
             "and a user request.\n"
             "TASK: Return a SINGLE valid SQL SELECT statement for SQLite that answers the request, "
             "using only tables/columns that exist.\n"
             "HARD RULES:\n"
             "• Never invent columns or tables. Do NOT use canonical/guessed names (e.g., 'customers.city') if not in schema.\n"
             "• Prefer including stable identifiers (e.g., customers.id AS customer_id, customers.name AS customer_name) when selecting customer info.\n"
             "• For 'highest/top', ORDER BY the numeric metric DESC and LIMIT appropriately.\n"
             "• If the user references a branch/city, use whichever column exists in the schema (e.g., loans.branch) — do not invent.\n"
             "• Output SQL only. No commentary. No markdown fences."
            ),
            ("human",
             "SCHEMA (tables → columns):\n{schema}\n\n"
             "RECENT CONTEXT (optional JSON):\n{context}\n\n"
             "USER REQUEST:\n{question}\n\n"
             "{validator_feedback}\n"
             "SQL:")
        ])
        return sql_prompt | self.llm_sql | self.parser

    def _build_verify_chain(self):
        verify_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a strict SQLite SQL checker. "
             "Given a schema (authoritative) and a SQL SELECT, respond in STRICT JSON:\n"
             "{\n"
             '  "valid": true|false,\n'
             '  "reasons": ["short reason 1", "short reason 2"],\n'
             '  "suggested_sql": "ONLY if invalid: a corrected single SELECT using ONLY schema columns; else empty string",\n'
             '  "used": {"tables":["t1","t2"], "columns":["t1.c1","t2.c2"]}\n'
             "}\n"
             "Rules:\n"
             "• Treat the provided schema as the ONLY truth. If a column is missing (e.g., customers.city), mark invalid.\n"
             "• If invalid, propose a corrected query (single SELECT) that follows the same intent.\n"
             "• Never add commentary outside JSON."
            ),
            ("human",
             "SCHEMA (tables → columns):\n{schema}\n\n"
             "SQL TO CHECK:\n{sql}\n\n"
             "JSON:")
        ])
        return verify_prompt | self.llm_sql | self.parser

    def _build_answer_chain(self):
        answer_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a banking assistant. Generate a concise, human-friendly answer based ONLY on the SQL rows provided.\n"
             "Rules:\n"
             "• NEVER invent facts. If rows are empty, say you couldn’t find matching records.\n"
             "• Use INR symbol (₹) for currency if amounts appear.\n"
             "• If customer info is present, format like: 'Name (Branch): Amount ₹X; Age Y; Segment Z; (Loan ID ...)' — keep it short.\n"
             "• No markdown tables or code."
            ),
            ("human",
             "USER REQUEST:\n{question}\n\n"
             "SQL USED:\n{sql}\n\n"
             "ROWS (JSON):\n{rows}\n\n"
             "Answer:")
        ])
        return answer_prompt | self.llm_answer | self.parser

    # ---------- Stage A ----------
    def generate_sql_with_retries(
        self,
        *,
        session_id: int,
        question: str,
        schema_text: str,
        context_obj: Dict[str, Any],
        validate_fn,               # callable(sql: str) -> Tuple[bool, Optional[str], str]
        trace_fn                   # callable(stage, prompt, response, error)
    ) -> Tuple[str, Optional[str]]:
        """
        Returns (final_sql, err). Up to 4 attempts; feeds verifier & engine feedback back to the model.
        """
        context_json = json.dumps(context_obj or {}, ensure_ascii=False)
        validator_feedback = ""
        last_sql = ""
        last_err = None

        for attempt in range(1, 5):
            # A1) Generate SQL
            prompt_vars = {
                "schema": schema_text,
                "context": context_json,
                "question": question,
                "validator_feedback": validator_feedback
            }
            log.info("A%02d: asking LLM to generate SQL", attempt)
            try:
                raw = self.sql_chain.invoke(prompt_vars).strip()
                trace_fn("A-generate-sql", json.dumps(prompt_vars, ensure_ascii=False), raw, None)
                sql = extract_sql(raw)
                log.info("A%02d SQL draft:\n%s", attempt, sql or "<empty>")
            except Exception as e:
                tb = traceback.format_exc()
                trace_fn("A-generate-sql", json.dumps(prompt_vars, ensure_ascii=False), None, tb)
                last_err = f"LLM call failed: {e}"
                log.error("A%02d LLM error: %s", attempt, last_err)
                sql = ""

            if not sql or not sql.lower().startswith("select"):
                last_err = "No valid SELECT produced."
                validator_feedback = (
                    f"Validator feedback (attempt {attempt}): {last_err}\n"
                    "Please return exactly one valid SQLite SELECT."
                )
                last_sql = sql
                continue

            # A2) LLM self-verification
            log.info("A%02d: verifying SQL via LLM", attempt)
            try:
                vjson_raw = self.verify_chain.invoke({"schema": schema_text, "sql": sql}).strip()
                trace_fn("A-verify-sql", json.dumps({"schema": schema_text, "sql": sql}, ensure_ascii=False), vjson_raw, None)
                ver = json.loads(vjson_raw[vjson_raw.find("{"): vjson_raw.rfind("}")+1])
            except Exception as e:
                log.warning("A%02d: verifier parse failed (%s); continuing to engine validator", attempt, e)
                ver = {"valid": True, "reasons": [], "suggested_sql": ""}

            if not ver.get("valid", True):
                last_err = "LLM verifier: " + "; ".join(ver.get("reasons") or ["invalid"])
                suggested = (ver.get("suggested_sql") or "").strip()
                log.info("A%02d: verifier says invalid: %s", attempt, last_err)
                if suggested.lower().startswith("select"):
                    sql = suggested
                    log.info("A%02d: using verifier's suggested SQL", attempt)
                else:
                    validator_feedback = (
                        f"Verifier says invalid (attempt {attempt}): {last_err}\n"
                        "Please fix and return a single valid SQLite SELECT."
                    )
                    last_sql = sql
                    continue

            # A3) Engine validator (EXPLAIN + dry-run)
            ok, err_msg, fixed_sql = validate_fn(sql)
            log.info("A%02d: engine validator -> ok=%s", attempt, ok)
            if not ok:
                last_err = f"Validator error: {err_msg}"
                validator_feedback = f"Validator feedback (attempt {attempt}): {last_err}"
                last_sql = sql
                continue

            return fixed_sql or sql, None

        return last_sql, (last_err or "Failed to produce valid SQL")

    # ---------- Stage C ----------
    def generate_answer_from_rows(
        self,
        *,
        session_id: int,
        question: str,
        sql_used: str,
        rows: List[Dict[str, Any]],
        trace_fn
    ) -> str:
        prompt_vars = {
            "question": question,
            "sql": sql_used,
            "rows": json.dumps(rows, ensure_ascii=False)
        }
        try:
            out = self.answer_chain.invoke(prompt_vars).strip()
            trace_fn("C-generate-answer", json.dumps(prompt_vars, ensure_ascii=False), out, None)
            return out
        except Exception as e:
            tb = traceback.format_exc()
            trace_fn("C-generate-answer", json.dumps(prompt_vars, ensure_ascii=False), None, tb)
            return "I couldn’t find matching records." if not rows else json.dumps(rows[:5], ensure_ascii=False, indent=2)
