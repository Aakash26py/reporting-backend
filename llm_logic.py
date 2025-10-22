# llm_logic.py
import json, re, traceback, logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
MODEL = os.getenv("MODEL")

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
    Handles:
      - Stage A: LLM → SQL (with verification and retry)
      - Stage C: LLM → human answer from rows
    """

    def __init__(
        self,
        sql_model: str = MODEL,
        answer_model: str = MODEL,
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
             "TASK: Return a SINGLE valid SQL SELECT statement for SQLite that answers the request.\n"
             "Use only the columns/tables that exist and ensure syntax correctness.\n\n"
             "HARD RULES:\n"
             "• Never invent columns or tables.\n"
             "• Use real schema names exactly as shown.\n"
             "• Include stable identifiers (e.g., customers.id AS customer_id) when selecting customer info.\n"
             "• For 'highest/top', ORDER BY the numeric metric DESC and LIMIT appropriately.\n"
             "• If the user references a branch/city, use whichever column exists (e.g., loans.branch).\n"
             "• If context provides categorical values (like context.loan_statuses or context.loan_branches), "
             "only use those and never invent others.\n"
             "• Output SQL only — no commentary or markdown fences."
            ),
            ("human",
             "SCHEMA (tables → columns and descriptions):\n{schema}\n\n"
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
             '  \"valid\": true|false,\n'
             '  \"reasons\": [\"short reason 1\", \"short reason 2\"],\n'
             '  \"suggested_sql\": \"ONLY if invalid: corrected single SELECT using ONLY schema columns\",\n'
             '  \"used\": {\"tables\":[], \"columns\":[]}\n'
             "}\n"
             "If a column or table does not exist in the schema, mark invalid.\n"
             "Never add commentary outside JSON."
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
             "• NEVER invent facts — if rows are empty, say you couldn’t find matching records.\n"
             "• Use INR symbol (₹) for amounts.\n"
             "• If customer info exists, format like: 'Name (Branch): Amount ₹X; Status: active'.\n"
             "• Keep it short — no markdown, no tables."
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
        validate_fn,
        trace_fn
    ) -> Tuple[str, Optional[str]]:
        """
        Returns (final_sql, err). Retries up to 4 times with feedback.
        """
        context_json = json.dumps(context_obj or {}, ensure_ascii=False)
        validator_feedback = ""
        last_sql = ""
        last_err = None

        for attempt in range(1, 5):
            prompt_vars = {
                "schema": schema_text,
                "context": context_json,
                "question": question,
                "validator_feedback": validator_feedback
            }

            try:
                raw = self.sql_chain.invoke(prompt_vars).strip()
                trace_fn("A-generate-sql", json.dumps(prompt_vars, ensure_ascii=False), raw, None)
                sql = extract_sql(raw)
            except Exception as e:
                tb = traceback.format_exc()
                trace_fn("A-generate-sql", json.dumps(prompt_vars, ensure_ascii=False), None, tb)
                last_err = f"LLM error: {e}"
                sql = ""

            if not sql.lower().startswith("select"):
                validator_feedback = f"Attempt {attempt}: No valid SELECT returned."
                last_sql = sql
                continue

            # Verify with LLM
            try:
                vjson_raw = self.verify_chain.invoke({"schema": schema_text, "sql": sql}).strip()
                trace_fn("A-verify-sql", json.dumps({"schema": schema_text, "sql": sql}, ensure_ascii=False), vjson_raw, None)
                ver = json.loads(vjson_raw[vjson_raw.find("{"): vjson_raw.rfind("}")+1])
            except Exception as e:
                ver = {"valid": True}

            if not ver.get("valid", True):
                reasons = "; ".join(ver.get("reasons", []))
                last_err = f"Verifier rejected SQL: {reasons}"
                suggested = (ver.get("suggested_sql") or "").strip()
                if suggested.lower().startswith("select"):
                    sql = suggested
                else:
                    validator_feedback = f"Attempt {attempt}: {reasons}"
                    last_sql = sql
                    continue

            # Validate by running explain plan
            ok, err_msg, fixed_sql = validate_fn(sql)
            if not ok:
                last_err = err_msg
                validator_feedback = f"Attempt {attempt}: Engine validation failed ({err_msg})"
                last_sql = sql
                continue

            return fixed_sql or sql, None

        return last_sql, last_err or "Failed to generate valid SQL"

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
