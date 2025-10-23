# reports_app.py (OpenAI version with HTML saving + SQL debug logging)
from flask import Blueprint, request, jsonify, send_file
import os, re, io, json, time, threading, sqlite3, logging
from datetime import datetime

# Matplotlib MUST be headless before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from xhtml2pdf import pisa
from dotenv import load_dotenv
from schema_context import get_schema_context  # ✅ Import schema context

# ---------------------- CONFIG ----------------------
report_bp = Blueprint("reports", __name__)
DB_PATH = "chat_history.db"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CHARTS_DIR = os.path.join(BASE_DIR, "report_charts")
HTML_DIR = os.path.join(BASE_DIR, "report_html")  # ✅ folder for raw HTML
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)

load_dotenv()
MODEL = os.getenv("MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

REPORT_STATUS = {}

# ---------------------- LOGGING ----------------------
log = logging.getLogger("reports")
if not log.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    log.addHandler(handler)
log.setLevel(logging.INFO)

# ---------------------- DB & UTILS ----------------------
def get_db():
    return sqlite3.connect(DB_PATH)

def slug(s: str) -> str:
    s = re.sub(r"[^\w\s-]", "", str(s)).strip().lower()
    return re.sub(r"[\s]+", "_", s)[:80] or "chart"

def inr_short(n):
    try:
        n = float(n or 0)
    except Exception:
        return str(n)
    absn = abs(n)
    if absn >= 1e7:
        return f"₹{n/1e7:.2f} Cr"
    if absn >= 1e5:
        return f"₹{n/1e5:.2f} L"
    if absn >= 1e3:
        return f"₹{n/1e3:.2f} K"
    return f"₹{n:,.2f}"

def inr_full(n):
    try:
        n = float(n or 0)
        return f"₹{n:,.2f}"
    except Exception:
        return str(n)

def link_callback(uri, rel):
    if uri.startswith("file://"):
        return uri.replace("file://", "")
    if os.path.isabs(uri):
        return uri
    for root in (CHARTS_DIR, REPORTS_DIR, HTML_DIR, BASE_DIR):
        cand = os.path.join(root, uri)
        if os.path.exists(cand):
            return cand
    return uri

def save_pdf(html: str, filename: str) -> str:
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "wb") as f:
        pisa.CreatePDF(src=html, dest=f, link_callback=link_callback)
    return path

def save_chart(df: pd.DataFrame, chart_type: str, x: str, y: str, title: str):
    plt.figure(figsize=(7, 3.8), dpi=160)
    plt.grid(axis="y", linestyle="--", alpha=0.25)
    try:
        if chart_type == "pie":
            plt.pie(df[y], labels=df[x], autopct="%1.1f%%", startangle=140)
            plt.title(title)
        else:
            if chart_type == "bar":
                plt.bar(df[x], df[y], color="#22c55e")
            elif chart_type == "area":
                plt.plot(df[x], df[y], linewidth=2, color="#22c55e")
                plt.fill_between(range(len(df[x])), df[y], color="#22c55e", alpha=0.25)
            else:
                plt.plot(df[x], df[y], marker="o", linewidth=2, color="#22c55e")

            plt.title(title)
            plt.xticks(rotation=35, ha="right", fontsize=8)
            vals = plt.gca().get_yticks()
            plt.gca().set_yticklabels([inr_short(v) for v in vals], fontsize=8)

        plt.tight_layout()
        name = f"{slug(title)}_{int(time.time())}.png"
        path = os.path.join(CHARTS_DIR, name)
        plt.savefig(path, bbox_inches="tight")
        return path
    finally:
        plt.close()

# ---------------------- LLM (OpenAI version) ----------------------
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = ChatOpenAI(model=MODEL, temperature=0.2, api_key=OPENAI_API_KEY)
parser = StrOutputParser()

def call_llm(prompt_text: str):
    """Send a plain text prompt and return response string."""
    prompt = ChatPromptTemplate.from_messages([("human", "{text}")])
    chain = prompt | llm | parser
    return chain.invoke({"text": prompt_text}).strip()

# ---------------------- CORE ----------------------
def generate_report_background(report_id: int, topic: str):
    try:
        REPORT_STATUS[report_id] = {"ready": False, "stage": "planning", "message": "Planning report…"}
        log.info(f"[{report_id}] Starting report for topic: {topic}")

        schema_ctx = get_schema_context()
        schema_text = schema_ctx["schema_text"]

        planner_prompt = f"""
You are a senior banking data analyst. Return STRICT JSON.

Database schema (with column meanings):
{schema_text}

User topic: "{topic}"

Goal:
- Identify up to 5 meaningful financial metrics and 5-6 analytical charts related to this topic.
- Each metric or chart must be based on valid SQL queries that use ONLY columns from the schema.
- Prefer joins between loans and customers when analyzing customers or branches.
- Always alias the horizontal axis as 'x' and the numeric/metric column as 'y'.
- Chart types allowed: bar, line, pie, area.

Return JSON format ONLY:
{{
  "metrics": [
    {{"title": "Total Loan Amount", "sql": "SELECT SUM(loans.amount) FROM loans"}}
  ],
  "charts": [
    {{
      "title": "Loan Distribution by Branch",
      "type": "bar",
      "sql": "SELECT loans.branch AS x, SUM(loans.amount) AS y FROM loans GROUP BY loans.branch",
      "x": "x",
      "y": "y",
      "insight": "Describes which branches dominate the loan portfolio."
    }}
  ]
}}
"""
        plan_raw = call_llm(planner_prompt)
        log.info(f"[{report_id}] Raw plan output:\n{plan_raw}")

        plan_raw = plan_raw.replace("```json", "").replace("```", "").strip()
        if not plan_raw.startswith("{"):
            plan_raw = plan_raw[plan_raw.find("{"):]
        if not plan_raw.endswith("}"):
            plan_raw = plan_raw[: plan_raw.rfind("}") + 1]

        try:
            plan = json.loads(plan_raw)
        except Exception:
            raise ValueError("❌ LLM failed to produce valid JSON plan")

        REPORT_STATUS[report_id] = {"ready": False, "stage": "query", "message": "Running queries…"}

        conn = get_db()
        metrics, charts = [], []

        # ---------- METRICS ----------
        for m in plan.get("metrics", []):
            title, sql = m.get("title"), m.get("sql")
            log.info(f"[{report_id}] Running metric SQL: {sql}")
            try:
                df = pd.read_sql_query(sql, conn)
                val = df.iloc[0, 0] if not df.empty else 0
                metrics.append({"title": title, "value_short": inr_short(val), "value_full": inr_full(val)})
            except Exception as e:
                log.error(f"[{report_id}] Metric SQL failed ({title}): {e}")
                metrics.append({"title": title, "value_short": "Error", "value_full": str(e)})

        # ---------- CHARTS ----------
        for c in plan.get("charts", []):
            title, sql = c.get("title"), c.get("sql")
            x, y, chart_type = c.get("x") or "x", c.get("y") or "y", c.get("type", "bar")
            insight = c.get("insight", "")
            log.info(f"[{report_id}] Chart '{title}' | SQL: {sql}")
            try:
                df = pd.read_sql_query(sql, conn)
                log.info(f"[{report_id}] Chart '{title}' returned {len(df)} rows, columns: {list(df.columns)}")

                # Try to auto-detect x/y if not found
                if x not in df.columns or y not in df.columns:
                    if len(df.columns) >= 2:
                        x, y = df.columns[0], df.columns[1]
                        log.warning(f"[{report_id}] Auto-detected x='{x}', y='{y}' for '{title}'")
                    else:
                        log.warning(f"[{report_id}] Skipped '{title}' (missing x/y columns)")
                        continue

                if df.empty:
                    log.warning(f"[{report_id}] Skipped '{title}' (empty dataframe)")
                    continue

                img_path = save_chart(df, chart_type, x, y, title)
                charts.append({"title": title, "img_path": img_path, "insight": insight})
            except Exception as e:
                log.error(f"[{report_id}] Chart '{title}' failed: {e}")
                charts.append({"title": f"{title} (failed)", "error": str(e)})
        conn.close()

        REPORT_STATUS[report_id] = {"ready": False, "stage": "writing", "message": "Writing PDF…"}

        # ---------- EXECUTIVE SUMMARY ----------
        metric_summary = "\n".join([f"- {m['title']}: {m['value_full']}" for m in metrics])
        summary_prompt = f"""
You are a senior banking executive.
Write 3 short paragraphs summarizing this report for "{topic}".
Use INR values and refer to branches or trends if relevant.
Metrics summary:
{metric_summary}
"""
        executive = call_llm(summary_prompt)
        log.info(f"[{report_id}] Executive summary generated.")

        # ---------- BUILD HTML ----------
        def metrics_html(items):
            return "".join([
                f"<div class='card'><div class='card-title'>{m['title']}</div><div class='card-value'>{m['value_short']}</div></div>"
                for m in items
            ])

        def charts_html(items):
            html = ""
            for c in items:
                if c.get("img_path"):
                    html += f"<div class='chart'><h3>{c['title']}</h3><img src='file://{c['img_path']}'/><p>{c.get('insight','')}</p></div>"
                else:
                    html += f"<div class='chart'><h3>{c['title']}</h3><div class='chart-failed'>Chart failed</div></div>"
            return html

        styles = """
        <style>
          @page { size: A4; margin: 28pt 28pt 36pt 28pt; }
          body { font-family: Helvetica, Arial, sans-serif; color:#111; }
          h1 { color:#0a7b55; border-bottom: 2px solid #0a7b55; padding-bottom:6pt; }
          .grid { display: flex; flex-wrap: wrap; gap: 8pt; }
          .card { flex: 1 1 45%; border: 1px solid #dfe4e8; border-radius:6pt; padding:10pt; }
          .card-title { font-size: 9pt; color:#6b7280; text-transform: uppercase; letter-spacing:.7pt; }
          .card-value { font-size: 16pt; font-weight: 700; color:#065f46; margin-top:4pt; }
          .chart img { width: 480pt; height:auto; }
          .chart p { font-size:10pt; color:#374151; margin-top:4pt; }
        </style>
        """

        html = f"""<html><head>{styles}</head><body>
        <h1>Banking Report</h1>
        <h2>Executive Summary</h2>
        <p>{executive}</p>
        <h2>Key Metrics</h2>
        <div class='grid'>{metrics_html(metrics)}</div>
        <h2>Visual Analysis</h2>
        {charts_html(charts)}
        <p style='font-size:9pt;color:#888;text-align:center;'>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </body></html>"""

        # ---------- SAVE HTML + PDF ----------
        timestamp = int(time.time())
        html_filename = f"report_{timestamp}.html"
        pdf_filename = f"report_{timestamp}.pdf"

        html_path = os.path.join(HTML_DIR, html_filename)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        pdf_path = save_pdf(html, pdf_filename)
        log.info(f"[{report_id}] Saved HTML: {html_path}")
        log.info(f"[{report_id}] Saved PDF: {pdf_path}")

        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute("ALTER TABLE reports ADD COLUMN html_path TEXT")
        except Exception:
            pass
        cur.execute(
            "INSERT INTO reports (prompt, pdf_path, html_path, created_at) VALUES (?,?,?,?)",
            (topic, os.path.abspath(pdf_path), os.path.abspath(html_path), datetime.now().isoformat())
        )
        conn.commit()
        conn.close()

        REPORT_STATUS[report_id] = {
            "ready": True,
            "url_pdf": f"http://localhost:8004/reports/{pdf_filename}",
            "url_html": f"http://localhost:8004/html/{html_filename}",
            "topic": topic
        }

        log.info(f"[{report_id}] Report generation complete ✅")

    except Exception as e:
        log.exception(f"[{report_id}] Report generation failed: {e}")
        REPORT_STATUS[report_id] = {"error": str(e)}

# ---------------------- ROUTES ----------------------
@report_bp.route("/request", methods=["POST"])
def request_report():
    data = request.get_json(force=True, silent=True) or {}
    topic = (data.get("prompt") or data.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "Missing 'prompt'"}), 400
    rid = int(time.time() * 1000)
    REPORT_STATUS[rid] = {"ready": False, "stage": "start", "message": "Queued…"}
    threading.Thread(target=generate_report_background, args=(rid, topic), daemon=True).start()
    return jsonify({"report_id": rid, "status": "started"})

@report_bp.route("/status/<int:rid>")
def report_status(rid):
    return jsonify(REPORT_STATUS.get(rid, {"error": "Report ID not found"}))

@report_bp.route("/<path:filename>")
def serve_report(filename):
    return send_file(os.path.join(REPORTS_DIR, filename), as_attachment=True)

@report_bp.route("/html/<path:filename>")
def serve_html(filename):
    return send_file(os.path.join(HTML_DIR, filename))

@report_bp.route("/reports_list")
def reports_list():
    conn = get_db()
    cur = conn.cursor()
    try:
        cur.execute("SELECT prompt,pdf_path,html_path,created_at FROM reports ORDER BY id DESC")
        rows = [
            {
                "prompt": r[0],
                "pdf_url": f"http://localhost:8004/reports/{os.path.basename(r[1])}",
                "html_url": f"http://localhost:8004/html/{os.path.basename(r[2])}" if r[2] else None,
                "created_at": r[3]
            }
            for r in cur.fetchall()
        ]
    except sqlite3.OperationalError:
        cur.execute("SELECT prompt,pdf_path,created_at FROM reports ORDER BY id DESC")
        rows = [
            {
                "prompt": r[0],
                "pdf_url": f"http://localhost:8004/reports/{os.path.basename(r[1])}",
                "created_at": r[2]
            }
            for r in cur.fetchall()
        ]
    finally:
        conn.close()
    return jsonify(rows)
