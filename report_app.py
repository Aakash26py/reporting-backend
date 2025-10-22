from flask import Blueprint, request, jsonify, send_file
import os, re, io, json, time, threading, sqlite3
from datetime import datetime

# Matplotlib MUST be headless before importing pyplot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from xhtml2pdf import pisa
from langchain_ollama import OllamaLLM

from schema_context import get_schema_context  # ✅ New import

# ---------------------- CONFIG ----------------------
report_bp = Blueprint("reports", __name__)
DB_PATH = "chat_history.db"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CHARTS_DIR  = os.path.join(BASE_DIR, "report_charts")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

from dotenv import load_dotenv
load_dotenv()
MODEL = os.getenv("MODEL")

REPORT_STATUS = {}

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
    for root in (CHARTS_DIR, REPORTS_DIR, BASE_DIR):
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

# ---------------------- LLM ----------------------
llm = OllamaLLM(model=MODEL, temperature=0.2)

# ---------------------- CORE ----------------------
def generate_report_background(report_id: int, topic: str):
    try:
        REPORT_STATUS[report_id] = {"ready": False, "stage": "planning", "message": "Planning report…"}

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
- Use realistic column names (e.g., loans.amount, customers.name).
- Each chart must include: title, sql, x, y, type, and a descriptive paragraph ("insight").
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

        plan_raw = llm.invoke(planner_prompt).strip()
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
        for m in plan.get("metrics", []):
            title, sql = m.get("title"), m.get("sql")
            try:
                df = pd.read_sql_query(sql, conn)
                val = df.iloc[0, 0] if not df.empty else 0
                metrics.append({"title": title, "value_short": inr_short(val), "value_full": inr_full(val)})
            except Exception as e:
                metrics.append({"title": title, "value_short": "Error", "value_full": str(e)})

        for c in plan.get("charts", []):
            title, sql = c.get("title"), c.get("sql")
            x, y, chart_type = c.get("x"), c.get("y"), c.get("type", "bar")
            insight = c.get("insight", "")
            try:
                df = pd.read_sql_query(sql, conn)
                if x not in df.columns or y not in df.columns:
                    continue
                img_path = save_chart(df, chart_type, x, y, title)
                charts.append({"title": title, "img_path": img_path, "insight": insight})
            except Exception as e:
                charts.append({"title": f"{title} (failed)", "error": str(e)})
        conn.close()

        REPORT_STATUS[report_id] = {"ready": False, "stage": "writing", "message": "Writing PDF…"}

        # Executive summary
        metric_summary = "\n".join([f"- {m['title']}: {m['value_full']}" for m in metrics])
        summary_prompt = f"""
You are a senior banking executive.
Write 3 short paragraphs summarizing this report for "{topic}".
Use INR values and refer to branches or trends if relevant.
Metrics summary:
{metric_summary}
"""
        executive = llm.invoke(summary_prompt).strip()

        # Build PDF
        def metrics_html(items):
            return "".join([f"<div class='card'><div class='card-title'>{m['title']}</div><div class='card-value'>{m['value_short']}</div></div>" for m in items])

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

        filename = f"report_{int(time.time())}.pdf"
        pdf_path = save_pdf(html, filename)

        conn = get_db()
        cur = conn.cursor()
        cur.execute("INSERT INTO reports (prompt, pdf_path, created_at) VALUES (?,?,?)",
                    (topic, os.path.abspath(pdf_path), datetime.now().isoformat()))
        conn.commit(); conn.close()

        REPORT_STATUS[report_id] = {
            "ready": True,
            "url": f"http://localhost:8004/reports/{filename}",
            "topic": topic
        }

    except Exception as e:
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

@report_bp.route("/reports_list")
def reports_list():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT prompt,pdf_path,created_at FROM reports ORDER BY id DESC")
    rows = [{"prompt": r[0], "url": f"http://localhost:8004/reports/{os.path.basename(r[1])}", "created_at": r[2]} for r in cur.fetchall()]
    conn.close()
    return jsonify(rows)
