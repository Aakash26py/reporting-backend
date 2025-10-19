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

# ---------------------- CONFIG ----------------------
report_bp = Blueprint("reports", __name__)
DB_PATH = "chat_history.db"
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
CHARTS_DIR  = os.path.join(BASE_DIR, "report_charts")
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)

# Track background jobs
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
    if absn >= 1e7:   # Crore
        return f"₹{n/1e7:.2f} Cr"
    if absn >= 1e5:   # Lakh
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
    """
    Allows xhtml2pdf to resolve local image paths.
    Accepts:
      - absolute paths
      - file:///... URIs
      - relative to current file (report_app.py) or /report_charts
    """
    if uri.startswith("file://"):
        path = uri.replace("file://", "")
        return path
    # If already absolute
    if os.path.isabs(uri):
        return uri
    # Try relative to charts and reports, then to this module dir
    for root in (CHARTS_DIR, REPORTS_DIR, BASE_DIR):
        cand = os.path.join(root, uri)
        if os.path.exists(cand):
            return cand
    return uri  # pisa will try, but likely fail

def save_pdf(html: str, filename: str) -> str:
    path = os.path.join(REPORTS_DIR, filename)
    with open(path, "wb") as f:
        pisa.CreatePDF(src=html, dest=f, link_callback=link_callback)
    return path

def save_chart(df: pd.DataFrame, chart_type: str, x: str, y: str, title: str):
    """
    Creates a chart image and returns an absolute path.
    chart_type: bar | line | area | pie
    """
    plt.figure(figsize=(7, 3.8), dpi=160)
    plt.grid(axis="y", linestyle="--", alpha=0.25)

    try:
        if chart_type == "pie":
            # Expect single series: labels in x, values in y
            plt.pie(df[y], labels=df[x], autopct="%1.1f%%", startangle=140)
            plt.title(title)
        else:
            if chart_type == "bar":
                plt.bar(df[x], df[y], color="#22c55e")
            elif chart_type == "area":
                plt.plot(df[x], df[y], linewidth=2, color="#22c55e")
                plt.fill_between(range(len(df[x])), df[y], color="#22c55e", alpha=0.25)
            else:  # line (default)
                plt.plot(df[x], df[y], marker="o", linewidth=2, color="#22c55e")

            plt.title(title)
            plt.xticks(rotation=35, ha="right", fontsize=8)
            # Indian currency on Y tick labels
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
llm = OllamaLLM(model="llama3", temperature=0.2)

def schema_text():
    conn = get_db()
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cur.fetchall()]
    parts = []
    for t in tables:
        try:
            cur.execute(f"PRAGMA table_info({t})")
            cols = [c[1] for c in cur.fetchall()]
            parts.append(f"{t}: {', '.join(cols)}")
        except Exception:
            pass
    conn.close()
    return "\n".join(parts)

def distinct_branches():
    conn = get_db()
    try:
        df = pd.read_sql_query("SELECT DISTINCT branch FROM loans", conn)
        return [b for b in df["branch"].dropna().astype(str).tolist()]
    except Exception:
        return []
    finally:
        conn.close()

def guess_focus_branch(topic: str) -> str | None:
    topic_low = topic.lower()
    for b in distinct_branches():
        if b and b.lower() in topic_low:
            return b
    return None

# ---------------------- CORE: report generation ----------------------
def generate_report_background(report_id: int, topic: str):
    try:
        REPORT_STATUS[report_id] = {"ready": False, "stage": "planning", "message": "Planning report…"}

        focus_branch = guess_focus_branch(topic)  # None if not mentioned
        schema = schema_text()

        planner = f"""
You are a financial data analyst. Output ONLY valid JSON.

SQLite schema:
{schema}

User topic: "{topic}"
Focus branch (if provided): {focus_branch or "None"}

Goal:
- Produce up to 5 executive metrics and up to 4 charts RELEVANT to the topic.
- Use columns that exist.
- If a focus branch is given, ensure your SQL filters include:
  WHERE branch = :focus_branch
- For total amounts, aggregate loans.amount.
- For interest insights, aggregate AVG(loans.interest).
- For status, aggregate COUNT(*) GROUP BY status.
- Each chart must include a "type": "bar" | "line" | "area" | "pie".
- Use keys: title, sql, x, y, type (for charts) and title, sql (for metrics).

Return JSON strictly like:

{{
  "metrics": [
    {{"title": "Total Loan Amount", "sql": "SELECT SUM(amount) FROM loans WHERE branch = :focus_branch"}}
  ],
  "charts": [
    {{"title": "Loan Amount by Status", "type": "bar", "sql": "SELECT status AS label, SUM(amount) AS value FROM loans WHERE branch = :focus_branch GROUP BY status", "x": "label", "y": "value"}}
  ]
}}
"""
        plan_raw = llm.invoke(planner).strip()
        plan_raw = plan_raw.replace("```json", "").replace("```", "").strip()
        if not plan_raw.startswith("{"):
            plan_raw = plan_raw[plan_raw.find("{"):]
        if not plan_raw.endswith("}"):
            plan_raw = plan_raw[: plan_raw.rfind("}") + 1]

        try:
            plan = json.loads(plan_raw)
        except Exception:
            # Fallback sensible plan
            plan = {
                "metrics": [
                    {"title": "Total Loan Amount", "sql": "SELECT SUM(amount) FROM loans {WHERE_CLAUSE}"},
                    {"title": "Average Interest (%)", "sql": "SELECT ROUND(AVG(interest),2) FROM loans {WHERE_CLAUSE}"},
                    {"title": "Active Loans", "sql": "SELECT COUNT(*) FROM loans {WHERE_CLAUSE_AND} status='active'"},
                ],
                "charts": [
                    {"title": "Loans per Branch", "type": "bar",
                     "sql": "SELECT branch AS label, COUNT(*) AS value FROM loans GROUP BY branch",
                     "x": "label", "y": "value"},
                    {"title": "Loan Amount by Status", "type": "pie",
                     "sql": "SELECT status AS label, SUM(amount) AS value FROM loans {WHERE_CLAUSE} GROUP BY status",
                     "x": "label", "y": "value"},
                    {"title": "Average Interest by Branch", "type": "line",
                     "sql": "SELECT branch AS label, ROUND(AVG(interest),2) AS value FROM loans GROUP BY branch",
                     "x": "label", "y": "value"},
                ],
            }

        # normalize placeholders
        where_clause = ""
        where_clause_and = ""
        params = {}
        if focus_branch:
            where_clause = "WHERE branch = :focus_branch"
            where_clause_and = "WHERE branch = :focus_branch AND"
            params["focus_branch"] = focus_branch

        def inject_where(sql: str) -> str:
            return (
                sql.replace("{WHERE_CLAUSE_AND}", where_clause_and)
                   .replace("{WHERE_CLAUSE}", where_clause)
            )

        REPORT_STATUS[report_id] = {"ready": False, "stage": "query", "message": "Running queries…"}

        # Run metrics
        metrics = []
        conn = get_db()
        for m in plan.get("metrics", [])[:5]:
            title = m.get("title", "Metric")
            sql = inject_where(m.get("sql", "SELECT 0"))
            try:
                df = pd.read_sql_query(sql, conn, params=params)
                val = df.iloc[0, 0] if not df.empty else 0
                # format INR if numeric
                if isinstance(val, (int, float)):
                    metrics.append({"title": title, "value_short": inr_short(val), "value_full": inr_full(val)})
                else:
                    metrics.append({"title": title, "value_short": str(val), "value_full": str(val)})
            except Exception as e:
                metrics.append({"title": title, "value_short": "Error", "value_full": str(e)})

        # Run charts
        charts = []
        for c in plan.get("charts", [])[:4]:
            title = c.get("title", "Chart")
            chart_type = c.get("type", "bar").lower()
            sql = inject_where(c.get("sql", "SELECT 0 AS x, 0 AS y"))
            x_key = c.get("x", "x")
            y_key = c.get("y", "y")
            try:
                df = pd.read_sql_query(sql, conn, params=params)
                # If LLM used generic column names, normalize to x/y
                if x_key not in df.columns or y_key not in df.columns:
                    # try common fallbacks
                    colmap = { "label": "x", "value": "y" }
                    for k in list(df.columns):
                        if k.lower() == "label": df.rename(columns={k: "x"}, inplace=True)
                        if k.lower() == "value": df.rename(columns={k: "y"}, inplace=True)
                    x_key = "x" if "x" in df.columns else (df.columns[0] if len(df.columns) else "x")
                    y_key = "y" if "y" in df.columns else (df.columns[1] if len(df.columns) > 1 else "y")

                img_path = save_chart(df, chart_type, x_key, y_key, title)
                charts.append({"title": title, "img_path": img_path})
            except Exception as e:
                charts.append({"title": f"{title} (failed)", "error": str(e)})
        conn.close()

        REPORT_STATUS[report_id] = {"ready": False, "stage": "writing", "message": "Writing PDF…"}

        # Summary with real numbers
        mtext = "\n".join([f"- {m['title']}: {m['value_full']}" for m in metrics])
        focus_line = f"Focus branch: {focus_branch}." if focus_branch else "Portfolio-wide view."
        summ_prompt = f"""
You are a senior banking analyst.
Write 2 crisp executive paragraphs (no bullets) using INR, then a short guidance paragraph.
Context:
{focus_line}
Metrics:
{mtext}
Keep it businesslike and specific. Avoid repeating the metric labels verbatim.
"""
        executive = llm.invoke(summ_prompt).strip()

        # Build HTML
        def metrics_grid_html(items):
            cards = []
            for m in items:
                cards.append(f"""
                  <div class="card">
                    <div class="card-title">{m['title']}</div>
                    <div class="card-value">{m['value_short']}</div>
                  </div>""")
            return "\n".join(cards)

        def charts_html(items):
            blocks = []
            for c in items:
                if c.get("img_path"):
                    blocks.append(f"""
                      <div class="chart">
                        <h3>{c['title']}</h3>
                        <img src="file://{c['img_path']}" />
                      </div>""")
                else:
                    blocks.append(f"""
                      <div class="chart">
                        <h3>{c['title']}</h3>
                        <div class="chart-failed">Chart failed</div>
                      </div>""")
            return "\n".join(blocks)

        styles = """
        <style>
          @page { size: A4; margin: 28pt 28pt 36pt 28pt; }
          body { font-family: Helvetica, Arial, sans-serif; color:#111; }
          h1 { color:#0a7b55; border-bottom: 2px solid #0a7b55; padding-bottom:6pt; }
          h2 { margin: 12pt 0 6pt; color:#1a1a1a; }
          p  { font-size: 11pt; line-height: 1.45; text-align: justify; }
          .grid { display: flex; flex-wrap: wrap; gap: 8pt; }
          .card { flex: 1 1 45%; border: 1px solid #dfe4e8; border-radius:6pt; padding:10pt; }
          .card-title { font-size: 9pt; color:#6b7280; text-transform: uppercase; letter-spacing:.7pt; }
          .card-value { font-size: 16pt; font-weight: 700; color:#065f46; margin-top:4pt; }
          .section { margin-top: 12pt; }
          .chart { margin: 10pt 0; text-align:center; }
          .chart img { width: 490pt; height: auto; }
          .chart-failed { padding: 18pt; border:1px dashed #c2410c; color:#9a3412; font-size:10pt; }
          table { width:100%; border-collapse: collapse; margin-top: 6pt; }
          th, td { border:1px solid #e5e7eb; padding:6pt; font-size: 9.5pt; }
          th { background:#0a7b55; color: #fff; }
          .footer { text-align:center; color:#6b7280; font-size:9pt; margin-top:10pt; }
        </style>
        """

        html = f"""<html><head>{styles}</head><body>
          <h1>Banking Report</h1>

          <div class="section">
            <h2>Executive Snapshot</h2>
            <div class="grid">
              {metrics_grid_html(metrics)}
            </div>
          </div>

          <div class="section">
            <h2>Executive Summary</h2>
            <p>{executive}</p>
          </div>

          <div class="section">
            <h2>Visual Analysis</h2>
            {charts_html(charts)}
          </div>

          <div class="footer">
            Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} • Topic: {topic}
          </div>
        </body></html>"""

        filename = f"report_{int(time.time())}.pdf"
        pdf_path = save_pdf(html, filename)

        # persist to DB
        conn = get_db()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO reports (prompt, pdf_path, created_at) VALUES (?,?,?)",
            (topic, os.path.abspath(pdf_path), datetime.now().isoformat()),
        )
        conn.commit(); conn.close()

        REPORT_STATUS[report_id] = {
            "ready": True,
            "url": f"http://localhost:8004/reports/{filename}",
            "topic": topic,
            "focus_branch": focus_branch,
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
    t = threading.Thread(target=generate_report_background, args=(rid, topic), daemon=True)
    t.start()
    return jsonify({"report_id": rid, "status": "started"})

@report_bp.route("/status/<int:rid>")
def report_status(rid):
    return jsonify(REPORT_STATUS.get(rid, {"error": "Report ID not found"}))

@report_bp.route("/<path:filename>")
def serve_report(filename):
    return send_file(os.path.join(REPORTS_DIR, filename), as_attachment=True)

@report_bp.route("/reports_list")
def reports_list():
    try:
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT prompt,pdf_path,created_at FROM reports ORDER BY id DESC")
        rows = [
            {
                "prompt": r[0],
                "url": f"http://localhost:8004/reports/{os.path.basename(r[1])}",
                "created_at": r[2],
            }
            for r in cur.fetchall()
        ]
        conn.close()
        return jsonify(rows)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
