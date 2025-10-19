from flask import Blueprint, jsonify
import sqlite3

dashboard_bp = Blueprint("dashboard", __name__)
DB_PATH = "chat_history.db"

def get_db_connection():
    return sqlite3.connect(DB_PATH)


@dashboard_bp.route("/summary")
def dashboard_summary():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM customers")
    total_customers = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM loans")
    total_loans = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM loans WHERE status='active'")
    active_loans = cur.fetchone()[0]
    cur.execute("SELECT ROUND(AVG(interest),2) FROM loans")
    avg_interest = cur.fetchone()[0] or 0
    conn.close()
    return jsonify({
        "total_customers": total_customers,
        "total_loans": total_loans,
        "active_loans": active_loans,
        "avg_interest": avg_interest
    })


@dashboard_bp.route("/loan_status")
def dashboard_loan_status():
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("SELECT status, COUNT(*) FROM loans GROUP BY status")
    data = [{"status": s, "count": c} for s, c in cur.fetchall()]
    conn.close(); return jsonify(data)


@dashboard_bp.route("/loans_per_branch")
def dashboard_loans_per_branch():
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("SELECT branch, COUNT(*) FROM loans GROUP BY branch")
    data = [{"branch": b, "count": c} for b, c in cur.fetchall()]
    conn.close(); return jsonify(data)


@dashboard_bp.route("/loan_amount_branch")
def dashboard_loan_amount_branch():
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("SELECT branch, SUM(amount) FROM loans GROUP BY branch")
    data = [{"branch": b, "amount": a or 0} for b, a in cur.fetchall()]
    conn.close(); return jsonify(data)


@dashboard_bp.route("/avg_interest_branch")
def dashboard_avg_interest_branch():
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("SELECT branch, ROUND(AVG(interest),2) FROM loans GROUP BY branch")
    data = [{"branch": b, "interest": i or 0} for b, i in cur.fetchall()]
    conn.close(); return jsonify(data)


@dashboard_bp.route("/age_distribution")
def dashboard_age_distribution():
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        SELECT 
            CASE 
                WHEN age < 25 THEN '18–24'
                WHEN age BETWEEN 25 AND 35 THEN '25–35'
                WHEN age BETWEEN 36 AND 45 THEN '36–45'
                WHEN age BETWEEN 46 AND 55 THEN '46–55'
                WHEN age BETWEEN 56 AND 65 THEN '56–65'
                ELSE '65+' 
            END as age_group, COUNT(*) 
        FROM customers GROUP BY age_group
    """)
    data = [{"age_group": g, "count": c} for g, c in cur.fetchall()]
    conn.close(); return jsonify(data)


@dashboard_bp.route("/segment_avg_loans")
def dashboard_segment_avg_loans():
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        SELECT c.segment, ROUND(AVG(l.amount),0) 
        FROM loans l 
        JOIN customers c ON l.customer_id = c.id 
        GROUP BY c.segment
    """)
    data = [{"segment": s, "avg_amount": a or 0} for s, a in cur.fetchall()]
    conn.close(); return jsonify(data)


@dashboard_bp.route("/monthly_trend")
def dashboard_monthly_trend():
    conn = get_db_connection(); cur = conn.cursor()
    cur.execute("""
        SELECT strftime('%Y-%m', disbursed) as month, SUM(amount)
        FROM loans GROUP BY month ORDER BY month
    """)
    data = [{"month": m, "amount": a or 0} for m, a in cur.fetchall()]
    conn.close(); return jsonify(data)
