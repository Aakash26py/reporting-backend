import sqlite3, os, random
from faker import Faker
from datetime import datetime

DB_PATH = "chat_history.db"
fake = Faker("en_IN")

# ============================================================
#  CREATE SCHEMA
# ============================================================
def create_schema():
    """Create all tables, including schema_descriptions."""
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)  # reset for clean schema

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executescript("""
    CREATE TABLE sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT
    );

    CREATE TABLE messages (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        role TEXT,
        content TEXT,
        type TEXT,
        created_at TEXT
    );

    CREATE TABLE reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER,
        prompt TEXT,
        pdf_path TEXT,
        created_at TEXT
    );

    CREATE TABLE customers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT,
        city TEXT,
        age INTEGER,
        segment TEXT
    );

    CREATE TABLE loans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        customer_id INTEGER,
        branch TEXT,
        amount REAL,
        interest REAL,
        status TEXT,
        disbursed TEXT
    );

    CREATE TABLE IF NOT EXISTS schema_descriptions (
        table_name TEXT NOT NULL,
        column_name TEXT NOT NULL,
        description TEXT NOT NULL,
        PRIMARY KEY (table_name, column_name)
    );
    """)
    conn.commit()
    conn.close()
    print("✅ Schema created successfully (with schema_descriptions).")


# ============================================================
#  SEED DUMMY DATA
# ============================================================
def seed_data(customers=50, loans=150):
    """Populate fake data for demo."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    print("🌱 Seeding dummy banking data...")
    segments = ["Retail", "Corporate", "SME"]

    for _ in range(customers):
        cur.execute(
            "INSERT INTO customers (name, city, age, segment) VALUES (?, ?, ?, ?)",
            (fake.name(), fake.city(), random.randint(22, 65), random.choice(segments))
        )

    cur.execute("SELECT id FROM customers")
    cust_ids = [r[0] for r in cur.fetchall()]

    branches = ["Delhi", "Mumbai", "Chennai", "Bangalore", "Kolkata"]
    statuses = ["active", "closed", "overdue", "NPA"]

    for _ in range(loans):
        cur.execute(
            "INSERT INTO loans (customer_id, branch, amount, interest, status, disbursed) VALUES (?, ?, ?, ?, ?, ?)",
            (
                random.choice(cust_ids),
                random.choice(branches),
                round(random.uniform(50000, 5000000), 2),
                round(random.uniform(6.5, 12.5), 2),
                random.choice(statuses),
                fake.date_between(start_date='-2y', end_date='today').isoformat()
            )
        )

    conn.commit()
    conn.close()
    print("✅ Dummy data seeded successfully.")


# ============================================================
#  POPULATE SCHEMA DESCRIPTIONS
# ============================================================
def seed_schema_descriptions():
    """Add human-readable descriptions for all tables and columns."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    descriptions = [
        # --- Customers table ---
        ("customers", "id", "Unique ID for each customer"),
        ("customers", "name", "Full name of the customer"),
        ("customers", "city", "City where the customer resides"),
        ("customers", "age", "Age of the customer"),
        ("customers", "segment", "Customer segment type — Retail, Corporate, or SME"),

        # --- Loans table ---
        ("loans", "id", "Unique loan ID"),
        ("loans", "customer_id", "Foreign key linking to customers.id"),
        ("loans", "branch", "Branch name where the loan was issued"),
        ("loans", "amount", "Loan amount sanctioned in INR"),
        ("loans", "interest", "Interest rate applied to the loan"),
        ("loans", "status", "Loan repayment status: active, closed, overdue, or NPA"),
        ("loans", "disbursed", "Date the loan was disbursed"),

        # --- Reports table ---
        ("reports", "id", "Unique ID for each generated report"),
        ("reports", "session_id", "Associated chat session ID"),
        ("reports", "prompt", "User prompt that triggered this report"),
        ("reports", "pdf_path", "Path to the generated PDF report"),
        ("reports", "created_at", "Timestamp when report was generated"),

        # --- Sessions table ---
        ("sessions", "id", "Unique ID for each chat session"),
        ("sessions", "created_at", "Session creation timestamp"),

        # --- Messages table ---
        ("messages", "id", "Unique message ID"),
        ("messages", "session_id", "Chat session to which the message belongs"),
        ("messages", "role", "Sender role: user or assistant"),
        ("messages", "content", "Message content text"),
        ("messages", "type", "Optional content type, e.g., query or report"),
        ("messages", "created_at", "Timestamp of the message"),
    ]

    cur.executemany("""
        INSERT OR REPLACE INTO schema_descriptions (table_name, column_name, description)
        VALUES (?, ?, ?)
    """, descriptions)

    conn.commit()
    conn.close()
    print("📘 schema_descriptions table populated successfully.")


# ============================================================
#  MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    create_schema()
    seed_data()
    seed_schema_descriptions()
    print("✅ Database setup complete with descriptive schema.")
