import sqlite3, os, random
from faker import Faker
from datetime import datetime

DB_PATH = "chat_history.db"
fake = Faker("en_IN")

def create_schema():
    """Create all tables."""
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
    """)
    conn.commit()
    conn.close()
    print("✅ Schema created successfully.")


def seed_data(customers=50, loans=150):
    """Populate fake data."""
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


if __name__ == "__main__":
    create_schema()
    seed_data()
