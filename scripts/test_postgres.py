from sqlalchemy import create_engine

db_url = "postgresql://postgres:Admin@localhost:5432/jio_advision"
engine=create_engine(db_url)

try:
    with engine.connect() as conn:
        print("✅ Connected successfully to PostgreSQL!")
except Exception as e:
    print("❌ Connection failed:", e)
