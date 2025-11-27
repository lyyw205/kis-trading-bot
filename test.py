import sqlite3

DB_PATH = "trading.db"

conn = sqlite3.connect(DB_PATH)
cur = conn.cursor()

def add_column_if_not_exists(table, column, coltype):
    cur.execute(f"PRAGMA table_info({table})")
    cols = [row[1] for row in cur.fetchall()]
    if column not in cols:
        print(f"➕ {table}.{column} 추가")
        cur.execute(f"ALTER TABLE {table} ADD COLUMN {column} {coltype}")

add_column_if_not_exists("trades", "order_no", "TEXT")
add_column_if_not_exists("trades", "source", "TEXT")
add_column_if_not_exists("trades", "entry_comment", "TEXT")
add_column_if_not_exists("trades", "exit_comment", "TEXT")

conn.commit()
conn.close()
print("✅ 완료")