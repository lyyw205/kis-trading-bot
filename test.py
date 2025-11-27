import sqlite3

conn = sqlite3.connect("trading.db")
cur = conn.cursor()

def add(col, type_):
    try:
        cur.execute(f"ALTER TABLE trades ADD COLUMN {col} {type_};")
        print(f"[OK] Added column {col}")
    except Exception as e:
        print(f"[SKIP] {col}:", e)

add("order_no", "TEXT")
add("source", "TEXT")

conn.commit()
conn.close()
