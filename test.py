# rebuild_trades_from_logs_simple.py

import sqlite3
import pandas as pd
import re

DB_PATH = "trading.db"

conn = sqlite3.connect(DB_PATH)
logs = pd.read_sql_query("SELECT time, message FROM logs ORDER BY time", conn)

buy_pattern = re.compile(
    r"\[(KR|US)매수체결\].*?([A-Z0-9]+)\s+(\d+)주\s*\| 체결가:(\d+(\.\d+)?)"
)

rows = []

for _, row in logs.iterrows():
    t = row["time"]
    msg = row["message"]

    m = buy_pattern.search(msg)
    if m:
        region = m.group(1)
        symbol = m.group(2)
        qty = int(m.group(3))
        price = float(m.group(4))

        rows.append(
            (
                t,          # time
                symbol,     # symbol
                "BUY",      # type
                price,      # price
                qty,        # qty
                0.0,        # profit → 매도 전이므로 0
                None,       # signal_id → 복구 데이터는 없음
                None,       # ml_proba
                None        # entry_allowed
            )
        )

cur = conn.cursor()
cur.executemany(
    """
    INSERT INTO trades
    (time, symbol, type, price, qty, profit, signal_id, ml_proba, entry_allowed)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """,
    rows,
)
conn.commit()
conn.close()

print("복구 완료:", len(rows), "건")
