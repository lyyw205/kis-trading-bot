# import_app_fills.py
from kis_api import KisDataFetcher
from config import APP_KEY, APP_SECRET, ACCOUNT_NO
from db import BotDatabase
from datetime import datetime, timedelta
import sqlite3

fetcher = KisDataFetcher(APP_KEY, APP_SECRET, ACCOUNT_NO, mode="real")
db = BotDatabase("trading.db")

# 최근 7일 가져오기
today = datetime.now()
start_date = (today - timedelta(days=7)).strftime("%Y%m%d")
end_date   = today.strftime("%Y%m%d")

fills = fetcher.get_us_fills_normalized(start_date=start_date, end_date=end_date)

# 이미 저장된 order_no 목록 불러오기 (중복 방지)
conn = sqlite3.connect("trading.db")
cur = conn.cursor()
cur.execute("SELECT DISTINCT order_no FROM trades WHERE order_no IS NOT NULL;")
existing_orders = {row[0] for row in cur.fetchall()}
conn.close()

inserted = 0

for f in fills:
    if f["order_no"] in existing_orders:
        continue  # 중복 스킵

    trade_time = f["time"]  # ✅ KIS 체결 시각 그대로 사용 (datetime 이든 문자열이든)

    db.save_trade(
        symbol=f["symbol"],
        trade_type=f["type"],     # BUY or SELL
        price=f["price"],
        qty=f["qty"],
        profit=0,
        ml_proba=None,
        extra={
            "source": f["source"],   # 모바일 or OpenAPI
            "order_no": f["order_no"]
        },
        trade_time=trade_time,       # ✅ 실제 체결 시간 전달
    )
    inserted += 1

print(f"동기화 완료: {inserted}개 체결 저장됨.")
