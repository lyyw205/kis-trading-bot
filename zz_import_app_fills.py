# import_app_fills.py
from brk_kis_client import KisDataFetcher
from config import APP_KEY, APP_SECRET, ACCOUNT_NO
from db_manager import BotDatabase
from datetime import datetime, timedelta
import sqlite3
import pytz

KST = pytz.timezone("Asia/Seoul")

def build_trade_time_from_fill(f):
    """
    항상 'YYYY-MM-DD HH:MM:SS' 문자열을 반환하도록 정리
    """
    date_str = f.get("date") or f.get("trd_date") or f.get("ord_dt")
    time_str = f.get("time_raw") or f.get("trd_time") or f.get("ord_tmd")

    # 1) date(YYYYMMDD) + time(HHMMSS) 조합 가능한 경우
    if date_str and time_str and len(date_str) == 8 and len(time_str) >= 6:
        dt = datetime.strptime(date_str + time_str[:6], "%Y%m%d%H%M%S")
        dt_kst = KST.localize(dt)
        # KST 기준 naive 문자열로
        return dt_kst.strftime("%Y-%m-%d %H:%M:%S")

    # 2) 이미 "YYYY-MM-DD HH:MM:SS" 형태 문자열이 f["time"]에 있는 경우
    raw = f.get("time")
    if isinstance(raw, str) and len(raw) >= 16:
        try:
            dt = datetime.strptime(raw[:19], "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%Y-%m-%d %H:%M:%S")
        except ValueError:
            pass

    # 3) 마지막 fallback: 그냥 문자열로 강제 변환
    if raw is not None:
        return str(raw)

    # 그래도 없으면 지금 시각이라도 넣기
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def sync_app_fills_main():
    fetcher = KisDataFetcher(APP_KEY, APP_SECRET, ACCOUNT_NO, mode="real")
    db = BotDatabase("trading.db")

    today = datetime.now()
    start_date = (today - timedelta(days=7)).strftime("%Y%m%d")
    end_date   = today.strftime("%Y%m%d")

    fills = fetcher.get_us_fills_normalized(start_date=start_date, end_date=end_date)

    conn = sqlite3.connect("trading.db")
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT order_no FROM trades WHERE order_no IS NOT NULL;")
    existing_orders = {row[0] for row in cur.fetchall()}
    conn.close()

    inserted = 0
    for f in fills:
        if f["order_no"] in existing_orders:
            continue

        trade_time = build_trade_time_from_fill(f)

        db.save_trade(
            symbol=f["symbol"],
            trade_type=f["type"],
            price=f["price"],
            qty=f["qty"],
            profit=0,
            ml_proba=None,
            extra={
                "source": f["source"],
                "order_no": f["order_no"],
            },
            trade_time=trade_time,
        )
        inserted += 1

    return inserted

if __name__ == "__main__":
    inserted = sync_app_fills_main()
    print(f"동기화 완료: {inserted}개 체결 저장됨.")
