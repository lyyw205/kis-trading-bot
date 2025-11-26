import sqlite3
import pandas as pd
import mplfinance as mpf

DB_PATH = "trading.db"

REGION = "KR"
SYMBOL = "005930"
INTERVAL = "5m"
LIMIT = 300

def load_ohlcv(region, symbol, interval, limit=300):
    conn = sqlite3.connect(DB_PATH)
    query = f"""
        SELECT dt, open, high, low, close, volume
        FROM ohlcv_data
        WHERE region = ?
          AND symbol = ?
          AND interval = ?
        ORDER BY dt DESC
        LIMIT {limit}
    """
    df = pd.read_sql_query(query, conn, params=[region, symbol, interval])
    conn.close()

    if df.empty:
        print("❌ 데이터가 없습니다.")
        return None

    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)
    df.set_index("dt", inplace=True)
    return df

if __name__ == "__main__":
    df = load_ohlcv(REGION, SYMBOL, INTERVAL, LIMIT)
    if df is None:
        exit()

    mpf.plot(
        df,
        type="candle",
        volume=True,
        title=f"{REGION} {SYMBOL} {INTERVAL}",
        style="yahoo",
    )
