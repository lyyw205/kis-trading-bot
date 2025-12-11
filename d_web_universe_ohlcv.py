# d_web_universe_ohlcv.py
import pandas as pd
from typing import Optional
from c_db_manager import BotDatabase

# [수정] SSL 설정이 포함된 연결 함수 사용
from d_web_data import get_connection

def get_universe_coverage() -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT
                region,
                symbol,
                interval,
                COUNT(*) AS candles,
                MIN(dt) AS first_dt,
                MAX(dt) AS last_dt
            FROM ohlcv_data
            GROUP BY region, symbol, interval
            ORDER BY region, symbol, interval
            """,
            conn,
        )
    except Exception as e:
        print(f"[Error] get_universe_coverage: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame(columns=["region", "symbol", "interval", "candles", "first_dt", "last_dt", "days_covered"])

    df["first_dt"] = pd.to_datetime(df["first_dt"], errors='coerce')
    df["last_dt"] = pd.to_datetime(df["last_dt"], errors='coerce')
    
    df = df.dropna(subset=["first_dt", "last_dt"])
    
    if not df.empty:
        df["days_covered"] = (df["last_dt"] - df["first_dt"]).dt.days + 1
    else:
        df["days_covered"] = 0

    return df

def get_last_universe_backfill_time(db: BotDatabase) -> str:
    try:
        value = db.get_setting("last_universe_ohlcv_backfill", "")
        return value
    except Exception:
        return ""

def get_recent_backfill_failures(limit: int = 20) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT
                region,
                symbol,
                excd,
                interval,
                error_type,
                error_message,
                created_at
            FROM universe_backfill_failures
            ORDER BY created_at DESC
            LIMIT %s
            """,
            conn,
            params=(limit,),
        )
    except Exception as e:
        print(f"[Error] get_recent_backfill_failures: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame(
            columns=[
                "region",
                "symbol",
                "excd",
                "interval",
                "error_type",
                "error_message",
                "created_at",
            ]
        )
    
    return df