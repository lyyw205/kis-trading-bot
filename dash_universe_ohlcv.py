# dash_universe_ohlcv.py
import sqlite3
from typing import Optional

import pandas as pd

from db_manager import BotDatabase

DB_PATH = "trading.db"


def get_universe_coverage() -> pd.DataFrame:
    """
    UNIVERSE 종목별 OHLCV 커버리지 리포트
    - region, symbol, interval
    - candles: 캔들 개수
    - first_dt, last_dt: 데이터 기간
    - days_covered: 커버 일수
    """
    conn = sqlite3.connect(DB_PATH)
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
    conn.close()

    if not df.empty:
        df["first_dt"] = pd.to_datetime(df["first_dt"])
        df["last_dt"] = pd.to_datetime(df["last_dt"])
        df["days_covered"] = (df["last_dt"] - df["first_dt"]).dt.days + 1

    return df


def get_last_universe_backfill_time(db: BotDatabase) -> str:
    """
    마지막 UNIVERSE OHLCV 백필 완료 시각
    - oh_universe_backfill.py 실행 끝에서 settings에 저장한 값 사용
    - 반환값: "YYYY-MM-DD HH:MM:SS" 또는 "" (없을 경우)
    """
    value = db.get_setting("last_universe_ohlcv_backfill", "")
    return value


def get_recent_backfill_failures(limit: int = 20) -> pd.DataFrame:
    """
    최근 백필 실패/데이터 없음 종목 리스트
    - universe_backfill_failures 테이블 기준
    - BotDatabase.init_db()에서 테이블 생성된다는 전제
    """
    conn = sqlite3.connect(DB_PATH)
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
            ORDER BY datetime(created_at) DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )
    except Exception:
        # 혹시라도 다른 이유로 실패하면 빈 DataFrame 반환
        df = pd.DataFrame(
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
    finally:
        conn.close()

    return df
