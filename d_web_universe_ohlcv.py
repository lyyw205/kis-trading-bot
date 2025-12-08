# "UNIVERSE OHLCV 커버리지/백필 상태 대시보드 헬퍼 (sqlite, trading.db 기반)

#  - ohlcv_data / universe_backfill_failures / settings를 읽어서
#    UNIVERSE 종목들의 OHLCV 데이터 커버리지와 백필 실패 현황을 리포트용으로 제공하는 모듈

# 주요 기능:
# 1) get_universe_coverage()
#    - trading.db (sqlite)에서 ohlcv_data 테이블을 집계
#    - region, symbol, interval 별로:
#      · candles: 캔들 개수(COUNT(*))
#      · first_dt: 최소 dt
#      · last_dt: 최대 dt
#      · days_covered: first_dt~last_dt 일수 (일수+1)
#    - 결과를 DataFrame으로 반환하여, 대시보드에서
#      “어느 심볼·타임프레임에 데이터가 얼마나 쌓여있는지” 확인하는 용도로 사용

# 2) get_last_universe_backfill_time(db)
#    - BotDatabase의 settings 테이블에서
#      key="last_universe_ohlcv_backfill" 값을 읽어 반환
#    - oh_universe_backfill.py 실행 끝에서 저장해 둔
#      “마지막 UNIVERSE OHLCV 백필 완료 시각”을 문자열(YYYY-MM-DD HH:MM:SS)로 제공

# 3) get_recent_backfill_failures(limit)
#    - trading.db에서 universe_backfill_failures 테이블을 조회
#    - 최근 실패/데이터 없음 케이스를 created_at 기준 최신 순으로 max limit개 로드
#    - 컬럼:
#      · region, symbol, excd, interval, error_type, error_message, created_at
#    - 테이블이 없거나 쿼리 실패 시, 같은 컬럼 구조를 가진 빈 DataFrame 반환
#    - 어떤 종목/거래소/타임프레임에서 백필이 실패했는지 모니터링용으로 사용"


import sqlite3
from typing import Optional

import pandas as pd

from c_db_manager import BotDatabase

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
