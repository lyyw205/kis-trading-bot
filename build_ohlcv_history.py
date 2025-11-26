# build_ohlcv_history.py
import time
import sqlite3
from datetime import datetime

import pandas as pd

from db import BotDatabase
from kis_api import KisDataFetcher
from config import APP_KEY, APP_SECRET, ACCOUNT_NO, UNIVERSE_STOCKS

DB_PATH = "trading.db"


# =========================================================
# 1) 백필 실패 기록 테이블 관련 (대시보드/로그용)
# =========================================================
def ensure_universe_backfill_fail_table():
    """UNIVERSE 백필 실패 내역 기록 테이블 생성"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS universe_backfill_failures (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region TEXT,
            symbol TEXT,
            excd TEXT,
            interval TEXT,
            error_type TEXT,      -- 'fetch_error' / 'empty_data'
            error_message TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def insert_backfill_failure(region, symbol, excd, interval, error_type, error_message):
    """백필 실패시 universe_backfill_failures 테이블에 기록"""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO universe_backfill_failures (
            region, symbol, excd, interval, error_type, error_message, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (
        region,
        symbol,
        excd,
        interval,
        error_type,
        str(error_message)[:200],  # 너무 길면 잘라냄
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    ))
    conn.commit()
    conn.close()


# =========================================================
# 2) 대시보드용 조회 함수들
#    (다른 코드/대시보드에서 import 해서 사용)
# =========================================================
def get_universe_coverage():
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


def get_last_universe_backfill_time(db: BotDatabase):
    """
    마지막 UNIVERSE OHLCV 백필 완료 시각
    - build_ohlcv_history.py 실행 끝에서 settings에 저장한 값 사용
    """
    value = db.get_setting("last_universe_ohlcv_backfill", "")
    return value  # "YYYY-MM-DD HH:MM:SS" 또는 ""


def get_recent_backfill_failures(limit: int = 20):
    """
    최근 백필 실패/데이터 없음 종목 리스트
    - universe_backfill_failures 테이블 기준
    """
    # ✅ 테이블이 없을 수도 있으니 여기서 한 번 보장해줌
    ensure_universe_backfill_fail_table()

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


# =========================================================
# 3) 메인: UNIVERSE OHLCV 백필 실행
# =========================================================
if __name__ == "__main__":
    db = BotDatabase(DB_PATH)
    db.log("📦 [UNIVERSE] OHLCV 과거 데이터 백필 시작")

    # 실패 기록 테이블 준비
    ensure_universe_backfill_fail_table()

    fetcher = KisDataFetcher(APP_KEY, APP_SECRET, ACCOUNT_NO, mode="real", logger=db.log)
    

    # 5분봉 개수 설정 (대략 1달치 수준)
    KR_COUNT = 1600
    US_COUNT = 1600

    for t in UNIVERSE_STOCKS:
        region = t["region"]
        symbol = t["symbol"]
        excd = t.get("excd")
        interval = "5m"

        db.log(f"⏳ 백필: {region} {symbol} ({excd})")

        try:
            if region == "KR":
                df = fetcher.get_ohlcv(
                    region,
                    symbol,
                    interval=interval,
                    count=KR_COUNT,
                )
            else:
                df = fetcher.get_ohlcv(
                    region,
                    symbol,
                    excd,
                    interval=interval,
                    count=US_COUNT,
                )

        except Exception as e:
            db.log(f"⚠️ OHLCV 조회 실패: {region} {symbol} | {e}")
            # 대시보드용 실패 기록
            insert_backfill_failure(region, symbol, excd, interval, "fetch_error", e)
            continue

        if df is None or df.empty:
            db.log(f"⚠️ 데이터 없음: {region} {symbol}")
            # 데이터 없음도 기록
            insert_backfill_failure(region, symbol, excd, interval, "empty_data", "no rows")
            continue

        # 실제 범위/개수 로그
        try:
            first_ts = df.index.min()
            last_ts = df.index.max()
            db.log(
                f"📏 {region} {symbol}: 5분봉 {len(df)}개 | "
                f"from {first_ts} → {last_ts}"
            )
        except Exception:
            db.log(f"📏 {region} {symbol}: 5분봉 {len(df)}개 (index 정보 없음)")

        # ohlcv_data 테이블에 저장 (BotDatabase.save_ohlcv_df 구현에 따름)
        db.save_ohlcv_df(region, symbol, interval, df)
        db.log(f"✅ 백필 완료: {region} {symbol} ({len(df)}개 저장)")

        time.sleep(0.2)

    # 마지막 백필 완료 시각 기록 → 대시보드 상단에 표시
    db.set_setting(
        "last_universe_ohlcv_backfill",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    db.log("🎉 [UNIVERSE] OHLCV 과거 데이터 백필 전체 완료")
