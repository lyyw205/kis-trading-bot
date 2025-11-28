# build_ohlcv_history_yf.py
"""
yfinance로 UNIVERSE_STOCKS의 과거 5분봉(최대 1~2개월)을 가져와서
ohlcv_data에 "KIS 데이터와 겹치지 않는 과거 구간만" 백필하는 스크립트.

- 기존 KIS 5분봉은 최근 구간만 존재 (당일/전일 수준)
- 이 파일은 그 앞 구간(과거 히스토리)을 채워 넣는 용도
"""

import time
import sqlite3
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf

from db import BotDatabase
from config import UNIVERSE_STOCKS, APP_KEY, APP_SECRET, ACCOUNT_NO

DB_PATH = "trading.db"
INTERVAL = "5m"   # DB의 interval 컬럼 값과 통일


# ----------------------------------------------------
# 1. 심볼 매핑: UNIVERSE_STOCKS -> yfinance 심볼
#    (필요하면 여기 커스터마이징)
# ----------------------------------------------------
def to_yf_symbol(region: str, symbol: str, excd: str | None = None) -> str | None:
    """
    region/symbol 정보를 yfinance 티커 문자열로 변환.
    - 지금은 KR만 가정해서 ".KS"로 붙여둠. 필요하면 직접 조정.
    - US는 그대로 사용 (예: AAPL, TSLA 등)
    """
    if region == "KR":
        # 코스피: .KS, 코스닥: .KQ 인데
        # 지금은 일단 전부 .KS로 가정하고, 나중에 필요하면 분리
        return f"{symbol}.KS"
    elif region == "US":
        # AAPL, TSLA 처럼 그대로 사용
        return symbol
    else:
        return None


# ----------------------------------------------------
# 2. DB에서 해당 종목/인터벌의 "가장 최근 dt" 가져오기
# ----------------------------------------------------
def get_last_dt(conn: sqlite3.Connection, region: str, symbol: str, interval: str) -> datetime | None:
    query = """
        SELECT MAX(dt) AS last_dt
        FROM ohlcv_data
        WHERE region = ? AND symbol = ? AND interval = ?
    """
    df = pd.read_sql_query(query, conn, params=(region, symbol, interval))
    if df.empty or df["last_dt"].iloc[0] is None:
        return None
    return pd.to_datetime(df["last_dt"].iloc[0])


# ----------------------------------------------------
# 3. yfinance에서 5분봉 1~2개월 분량 가져오기
# ----------------------------------------------------
def fetch_yf_5m(yf_symbol: str, months: int = 1) -> pd.DataFrame:
    """
    yfinance에서 5분봉 데이터 다운로드.
    - period는 최대 60d까지 가능하다고 알려져 있어서 넉넉히 60d 요청 후
      실제로는 months 인자(1개월 등)에 맞게 잘라 쓸 수 있음.
    """
    # yfinance는 period="60d" + interval="5m" 조합 자주 사용
    period = "60d"  # 넉넉히 받아서 안쪽에서 잘라 쓰기
    try:
        df = yf.download(
            yf_symbol,
            interval="5m",
            period=period,
            progress=False,
            auto_adjust=False,
        )
    except Exception as e:
        print(f"❌ [yfinance 에러] {yf_symbol} | {e}")
        return pd.DataFrame()

    if df.empty:
        print(f"⚠️ [yfinance 데이터 없음] {yf_symbol}")
        return pd.DataFrame()

    # 인덱스(utc/타임존) 제거하고, 컬럼명 통일
    if df.index.tz is not None:
        df.index = df.index.tz_convert("Asia/Seoul").tz_localize(None)
    else:
        df.index = pd.to_datetime(df.index)

    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    # 우리가 쓸 컬럼만 남기기
    df = df[["open", "high", "low", "close", "volume"]].copy()
    df = df.dropna()

    # months 인자에 맞게 최근 N개월만 슬라이싱 (기본 1개월)
    now = datetime.now()
    start_cut = now - timedelta(days=30 * months)
    df = df[df.index >= start_cut]

    return df


# ----------------------------------------------------
# 4. 메인: UNIVERSE_STOCKS에 대해 과거 구간만 백필
# ----------------------------------------------------
if __name__ == "__main__":
    db = BotDatabase(DB_PATH)
    db.log("📦 [YF] UNIVERSE 5분봉 과거 데이터 백필 시작")

    conn = sqlite3.connect(DB_PATH)

    for t in UNIVERSE_STOCKS:
        region = t.get("region", "KR")
        symbol = t["symbol"]
        excd = t.get("excd")  # 지금은 KR만 있으니까 거의 None일 것

        yf_symbol = to_yf_symbol(region, symbol, excd)
        if yf_symbol is None:
            db.log(f"⚠️ [YF 스킵] region={region}, symbol={symbol} → 매핑 실패")
            continue

        db.log(f"⏳ [YF] 백필 대상: {region} {symbol} (yf={yf_symbol})")

        # 1) DB에 이미 있는 가장 최근 dt 확인
        last_dt = get_last_dt(conn, region, symbol, INTERVAL)
        if last_dt is not None:
            db.log(f"   └ DB 최신 dt: {last_dt}")
        else:
            db.log("   └ DB 최신 dt 없음 (처음 백필)")

        # 2) yfinance에서 5분봉 받아오기
        df_yf = fetch_yf_5m(yf_symbol, months=1)  # 1개월 기준
        if df_yf.empty:
            db.log(f"⚠️ [YF] 데이터 없음: {region} {symbol}")
            continue

        # 3) 이미 DB에 있는 구간과 겹치지 않게 "과거 구간만" 남기기
        if last_dt is not None:
            df_yf = df_yf[df_yf.index < last_dt]

        if df_yf.empty:
            db.log(f"ℹ️ [YF] {region} {symbol}: 추가할 과거 구간 없음 (모두 겹침)")
            continue

        # 4) 저장 (BotDatabase.save_ohlcv_df: region, symbol, interval, df)
        try:
            db.save_ohlcv_df(region, symbol, INTERVAL, df_yf)
            db.log(
                f"✅ [YF 저장완료] {region} {symbol}: 5분봉 {len(df_yf)}개 백필 "
                f"({df_yf.index.min()} → {df_yf.index.max()})"
            )
        except Exception as e:
            db.log(f"❌ [YF 저장에러] {region} {symbol} | {e}")

        time.sleep(0.2)

    conn.close()

    db.log("🎉 [YF] UNIVERSE 5분봉 과거 데이터 백필 전체 완료")
