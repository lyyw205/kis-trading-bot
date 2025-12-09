""" Binance 코인 5분봉 OHLCV 백필 스크립트 (tcn_backfill.py)

 - Binance Spot / Futures 에서 5분봉 캔들 데이터를 가져와
   내부 DB(ohlcv_data)에 KST 기준으로 저장하는 전용 백필 유틸리티.

주요 기능:
1) fetch_binance_5m_ohlcv_kst()
   - Binance REST API (Spot / Futures)에서 5분 봉 데이터를 조회
   - 최초 백필 시: 최근 구간부터 과거로 최대 max_initial 개까지 수집
   - 증분 백필 시: DB에 저장된 마지막 시각 이후 구간만 앞으로 이어서 수집
   - UTC 타임스탬프를 KST 로 변환하여 DatetimeIndex(dt) 로 정리

2) backfill_cr_ohlcv_binance()
   - BI_UNIVERSE_STOCKS (region="BI", symbol="BTCUSDT" 등)를 순회
   - BotDatabase.get_last_ohlcv_dt()로 마지막 저장 시각 조회
   - Binance에서 신규 5분봉만 가져와 중복 제거 후 ohlcv_data 테이블에 저장
   - 심볼별 개수/기간 로그 기록 + 마지막 백필 시각을 settings("last_cr_backfill")에 저장

※ 코멘트의 함수명은 'CR' 이지만, 실제로는 region="BI" (Binance 코인 유니버스) 전용 백필 스크립트.
"""

import time
from datetime import datetime
from typing import Optional

import pandas as pd
from binance.client import Client
from binance.enums import KLINE_INTERVAL_5MINUTE

from c_db_manager import BotDatabase
from c_config import BI_UNIVERSE_STOCKS  # region="BI", symbol="BTCUSDT" 형태


# -------------------------------------------
# 0. Binance 클라이언트
# -------------------------------------------
BINANCE_API_KEY = ""      # 필요하면 채우기
BINANCE_API_SECRET = ""   # 필요하면 채우기

binance_client = Client(api_key=BINANCE_API_KEY, api_secret=BINANCE_API_SECRET)

KST_TZ = "Asia/Seoul"

# 한 심볼에 대해 "최초 백필 시" 최대 몇 개까지 가져올지
MAX_INITIAL_BARS = 50000   


# -------------------------------------------
# 1. 바이낸스에서 5분봉 가져오기
# -------------------------------------------
def fetch_binance_5m_ohlcv_kst(
    symbol: str,
    since_utc: Optional[datetime] = None,
    max_initial: int = MAX_INITIAL_BARS,
    market: str = "spot",
) -> pd.DataFrame:
    
    # 엔드포인트 선택 (Spot / Futures)
    if market == "spot":
        get_fn = binance_client.get_klines
    elif market == "futures":
        get_fn = binance_client.futures_klines
    else:
        raise ValueError(f"Unknown market: {market}")

    all_rows = []

    # -------------------------
    # A) 증분 모드: since_utc 이후 앞으로 쭉 (과거 -> 현재)
    # -------------------------
    if since_utc is not None:
        start_time_ms = int(since_utc.timestamp() * 1000) + 1
        limit = 1000

        while True:
            klines = get_fn(
                symbol=symbol,
                interval=KLINE_INTERVAL_5MINUTE,
                startTime=start_time_ms,
                limit=limit,
            )
            if not klines:
                break

            all_rows.extend(klines)

            if len(klines) < limit:
                break

            last_open_time = klines[-1][0]
            start_time_ms = last_open_time + 5 * 60 * 1000
            time.sleep(0.15)

    # -------------------------
    # B) 초기 모드: 최근 데이터 기준 -> 과거로 거슬러 올라가기
    # -------------------------
    else:
        remaining = max_initial
        end_time_ms = None

        while remaining > 0:
            batch_limit = min(1000, remaining)

            if end_time_ms is None:
                # 가장 최근 구간
                klines = get_fn(
                    symbol=symbol,
                    interval=KLINE_INTERVAL_5MINUTE,
                    limit=batch_limit,
                )
            else:
                # 더 과거 구간
                klines = get_fn(
                    symbol=symbol,
                    interval=KLINE_INTERVAL_5MINUTE,
                    endTime=end_time_ms,
                    limit=batch_limit,
                )

            if not klines:
                break

            # 과거 데이터를 리스트 앞에 붙임
            all_rows = klines + all_rows
            remaining -= len(klines)

            if len(klines) < batch_limit:
                break

            oldest_open_time = klines[0][0]
            end_time_ms = oldest_open_time - 1
            time.sleep(0.15)

    # ---------------------------------------------------------
    # 결과 처리 (들여쓰기 주의: if/else 블록 바깥으로 나와야 함)
    # ---------------------------------------------------------
    if not all_rows:
        return pd.DataFrame()

    # Binance kline 구조 파싱
    # [ open_time, open, high, low, close, volume, ... ]
    opens = [float(row[1]) for row in all_rows]
    highs = [float(row[2]) for row in all_rows]
    lows = [float(row[3]) for row in all_rows]
    closes = [float(row[4]) for row in all_rows]
    volumes = [float(row[5]) for row in all_rows]

    # UTC → KST 변환
    dt_utc = pd.to_datetime([row[0] for row in all_rows], unit="ms", utc=True)
    dt_kst = dt_utc.tz_convert(KST_TZ).tz_localize(None)  # KST naive

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        },
        index=pd.DatetimeIndex(dt_kst, name="dt"),
    )

    df = df.dropna().sort_index().drop_duplicates()

    return df


# -------------------------------------------
# 2. CR 코인 5분봉 백필 (Binance 전용)
# -------------------------------------------
def backfill_cr_ohlcv_binance():
    db = BotDatabase()
    db.log("📦 코인 5분봉 백필 시작 (Source: Binance -> DB KST)")

    interval = "5m" 

    for t in BI_UNIVERSE_STOCKS:
        region = t["region"]   # "BI"
        symbol = t["symbol"]   # "BTCUSDT"

        if region != "BI":
            continue

        db.log(f"⏳ 백필 시작: {symbol} (Binance)")

        try:
            # 0) DB에서 마지막 저장된 dt 확인
            last_dt_str = db.get_last_ohlcv_dt(region, symbol, interval)

            last_dt_kst = None
            last_dt_utc = None

            if last_dt_str:
                last_dt_kst = pd.to_datetime(last_dt_str)
                if last_dt_kst.tzinfo is None:
                    last_dt_kst = last_dt_kst.tz_localize(KST_TZ)
                else:
                    last_dt_kst = last_dt_kst.tz_convert(KST_TZ)

                last_dt_utc = last_dt_kst.tz_convert("UTC").to_pydatetime()

            # 1) 바이낸스 API 호출
            market = t.get("market", "spot")
            df = fetch_binance_5m_ohlcv_kst(
                symbol=symbol,
                since_utc=last_dt_utc,
                market=market,
            )

            if df is None or df.empty:
                db.log(f"⚠️ 새 데이터 없음: {symbol}")
                continue

            # 중복 방지 필터
            if last_dt_kst is not None:
                last_dt_kst_naive = last_dt_kst.tz_localize(None)
                df = df[df.index > last_dt_kst_naive]
                if df.empty:
                    db.log(f"⏭ 새 5분봉 없음 (Last: {last_dt_kst_naive}) → 스킵: {symbol}")
                    continue

            first_ts = df.index.min()
            last_ts = df.index.max()
            db.log(f"📏 {symbol}: {len(df)}개 확보 (KST {first_ts} ~ {last_ts})")

            # 2) DB 저장
            db.save_ohlcv_df(region, symbol, interval, df)
            db.log(f"✅ 저장 완료: {symbol}")

        except Exception as e:
            db.log(f"❌ {symbol} 처리 중 에러: {e}")
            continue

        time.sleep(0.3)

    db.set_setting(
        "last_cr_backfill",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    db.log("🎉 코인 데이터 백필 작업 완료 (Source: Binance, KST 저장)")


if __name__ == "__main__":
    backfill_cr_ohlcv_binance()