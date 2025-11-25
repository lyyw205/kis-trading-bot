# build_ml_samples.py
import sqlite3
from datetime import datetime
import pandas as pd

from db import BotDatabase
from kis_api import KisDataFetcher
from config import APP_KEY, APP_SECRET, ACCOUNT_NO, TARGET_STOCKS

DB_PATH = "trading.db"

# 전략 파라미터 (지금 실시간 봇이랑 맞춰서)
LOOKBACK = 120
BAND_PCT = 0.01

# 레이블링 룰
TP_PCT = 0.03   # +3% 익절
SL_PCT = -0.04  # -4% 손절

def ensure_ml_table():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region TEXT,
            symbol TEXT,
            dt TEXT,
            price REAL,
            at_support INTEGER,
            is_bullish INTEGER,
            price_up INTEGER,
            lookback INTEGER,
            band_pct REAL,
            label INTEGER            -- 1: 좋은 진입, 0: 나쁜 진입
        )
    """)
    # 매번 새로 만들고 싶으면 아래 주석 해제 (기존 자료 삭제)
    cur.execute("DELETE FROM ml_samples")
    conn.commit()
    conn.close()

def save_samples(rows):
    if not rows:
        return
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executemany("""
        INSERT INTO ml_samples
        (region, symbol, dt, price,
         at_support, is_bullish, price_up,
         lookback, band_pct, label)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()

if __name__ == "__main__":
    db = BotDatabase(DB_PATH)
    ensure_ml_table()

    # 모의투자 모드로 과거 데이터만 땡겨오면 됨
    fetcher = KisDataFetcher(APP_KEY, APP_SECRET, ACCOUNT_NO, mode="virtual")

    total_rows = 0

    for t in TARGET_STOCKS:
        region = t["region"]
        symbol = t["symbol"]
        excd = t.get("excd")

        db.log(f"📈 ML 샘플 생성 시작: {region} {symbol} ({excd})")

        try:
            # 실시간 봇이 쓰는 것과 동일한 함수 사용
            df = fetcher.get_ohlcv(region, symbol, excd)
        except Exception as e:
            db.log(f"⚠️ OHLCV 조회 실패: {region} {symbol} | {e}")
            continue

        if df is None or df.empty:
            db.log(f"⚠️ 데이터 없음: {region} {symbol}")
            continue

        # 전략 지표 계산 (실시간 로직과 동일)
        df["support"] = df["low"].rolling(LOOKBACK).min()
        df["at_support"] = df["low"] <= df["support"] * (1 + BAND_PCT)
        df["is_bullish"] = df["close"] > df["open"]
        df["price_up"] = df["close"] > df["close"].shift(1)

        closes = df["close"].values
        highs = df["high"].values
        lows = df["low"].values

        # KR = 5분봉, US = 일봉이니까 미래 구간 길이 다르게
        if region == "KR":
            future_bars = 12   # 5분봉 12개 ≒ 1시간
        else:
            future_bars = 5    # 5일

        rows = []
        # lookback 이후부터, 미래 future_bars 만큼 여유 있는 구간까지만
        for i in range(LOOKBACK, len(df) - future_bars):
            row = df.iloc[i]

            # 룰 기준 진입 신호가 뜬 구간만 샘플로 사용
            if not (row["at_support"] and row["is_bullish"] and row["price_up"]):
                continue

            entry_price = closes[i]
            future_high = highs[i+1 : i+1+future_bars].max()
            future_low = lows[i+1 : i+1+future_bars].min()

            tp_hit = future_high >= entry_price * (1 + TP_PCT)
            sl_hit = future_low <= entry_price * (1 + SL_PCT)

            # 둘 다 안 맞으면 애매하니까 스킵
            if not tp_hit and not sl_hit:
                continue

            # TP 먼저 맞았다고 가정 (보수적으로 손절 먼저 고려하고 싶으면 로직 더 세분화 가능)
            label = 1 if tp_hit and not sl_hit else 0

            rows.append([
                region,
                symbol,
                row.name.strftime("%Y-%m-%d %H:%M:%S"),
                float(row["close"]),
                int(row["at_support"]),
                int(row["is_bullish"]),
                int(row["price_up"]),
                LOOKBACK,
                BAND_PCT,
                int(label),
            ])

        save_samples(rows)
        total_rows += len(rows)
        db.log(f"✅ {region} {symbol} 샘플 {len(rows)}개 저장 (누적 {total_rows}개)")

    db.log("🎉 ML 샘플 생성 완료 (ml_samples 테이블)")
