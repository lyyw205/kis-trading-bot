# build_ml_seq_samples.py (수정 완료)
import os
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import UNIVERSE_STOCKS
from db import BotDatabase

DB_PATH = "trading.db"

SEQ_LEN = 30
LOOKBACK = 20
BAND_PCT = 0.005

FUTURE_WINDOW = 20 
TP_RATE = 0.02
SL_RATE = -0.02


# -----------------------------------------------------------
# 헬퍼 함수
# -----------------------------------------------------------
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def load_ohlcv_all():
    conn = sqlite3.connect(DB_PATH)
    # 필요한 모든 컬럼 가져오기
    df = pd.read_sql_query(
        """
        SELECT region, symbol, interval, dt,
               open, high, low, close, volume
        FROM ohlcv_data
        ORDER BY region, symbol, interval, dt
        """,
        conn,
    )
    conn.close()

    if df.empty:
        return {}

    df["dt"] = pd.to_datetime(df["dt"])

    groups = {}
    for (region, symbol, interval), g in df.groupby(["region", "symbol", "interval"], sort=False):
        g = g.copy()
        g = g.sort_values("dt")
        g.set_index("dt", inplace=True)
        # 문자열이 섞여있을 수 있으므로 강제 형변환
        g = g[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors='coerce')
        groups[(region, symbol, interval)] = g

    return groups


def calc_label(future_df, entry_price):
    for _, row in future_df.iterrows():
        price = row["close"]
        profit = (price - entry_price) / entry_price

        if profit >= TP_RATE:
            return 1
        if profit <= SL_RATE:
            return 0
    return 0


def build_samples(ohlcv_dict):
    samples = []

    for t in UNIVERSE_STOCKS:
        region = t["region"]
        symbol = t["symbol"]
        interval = "5m"

        key = (region, symbol, interval)
        if key not in ohlcv_dict:
            continue

        df = ohlcv_dict[key].copy()
        if len(df) < 60: 
            continue

        # -----------------------------------------------------------
        # 1. 지표 계산 (Vectorized Operation)
        # -----------------------------------------------------------
        df["support"] = df["low"].rolling(LOOKBACK).min()
        df["ma20"] = df["close"].rolling(20).mean()
        df["ma60"] = df["close"].rolling(60).mean()
        df["rsi"] = calculate_rsi(df["close"], 14)
        df["vol_ma20"] = df["volume"].rolling(20).mean()

        # 이전 봉 데이터 (shift 사용)
        df["close_prev"] = df["close"].shift(1)
        df["vol_prev"] = df["volume"].shift(1)

        # -----------------------------------------------------------
        # 2. 기본 조건
        # -----------------------------------------------------------
        df["is_bullish"] = df["close"] > df["open"]
        df["price_up"] = df["close"] > df["close_prev"]
        
        # -----------------------------------------------------------
        # 전략 A: Reversal (역추세)
        # -----------------------------------------------------------
        df["at_support"] = df["low"] <= df["support"] * (1 + BAND_PCT)
        sig_reversal = df["at_support"] & df["is_bullish"] & df["price_up"]

        # -----------------------------------------------------------
        # 전략 B: Momentum Strong (추세추종)
        # -----------------------------------------------------------
        # 1. 정배열 (Close > 20 > 60)
        cond_align = (df["close"] > df["ma20"]) & (df["ma20"] > df["ma60"])
        
        # 2. RSI (50~75, 상한선 80으로 완화 가능)
        # (trader.py에서는 80까지 봐줬으니 여기도 80으로 맞추는 게 좋음)
        cond_rsi = (df["rsi"] >= 50) & (df["rsi"] <= 80)
        
        # 3. 거래량 (이전 봉 OR 현재 봉)
        # prev['volume'] > last['vol_ma20']  ---> df['vol_prev'] > df['vol_ma20']
        # last['volume'] > last['vol_ma20']*0.2 ---> df['volume'] > df['vol_ma20']*0.2
        # (주의: 여기서 vol_ma20은 현재 봉까지 포함된 평균이지만, 큰 오차 없으므로 사용)
        cond_vol = (df["vol_prev"] > df["vol_ma20"]) | \
                   (df["volume"] > df["vol_ma20"] * 0.2)
        
        sig_momentum = cond_align & cond_rsi & cond_vol & df["is_bullish"]

        # -----------------------------------------------------------
        # 3. 최종 신호 (Signal Merge)
        # -----------------------------------------------------------
        df["entry_signal"] = sig_reversal | sig_momentum

        # -----------------------------------------------------------
        # 4. 샘플 추출
        # -----------------------------------------------------------
        entry_points = df[df["entry_signal"]].dropna() # NaN 제거

        for dt_entry, row in entry_points.iterrows():
            entry_price = row["close"]

            # 인덱스 위치 찾기
            try:
                idx = df.index.get_loc(dt_entry)
            except KeyError:
                continue
            
            # 미래 데이터 확인
            if idx + 1 + FUTURE_WINDOW > len(df):
                continue

            future_df = df.iloc[idx + 1 : idx + 1 + FUTURE_WINDOW]
            if future_df.empty:
                continue

            label = calc_label(future_df, entry_price)

            samples.append(
                {
                    "region": region,
                    "symbol": symbol,
                    "interval": interval,
                    "dt_entry": dt_entry.strftime("%Y-%m-%d %H:%M:%S"),
                    "label": int(label),
                }
            )

    return samples


def save_samples(samples):
    if not samples:
        print("❌ 생성된 샘플이 없습니다.")
        return

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ml_seq_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region TEXT,
            symbol TEXT,
            interval TEXT,
            dt_entry TEXT,
            label INTEGER
        )
        """
    )
    cur.execute("DELETE FROM ml_seq_samples")

    cur.executemany(
        """
        INSERT INTO ml_seq_samples (region, symbol, interval, dt_entry, label)
        VALUES (:region, :symbol, :interval, :dt_entry, :label)
        """,
        samples
    )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    db = BotDatabase(DB_PATH)
    db.log(" build_ml_seq_samples.py 시작 (Reversal + Momentum)")

    ohlcv_dict = load_ohlcv_all()
    if not ohlcv_dict:
        print(" OHLCV가 없습니다. build_ohlcv_history.py 먼저 실행하세요.")
        raise SystemExit

    samples = build_samples(ohlcv_dict)
    print(f" 생성된 샘플 수: {len(samples)}")

    save_samples(samples)

    db.log(f"🎉 ml_seq_samples 생성 완료: {len(samples)}개")