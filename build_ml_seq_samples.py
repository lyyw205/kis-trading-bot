# build_ml_seq_samples.py
import os
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from config import UNIVERSE_STOCKS
from db import BotDatabase

DB_PATH = "trading.db"

SEQ_LEN = 30       # trader.py, train_seq_model.py 와 맞춰야 함
LOOKBACK = 20
BAND_PCT = 0.005   # 지지선 밴드 허용치

FUTURE_WINDOW = 20 # 향후 20봉 내 TP/SL 판단
TP_RATE = 0.02     # +2%
SL_RATE = -0.02    # -2%


def load_ohlcv_all():
    conn = sqlite3.connect(DB_PATH)
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
        g = g[["open", "high", "low", "close", "volume"]].astype(float)
        groups[(region, symbol, interval)] = g

    return groups


def calc_label(future_df, entry_price):
    """
    TP / SL / TIMEOUT 기준으로 label 생성
    """
    for _, row in future_df.iterrows():
        price = row["close"]
        profit = (price - entry_price) / entry_price

        if profit >= TP_RATE:
            return 1
        if profit <= SL_RATE:
            return 0

    # 미래 구간 동안 TP/SL 둘 다 안 맞으면 실패(0)로 간주
    return 0


def build_samples(ohlcv_dict):
    samples = []

    for t in UNIVERSE_STOCKS:
        region = t["region"]
        symbol = t["symbol"]
        interval = "5m"  # 시퀀스/트레이더 모두 5m 기준

        key = (region, symbol, interval)
        if key not in ohlcv_dict:
            continue

        df = ohlcv_dict[key].copy()

        # 룰 기반 시그널 계산 (trader.py와 컨셉 맞춤)
        df["support"] = df["low"].rolling(LOOKBACK).min()
        df["at_support"] = df["low"] <= df["support"] * (1 + BAND_PCT)
        df["is_bullish"] = df["close"] > df["open"]
        df["price_up"] = df["close"] > df["close"].shift(1)

        df["entry_signal"] = df["at_support"] & df["is_bullish"] & df["price_up"]

        entry_points = df[df["entry_signal"]].copy()

        for dt_entry, row in entry_points.iterrows():
            entry_price = row["close"]

            # 미래 봉 확보
            idx = df.index.get_loc(dt_entry)
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
    # 매번 새로 만들고 싶으면 여기서 TRUNCATE
    cur.execute("DELETE FROM ml_seq_samples")

    for s in samples:
        cur.execute(
            """
            INSERT INTO ml_seq_samples (region, symbol, interval, dt_entry, label)
            VALUES (?, ?, ?, ?, ?)
            """,
            (s["region"], s["symbol"], s["interval"], s["dt_entry"], s["label"]),
        )

    conn.commit()
    conn.close()


if __name__ == "__main__":
    db = BotDatabase(DB_PATH)
    db.log("🧪 build_ml_seq_samples.py 시작")

    ohlcv_dict = load_ohlcv_all()
    if not ohlcv_dict:
        print("❌ OHLCV가 없습니다. universe_collector.py 먼저 실행하세요.")
        raise SystemExit

    samples = build_samples(ohlcv_dict)
    print(f"📦 생성된 샘플 수: {len(samples)}")

    save_samples(samples)

    db.log(f"🎉 ml_seq_samples 생성 완료: {len(samples)}개")
