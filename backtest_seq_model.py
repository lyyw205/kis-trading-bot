# backtest_seq_model.py
"""
새로 학습된 모델을 백테스트하는 스크립트.
train_seq_model.py → settings.active_model_path 에 저장된 모델을 자동으로 사용한다.

※ 백테스트 핵심 흐름
1) settings.active_model_path 에서 모델 경로 로드
2) DB에서 universe OHLCV 불러오기
3) 시퀀스(SEQ_LEN) 기반 feature 생성
4) 모델로 예측
5) TP/SL 또는 FUTURE_WINDOW 기반으로 실제 결과 계산하여 PnL 기록
6) backtests 테이블에 백테스트 결과 저장
"""

import os
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib

from db import BotDatabase
from config import UNIVERSE_STOCKS

DB_PATH = "trading.db"
SEQ_LEN = 30
FUTURE_WINDOW = 20
TP_RATE = 0.02
SL_RATE = -0.02


# -------------------------------------------------------
# OHLCV 로드
# -------------------------------------------------------
def load_all_ohlcv():
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
    for (region, symbol, interval), g in df.groupby(["region", "symbol", "interval"]):
        g = g.sort_values("dt")
        g = g.set_index("dt")
        # 타입 강제 변환
        g = g[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
        groups[(region, symbol, interval)] = g

    return groups


# -------------------------------------------------------
# Feature builder (train_seq_model.py와 동일)
# -------------------------------------------------------
def build_feature_from_seq(df_seq):
    if len(df_seq) != SEQ_LEN:
        return None

    close = df_seq["close"].values
    high = df_seq["high"].values
    low = df_seq["low"].values
    vol = df_seq["volume"].values

    base = close[0]
    if base <= 0:
        return None

    close_rel = close / base - 1.0
    high_rel = high / base - 1.0
    low_rel = low / base - 1.0
    vol_mean = vol.mean() if vol.mean() > 0 else 1.0
    vol_norm = vol / vol_mean

    feat = np.concatenate([close_rel, high_rel, low_rel, vol_norm])
    return feat


# -------------------------------------------------------
# 라벨 계산 (TP/SL 또는 FUTURE_WINDOW)
# -------------------------------------------------------
def calc_label(future_df, entry_price):
    for _, row in future_df.iterrows():
        price = row["close"]
        pnl = (price - entry_price) / entry_price

        if pnl >= TP_RATE:
            return 1
        if pnl <= SL_RATE:
            return 0
    return 0


# -------------------------------------------------------
# 백테스트 실행
# -------------------------------------------------------
def run_backtest(model, ohlcv_dict):
    rows = []

    for t in UNIVERSE_STOCKS:
        region = t["region"]
        symbol = t["symbol"]
        interval = "5m"

        key = (region, symbol, interval)
        if key not in ohlcv_dict:
            continue

        df = ohlcv_dict[key]
        if len(df) < 100:
            continue

        for i in range(SEQ_LEN, len(df) - FUTURE_WINDOW):
            df_seq = df.iloc[i - SEQ_LEN : i]
            entry_price = df.iloc[i]["close"]
            future_df = df.iloc[i + 1 : i + 1 + FUTURE_WINDOW]

            feat = build_feature_from_seq(df_seq)
            if feat is None:
                continue

            pred = model.predict([feat])[0]

            # 예측이 1 이면 진입했다고 가정 → 실제 결과 계산
            if pred == 1:
                label = calc_label(future_df, entry_price)
                pnl = (future_df.iloc[-1]["close"] - entry_price) / entry_price

                rows.append(
                    {
                        "region": region,
                        "symbol": symbol,
                        "entry_time": df.index[i],
                        "entry_price": entry_price,
                        "label": int(label),
                        "pnl": float(pnl),
                    }
                )

    return pd.DataFrame(rows)


# -------------------------------------------------------
# 메인 실행
# -------------------------------------------------------
if __name__ == "__main__":
    db = BotDatabase(DB_PATH)
    db.log("백테스트 시작")

    # 1) settings에서 현재 모델 경로 가져오기
    model_path = db.get_setting("active_model_path", None)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"active_model_path 없음: {model_path}")

    model = joblib.load(model_path)
    print(f"모델 로드 완료: {model_path}")

    # 2) OHLCV 로드
    ohlcv_dict = load_all_ohlcv()
    if not ohlcv_dict:
        raise RuntimeError("ohlcv_data가 비어 있음. 먼저 build_ohlcv_history.py 실행 필요.")

    # 3) 백테스트 실행
    df_bt = run_backtest(model, ohlcv_dict)
    print(f"백테스트 진입 수: {len(df_bt)}")

    # 요약 통계
    if df_bt.empty:
        db.log("백테스트 결과 없음: 샘플 부족 또는 모델 불만족")
        exit(0)

    win_rate = (df_bt["label"] == 1).mean() * 100
    avg_profit = df_bt["pnl"].mean()
    cum_return = df_bt["pnl"].sum()
    max_dd = df_bt["pnl"].min()

    print(f"승률: {win_rate:.2f}%")
    print(f"평균 수익률: {avg_profit:.5f}")
    print(f"누적 수익률: {cum_return:.5f}")
    print(f"최대 손실(pnl): {max_dd:.5f}")

    # 4) DB 저장
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cur.execute(
            """
            INSERT INTO backtests
            (model_id, start_date, end_date,
             trades, win_rate, avg_profit,
             cum_return, max_dd, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                None,  # model_versions와 연결할 경우 여기에 model_id를 넣을 수 있음
                df_bt["entry_time"].min().strftime("%Y-%m-%d"),
                df_bt["entry_time"].max().strftime("%Y-%m-%d"),
                int(len(df_bt)),
                float(win_rate),
                float(avg_profit),
                float(cum_return),
                float(max_dd),
                f"Backtest for {model_path}",
            ),
        )

        conn.commit()
        conn.close()

        db.log("백테스트 완료 및 DB 저장 완료")
    except Exception as e:
        db.log(f"백테스트 저장 실패: {e}")
