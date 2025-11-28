# core_backtest.py (리팩토링 버전 예시)
"""
새로 학습된 모델을 백테스트하는 스크립트.
train_seq_model.py → settings.active_model_path 에 저장된 모델을 자동으로 사용한다.

개선 포인트:
1) OHLCV를 기간 필터로 줄여서 로드 가능 (BACKTEST_DAYS)
2) 심볼별로 시퀀스 feature를 벡터화해서 한 번에 생성
3) model.predict를 캔들마다가 아니라 '배치'로 한 번에 호출
"""

import os
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
from numpy.lib.stride_tricks import sliding_window_view

from db_manager import BotDatabase

DB_PATH = "trading.db"
SEQ_LEN = 30
FUTURE_WINDOW = 20
TP_RATE = 0.02
SL_RATE = -0.02

# -------------------------------------------------------
# OHLCV 로드 (+ 기간 필터)
# -------------------------------------------------------
def load_all_ohlcv(start_date: str | None = None, end_date: str | None = None):
    """
    ohlcv_data 전체를 로드하되, start_date/end_date로 기간을 줄일 수 있음.
    start_date, end_date 형식: 'YYYY-MM-DD' 또는 'YYYY-MM-DD HH:MM:SS'
    """
    conn = sqlite3.connect(DB_PATH)

    base_query = """
        SELECT region, symbol, interval, dt,
               open, high, low, close, volume
        FROM ohlcv_data
    """

    conditions = []
    params = []

    if start_date is not None:
        conditions.append("dt >= ?")
        # 시간까지 붙이고 싶으면 'YYYY-MM-DD 00:00:00' 형식으로 넘겨도 됨
        params.append(start_date)

    if end_date is not None:
        conditions.append("dt <= ?")
        params.append(end_date)

    if conditions:
        base_query += " WHERE " + " AND ".join(conditions)

    base_query += " ORDER BY region, symbol, interval, dt"

    df = pd.read_sql_query(base_query, conn, params=params)
    conn.close()

    if df.empty:
        return {}

    df["dt"] = pd.to_datetime(df["dt"])

    groups = {}
    for (region, symbol, interval), g in df.groupby(["region", "symbol", "interval"]):
        g = g.sort_values("dt").set_index("dt")
        # 타입 강제 변환
        g = g[["open", "high", "low", "close", "volume"]].apply(
            pd.to_numeric, errors="coerce"
        )
        groups[(region, symbol, interval)] = g

    return groups


# -------------------------------------------------------
# Feature builder - 벡터화 버전
# -------------------------------------------------------
def build_features_for_symbol(df: pd.DataFrame):
    """
    하나의 (region, symbol, interval) 에 대한 df에서
    모든 시퀀스에 대한 feature 배열을 한 번에 생성한다.

    반환:
      feats: (N, SEQ_LEN * 4) numpy array
      entry_indices: 길이 N, 각 시퀀스의 진입 인덱스 (df 기준)
      entry_prices: 길이 N, 각 진입 가격
    """
    close = df["close"].to_numpy(dtype=float)
    high = df["high"].to_numpy(dtype=float)
    low = df["low"].to_numpy(dtype=float)
    vol = df["volume"].to_numpy(dtype=float)

    T = len(df)
    # 시퀀스 + 미래 구간까지 고려했을 때 최소 길이
    if T < SEQ_LEN + FUTURE_WINDOW + 1:
        return None, None, None

    # 유효한 시퀀스 개수
    # i: 진입 인덱스 (SEQ_LEN ~ T - FUTURE_WINDOW - 1)
    # N = T - SEQ_LEN - FUTURE_WINDOW
    N = T - SEQ_LEN - FUTURE_WINDOW

    # sliding_window_view:
    # shape: (T - SEQ_LEN + 1, SEQ_LEN) → 맨 뒤 FUTURE_WINDOW 구간 제외
    close_win = sliding_window_view(close, SEQ_LEN)[:N]
    high_win = sliding_window_view(high, SEQ_LEN)[:N]
    low_win = sliding_window_view(low, SEQ_LEN)[:N]
    vol_win = sliding_window_view(vol, SEQ_LEN)[:N]

    # 기준 가격은 각 시퀀스의 첫 종가
    base = close_win[:, [0]]  # shape: (N, 1)
    # 0 또는 음수인 경우 필터링
    valid_mask = base[:, 0] > 0
    if not np.any(valid_mask):
        return None, None, None

    close_rel = close_win / base - 1.0
    high_rel = high_win / base - 1.0
    low_rel = low_win / base - 1.0

    vol_mean = vol_win.mean(axis=1, keepdims=True)
    vol_mean[vol_mean <= 0] = 1.0
    vol_norm = vol_win / vol_mean

    feats = np.concatenate(
        [close_rel, high_rel, low_rel, vol_norm],
        axis=1,
    )  # shape: (N, SEQ_LEN*4)

    # 진입 인덱스: 기존 for 루프에서 i (SEQ_LEN ~ T - FUTURE_WINDOW - 1) 에 해당
    entry_indices = np.arange(SEQ_LEN, SEQ_LEN + N)
    entry_prices = close[entry_indices]

    # base <= 0 이었던 시퀀스는 제외
    feats = feats[valid_mask]
    entry_indices = entry_indices[valid_mask]
    entry_prices = entry_prices[valid_mask]

    return feats, entry_indices, entry_prices


# -------------------------------------------------------
# 라벨 계산 - numpy 버전 (TP/SL 순서 고려)
# -------------------------------------------------------
def calc_label_from_array(future_prices: np.ndarray, entry_price: float) -> int:
    """
    future_prices: FUTURE_WINDOW 길이의 close 배열
    entry_price: 진입 가격

    TP/SL 중 누가 먼저 맞았는지 순서를 고려해서 라벨 계산.
    """
    pnl = (future_prices - entry_price) / entry_price

    hit_tp = np.where(pnl >= TP_RATE)[0]
    hit_sl = np.where(pnl <= SL_RATE)[0]

    if len(hit_tp) == 0 and len(hit_sl) == 0:
        return 0

    first_tp = hit_tp[0] if len(hit_tp) > 0 else np.inf
    first_sl = hit_sl[0] if len(hit_sl) > 0 else np.inf

    return 1 if first_tp < first_sl else 0


def run_backtest_for_universe(
    universe,
    *,
    model_setting_key: str,
    note_prefix: str = "",
    backtest_days: int | None = 60,
):
    """
    하나의 유니버스(KR / US / COIN)에 대해 백테스트 전체를 수행하고
    backtests 테이블에 결과를 저장하는 공통 함수.

    - universe: [{"region": "...", "symbol": "...", "excd": "..."}, ...]
    - model_setting_key: settings 테이블에 저장한 모델 경로 키 이름
      예) "active_model_path_kr", "active_model_path_us", "active_model_path_coin"
    - note_prefix: backtests.note 에 붙일 접두어 (예: "[KR]" "[US]" "[COIN]")
    - backtest_days: 최근 N일만 사용할지 (None이면 전체)
    """
    db = BotDatabase(DB_PATH)
    db.log(f"백테스트 시작: {note_prefix} (setting_key={model_setting_key})")

    # 1) settings에서 현재 모델 경로 가져오기
    model_path = db.get_setting(model_setting_key, None)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_setting_key} 없음: {model_path}")

    model = joblib.load(model_path)
    print(f"모델 로드 완료: {model_path}")

    # 2) OHLCV 로드 (기간 필터 적용)
    start_date = None
    if backtest_days is not None:
        start_dt = datetime.now() - timedelta(days=backtest_days)
        start_date = start_dt.strftime("%Y-%m-%d 00:00:00")

    ohlcv_dict = load_all_ohlcv(start_date=start_date)
    if not ohlcv_dict:
        raise RuntimeError("ohlcv_data가 비어 있음. 먼저 OHLCV 백필 필요.")

    # 3) 유니버스에 대해서만 백테스트 실행
    rows = []

    for t in universe:
        region = t["region"]
        symbol = t["symbol"]
        interval = "5m"

        key = (region, symbol, interval)
        if key not in ohlcv_dict:
            print(f"[SKIP] OHLCV 없음: {region} {symbol} {interval}")
            continue

        df = ohlcv_dict[key]
        if len(df) < SEQ_LEN + FUTURE_WINDOW + 1:
            print(f"[SKIP] 샘플 부족: {region} {symbol} ({len(df)} rows)")
            continue

        feats, entry_indices, entry_prices = build_features_for_symbol(df)
        if feats is None or len(feats) == 0:
            print(f"[SKIP] feature 없음: {region} {symbol}")
            continue

        preds = model.predict(feats)
        preds = np.asarray(preds)

        close_arr = df["close"].to_numpy(dtype=float)
        index_arr = df.index.to_numpy()

        trade_count = 0

        for idx_in_batch, entry_idx in enumerate(entry_indices):
            if preds[idx_in_batch] != 1:
                continue

            entry_price = float(entry_prices[idx_in_batch])
            entry_time = index_arr[entry_idx]

            start = entry_idx + 1
            end = entry_idx + 1 + FUTURE_WINDOW
            future_prices = close_arr[start:end]

            if len(future_prices) < FUTURE_WINDOW:
                continue

            label = calc_label_from_array(future_prices, entry_price)
            pnl = (future_prices[-1] - entry_price) / entry_price

            rows.append(
                {
                    "region": region,
                    "symbol": symbol,
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "label": int(label),
                    "pnl": float(pnl),
                }
            )
            trade_count += 1

        print(
            f"[{region} {symbol}] 샘플: {len(feats)}, 진입 수: {trade_count}"
        )

    df_bt = pd.DataFrame(rows)
    print(f"백테스트 진입 수: {len(df_bt)}")

    if df_bt.empty:
        db.log(f"{note_prefix} 백테스트 결과 없음: 샘플 부족 또는 모델 불만족")
        return

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

        cur.execute(
            """
            INSERT INTO backtests
            (model_id, start_date, end_date,
             trades, win_rate, avg_profit,
             cum_return, max_dd, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                None,
                df_bt["entry_time"].min().strftime("%Y-%m-%d"),
                df_bt["entry_time"].max().strftime("%Y-%m-%d"),
                int(len(df_bt)),
                float(win_rate),
                float(avg_profit),
                float(cum_return),
                float(max_dd),
                f"{note_prefix}Backtest for {model_path}",
            ),
        )

        conn.commit()
        conn.close()

        db.log(f"{note_prefix} 백테스트 완료 및 DB 저장 완료")
    except Exception as e:
        db.log(f"{note_prefix} 백테스트 저장 실패: {e}")