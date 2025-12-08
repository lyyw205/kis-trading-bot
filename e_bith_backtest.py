# 멀티 자산(주식 + 코인) 공용 ML 백테스트 엔진

# - trading.db의 ohlcv_data를 불러와 시퀀스(feature) 생성 후, 학습된 모델로 진입 시점을 예측
# - 예측된 진입 시점마다 주식(KR/US)과 코인(CR)에 대해 서로 다른 익절/손절 로직으로 트레이드를 시뮬레이션
# - 결과를 backtests 테이블에 집계(승률, 평균 수익률, 누적 수익률, 최대 손실 등)하여 저장

# 주요 기능:
# 1) load_all_ohlcv()
#    : trading.db의 ohlcv_data 테이블에서 region/symbol/interval별 OHLCV를 조회 및 그룹화
# 2) build_features_for_symbol()
#    : 한 심볼의 5분봉 데이터로부터 (SEQ_LEN 길이) 시퀀스 기반 피처 벡터(feats)를 벡터화 방식으로 생성
# 3) simulate_coin_trade_with_exit_logic()
#    : 코인(CR)용 실전 청산 로직(decide_exit_coin)을 사용해 한 트레이드의 실제 수익률과 청산 시점 시뮬레이션
# 4) calc_label_from_array()
#    : TP/SL(FUTURE_WINDOW 구간)에서 어느 쪽이 먼저 맞는지에 따라 0/1 라벨 계산
# 5) run_backtest_for_universe()
#    : 유니버스(여러 심볼 리스트)에 대해
#      - settings에서 모델 경로 로드 → joblib로 모델 로딩
#      - 피처 생성 후 모델 예측으로 진입 후보 선택
#      - region이 CR이면 코인용 exit 로직, 그 외(KR/US)는 TP/SL 기반 라벨링으로 수익률 계산
#      - backtests 테이블에 전체 백테스트 결과(트레이드 수, 승률, 평균/누적 수익률, 최대 손실 등) 저장


import os
import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import joblib
from numpy.lib.stride_tricks import sliding_window_view

from e_exit import decide_exit_coin
from c_db_manager import BotDatabase

DB_PATH = "trading.db"
SEQ_LEN = 30
FUTURE_WINDOW = 20  # 코인에서는 exit 로직을 쓰지만, 최소 길이 체크에 사용


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
    if T < SEQ_LEN + FUTURE_WINDOW + 1:
        return None, None, None

    N = T - SEQ_LEN - FUTURE_WINDOW

    close_win = sliding_window_view(close, SEQ_LEN)[:N]
    high_win = sliding_window_view(high, SEQ_LEN)[:N]
    low_win = sliding_window_view(low, SEQ_LEN)[:N]
    vol_win = sliding_window_view(vol, SEQ_LEN)[:N]

    base = close_win[:, [0]]
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
    )

    entry_indices = np.arange(SEQ_LEN, SEQ_LEN + N)
    entry_prices = close[entry_indices]

    feats = feats[valid_mask]
    entry_indices = entry_indices[valid_mask]
    entry_prices = entry_prices[valid_mask]

    return feats, entry_indices, entry_prices


# -------------------------------------------------------
# 코인 전용: 실전 청산 로직 기반 시뮬레이션
# -------------------------------------------------------
def simulate_coin_trade_with_exit_logic(
    df: pd.DataFrame,
    entry_idx: int,
    entry_price: float,
) -> tuple[float, datetime]:
    """
    코인 스캘핑 전용 청산 로직(decide_exit_coin)을 사용해서
    '실제 실시간처럼' 한 트레이드를 시뮬레이션한다.

    - df: 해당 심볼의 5분봉 OHLCV (index: datetime, col: open/high/low/close/volume)
    - entry_idx: 진입 시점 인덱스 (df.index 기반 위치)
    - entry_price: 진입 가격

    반환:
      realized_pnl: 최종 실현 수익률 (소수, 0.003 = 0.3%)
      exit_time: 최종 청산 시각
    """
    close_arr = df["close"].to_numpy(dtype=float)
    index_arr = df.index.to_pydatetime()

    T = len(df)
    if entry_idx + 1 >= T:
        return 0.0, index_arr[entry_idx]

    qty = 1.0
    qty_remain = qty
    realized_pnl = 0.0

    entry_time = index_arr[entry_idx]
    state = {
        "tp1": False,
        "tp2": False,
        "entry_time": entry_time,
        "max_profit": 0.0,
    }

    for i in range(entry_idx + 1, T):
        now = index_arr[i]
        price = close_arr[i]

        sell_qty, sell_type, new_state, profit_rate, elapsed_min = decide_exit_coin(
            symbol="",
            region="CR",
            price=price,
            avg_price=entry_price,
            qty=qty_remain,
            state=state,
            now=now,
        )

        state = new_state

        if sell_qty > 0:
            portion = sell_qty / qty
            realized_pnl += profit_rate * portion
            qty_remain -= sell_qty

            if new_state.get("delete", False) or qty_remain <= 0:
                return realized_pnl, now

        if elapsed_min >= 60:
            if qty_remain > 0:
                final_pnl = (price - entry_price) / entry_price
                realized_pnl += final_pnl * (qty_remain / qty)
                qty_remain = 0.0
            return realized_pnl, now

    last_price = close_arr[-1]
    last_time = index_arr[-1]
    if qty_remain > 0:
        final_pnl = (last_price - entry_price) / entry_price
        realized_pnl += final_pnl * (qty_remain / qty)
    return realized_pnl, last_time


# -------------------------------------------------------
# 코인(CR) 전용 백테스트
# -------------------------------------------------------
def run_backtest_for_coin_universe(
    universe,
    *,
    model_setting_key: str,
    note_prefix: str = "[COIN] ",
    backtest_days: int | None = 60,
):
    """
    코인(CR) 유니버스에 대해 백테스트 전체를 수행하고
    backtests 테이블에 결과를 저장하는 함수.

    - universe: [{"region": "CR", "symbol": "...", ...}, ...]
    - model_setting_key: settings 테이블에 저장한 모델 경로 키 이름
      예) "active_model_path_coin"
    - note_prefix: backtests.note 에 붙일 접두어 (예: "[COIN]")
    - backtest_days: 최근 N일만 사용할지 (None이면 전체)
    """
    db = BotDatabase(DB_PATH)
    db.log(f"백테스트 시작: {note_prefix} (setting_key={model_setting_key})")

    model_path = db.get_setting(model_setting_key, None)
    if not model_path or not os.path.exists(model_path):
        raise FileNotFoundError(f"{model_setting_key} 없음: {model_path}")

    model = joblib.load(model_path)
    print(f"모델 로드 완료: {model_path}")

    start_date = None
    if backtest_days is not None:
        start_dt = datetime.now() - timedelta(days=backtest_days)
        start_date = start_dt.strftime("%Y-%m-%d 00:00:00")

    ohlcv_dict = load_all_ohlcv(start_date=start_date)
    if not ohlcv_dict:
        raise RuntimeError("ohlcv_data가 비어 있음. 먼저 OHLCV 백필 필요.")

    rows = []

    for t in universe:
        region = t["region"]
        symbol = t["symbol"]
        interval = "5m"

        # 코인 전용: 다른 region은 스킵
        if region != "CR":
            print(f"[SKIP] 코인 백테스트에서 제외된 region: {region} {symbol}")
            continue

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

        preds = np.asarray(model.predict(feats))

        index_arr = df.index.to_pydatetime()

        trade_count = 0

        for idx_in_batch, entry_idx in enumerate(entry_indices):
            if preds[idx_in_batch] != 1:
                continue

            entry_price = float(entry_prices[idx_in_batch])
            entry_time = index_arr[entry_idx]

            realized_pnl, exit_time = simulate_coin_trade_with_exit_logic(
                df=df,
                entry_idx=entry_idx,
                entry_price=entry_price,
            )

            rows.append(
                {
                    "region": region,
                    "symbol": symbol,
                    "entry_time": entry_time,
                    "entry_price": entry_price,
                    "label": int(realized_pnl > 0),
                    "pnl": float(realized_pnl),
                }
            )
            trade_count += 1

        print(f"[{region} {symbol}] 샘플: {len(feats)}, 진입 수: {trade_count}")

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
