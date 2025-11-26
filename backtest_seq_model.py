# backtest_seq_model.py
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from db import BotDatabase
from config import TARGET_STOCKS

DB_PATH = "trading.db"
MODEL_DEFAULT_PATH = "models/seq_model_latest.pkl"

# === 전략 / 피처 설정 (train_seq_model.py / build_ml_seq_samples.py 와 맞추기) ===
SEQ_LEN = 30          # 과거 캔들 개수
FUTURE_BARS_KR = 12   # KR: 5m * 12 = 약 1시간
FUTURE_BARS_US = 5    # US: 1d * 5 = 약 5일
TP_PCT = 0.03         # +3% 익절
SL_PCT = -0.04        # -4% 손절


# ---------------------------------------------------------
# OHLCV 로딩 유틸
# ---------------------------------------------------------
def load_ohlcv(region, symbol, interval):
    """
    ohlcv_data 에서 특정 (region, symbol, interval)의 시계열을 불러온다.
    index=dt, columns=['open','high','low','close','volume']
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT dt, open, high, low, close, volume
        FROM ohlcv_data
        WHERE region = ? AND symbol = ? AND interval = ?
        ORDER BY dt
        """,
        conn,
        params=(region, symbol, interval),
    )
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt")
    df.set_index("dt", inplace=True)
    df = df.astype(float)
    return df


# ---------------------------------------------------------
# 시퀀스 기반 피처 (train_seq_model.py 와 동일 로직)
# ---------------------------------------------------------
def build_feature_from_seq(df_seq: pd.DataFrame):
    """
    df_seq: 길이 = SEQ_LEN, columns=['open','high','low','close','volume']

    - close / high / low : 첫 종가 대비 수익률
    - volume             : 평균 대비 비율

    => 4 * SEQ_LEN 차원 벡터
    """
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

    vol_mean = np.mean(vol) if np.mean(vol) > 0 else 1.0
    vol_norm = vol / vol_mean

    feat = np.concatenate([close_rel, high_rel, low_rel, vol_norm])
    return feat


# ---------------------------------------------------------
# 심볼 하나에 대한 백테스트
# ---------------------------------------------------------
def backtest_symbol(df: pd.DataFrame,
                    model,
                    ml_threshold: float,
                    lookback: int,
                    band_pct: float):
    """
    단일 심볼 시계열(df)에 대해:
      - 룰 기반 + (옵션) ML 필터로 진입 판단
      - TP/SL 로 청산
      - 각 트레이드의 수익률 계산

    return: trades (list[dict])
    """
    if df is None or df.empty:
        return []

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    index = df.index

    # 미래 구간 길이 결정 (KR/US 는 호출하는 쪽에서 구분)
    # 여기서는 그냥 len(df) 로부터 추정하지 않고,
    # 호출부에서 적절한 future_bars 를 인자로 계산해서 넘겨줘도 된다.
    # 이번 버전에서는 간단히:  이미 잘린 df 를 그대로 사용.
    # (실제 future_bars 로직은 호출부에서 loop 범위로 구현)

    # 룰 기반 지표
    df = df.copy()
    df["support"] = df["low"].rolling(lookback).min()
    df["at_support"] = df["low"] <= df["support"] * (1 + band_pct)
    df["is_bullish"] = df["close"] > df["open"]
    df["price_up"] = df["close"] > df["close"].shift(1)

    trades = []
    in_position = False
    entry_idx = None
    entry_price = None
    entry_ml = None

    n = len(df)

    i = SEQ_LEN - 1  # 최소한 SEQ_LEN 만큼 쌓인 후부터 판단
    while i < n - 1:
        # 포지션이 없으면 → 진입 후보 체크
        if not in_position:
            # 룰 신호
            row = df.iloc[i]
            entry_signal = bool(
                row["at_support"] and row["is_bullish"] and row["price_up"]
            )

            # ML 점수 계산
            ml_proba = None
            if model is not None:
                df_seq = df.iloc[i - SEQ_LEN + 1 : i + 1]
                feat = build_feature_from_seq(df_seq)
                if feat is not None:
                    try:
                        ml_proba = float(model.predict_proba([feat])[0][1])
                    except Exception:
                        ml_proba = None

            # 최종 진입 여부
            if model is None:
                entry_allowed = entry_signal
            else:
                entry_allowed = entry_signal and (ml_proba is not None) and (ml_proba >= ml_threshold)

            if entry_allowed:
                in_position = True
                entry_idx = i
                entry_price = closes[i]
                entry_ml = ml_proba
                i += 1
                continue
            else:
                i += 1
                continue

        # 이미 포지션이 있을 때 → TP / SL / 만기 청산
        else:
            # 미래바 길이 (KR / US 는 심볼별 설정으로 넘겨줌)
            # 여기서는 entry_idx 기준으로 몇 개 바 안에서 TP/SL 확인
            # → 호출 쪽에서 future_bars 를 넘겨줘야 하므로 여기서는
            #   전역 상수를 그냥 사용하지 않고, 간단히 아래와 같이 구현:
            #   일단 전체 시퀀스 끝까지 보되, 적당한 max_holding 으로 제한
            max_holding = 30  # 기본값 (필요하면 밖에서 조정 가능)

            # 청산 시점 계산
            exit_idx = None
            exit_type = "TIMEOUT"

            for j in range(entry_idx + 1, min(n, entry_idx + 1 + max_holding)):
                high = highs[j]
                low = lows[j]

                up_ret = (high / entry_price) - 1.0
                dn_ret = (low / entry_price) - 1.0

                if up_ret >= TP_PCT:
                    exit_idx = j
                    exit_type = "TP"
                    break
                if dn_ret <= SL_PCT:
                    exit_idx = j
                    exit_type = "SL"
                    break

            if exit_idx is None:
                # 끝까지 도달하면 마지막 종가로 청산
                exit_idx = min(n - 1, entry_idx + 1 + max_holding)
                exit_type = "TIMEOUT"

            exit_price = closes[exit_idx]
            ret = (exit_price / entry_price - 1.0) * 100.0

            trades.append({
                "entry_time": index[entry_idx],
                "exit_time": index[exit_idx],
                "entry_price": float(entry_price),
                "exit_price": float(exit_price),
                "ret_pct": float(ret),
                "ml_proba": entry_ml,
                "exit_type": exit_type,
            })

            # 포지션 정리
            in_position = False
            entry_idx = None
            entry_price = None
            entry_ml = None

            # 청산 다음 캔들부터 다시 탐색
            i = exit_idx + 1

    return trades


# ---------------------------------------------------------
# 전체 심볼 백테스트 + 성과 요약
# ---------------------------------------------------------
def run_backtest(db: BotDatabase,
                 model=None,
                 ml_threshold: float = 0.55,
                 label: str = "rule_only",
                 lookback: int = 120,
                 band_pct: float = 0.01):
    """
    - TARGET_STOCKS 전체에 대해 백테스트 실행
    - trades 리스트 + 요약 통계 + backtests 테이블 저장
    """
    all_trades = []
    global_start = None
    global_end = None

    for t in TARGET_STOCKS:
        region = t["region"]
        symbol = t["symbol"]
        interval = "5m" if region == "KR" else "1d"

        df = load_ohlcv(region, symbol, interval)
        if df.empty or len(df) < SEQ_LEN + 5:
            db.log(f"⚠️ 백테스트용 데이터 부족: {region} {symbol} ({interval})")
            continue

        # KR / US 별 미래바 길이 설정 (현재는 max_holding 으로 처리하지만
        # 필요하면 backtest_symbol 안으로 전달 가능)
        trades = backtest_symbol(df, model, ml_threshold, lookback, band_pct)
        if not trades:
            continue

        for tr in trades:
            tr["region"] = region
            tr["symbol"] = symbol
        all_trades.extend(trades)

        # 전체 기간 업데이트
        s = df.index[0]
        e = df.index[-1]
        if global_start is None or s < global_start:
            global_start = s
        if global_end is None or e > global_end:
            global_end = e

    if not all_trades:
        db.log(f"❌ 백테스트 결과 트레이드가 없습니다. (label={label})")
        return

    df_tr = pd.DataFrame(all_trades).sort_values("entry_time")
    df_tr.reset_index(drop=True, inplace=True)

    # 성과 요약
    n_trades = len(df_tr)
    wins = df_tr[df_tr["ret_pct"] > 0]
    win_rate = (len(wins) / n_trades) * 100.0
    avg_profit = df_tr["ret_pct"].mean()

    # 누적 수익률 & 최대 낙폭
    equity = (1 + df_tr["ret_pct"] / 100.0).cumprod()
    peak = equity.cummax()
    dd = (equity / peak - 1.0) * 100.0
    cum_return = (equity.iloc[-1] - 1.0) * 100.0
    max_dd = dd.min()

    print("\n=== Backtest Summary:", label, "===")
    print(f"Trades       : {n_trades}")
    print(f"Win rate (%) : {win_rate:.2f}")
    print(f"Avg profit %% : {avg_profit:.2f}")
    print(f"Cum return %% : {cum_return:.2f}")
    print(f"Max DD   %%   : {max_dd:.2f}")

    # backtests 테이블에 저장 (model_id 는 일단 None)
    start_str = global_start.strftime("%Y-%m-%d %H:%M:%S") if global_start else ""
    end_str = global_end.strftime("%Y-%m-%d %H:%M:%S") if global_end else ""

    db.save_backtest(
        model_id=None,
        start_date=start_str,
        end_date=end_str,
        trades=n_trades,
        win_rate=float(win_rate),
        avg_profit=float(avg_profit),
        cum_return=float(cum_return),
        max_dd=float(max_dd),
        note=label,
    )


# ---------------------------------------------------------
# 실행 엔트리포인트
# ---------------------------------------------------------
if __name__ == "__main__":
    db = BotDatabase(DB_PATH)
    db.log("[BACKTEST] 시퀀스 기반 백테스트 시작")

    # 1) 룰 기반만 (ML 없이)
    run_backtest(
        db=db,
        model=None,
        ml_threshold=0.0,
        label="rule_only",
        lookback=120,
        band_pct=0.01,
    )

    # 2) ML 모델이 있으면 로드해서 실행
    if os.path.exists(MODEL_DEFAULT_PATH):
        try:
            model = joblib.load(MODEL_DEFAULT_PATH)
            db.log(f"[BACKTEST] ML 모델 사용: {MODEL_DEFAULT_PATH}")
            run_backtest(
                db=db,
                model=model,
                ml_threshold=0.55,
                label="seq_model_latest",
                lookback=120,
                band_pct=0.01,
            )
        except Exception as e:
            db.log(f"⚠️ ML 모델 로드 실패: {e}")
    else:
        db.log("⚠️ ML 모델 파일 없음 → ML 백테스트 스킵")

    db.log("[BACKTEST] 백테스트 종료")
