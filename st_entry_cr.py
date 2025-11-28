# st_entry_coin.py
"""
코인(CR) 전용 강화 엔트리 전략

공통 엔트리(base) + CR 필터 + CR 전용 시그널(CR_ENHANCED) 조합
"""

from typing import Dict, Any
import pandas as pd
import numpy as np

from ml_features import SEQ_LEN
from st_entry_common import make_common_entry_signal, add_common_entry_columns


# -----------------------------
# 🔥 1) ATR 계산 (코인 특, 변동성 필터)
# -----------------------------
def calculate_atr(df: pd.DataFrame, period: int = 14):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


# -----------------------------
# 🔥 2) CR 전용 리버설 (wick 기반)
# -----------------------------
def detect_cr_reversal(last, params):
    """
    코인 전용 역추세:
      - 아래 꼬리가 긴 hammer 형태
      - 하락 RSI 구간(<=40) 후 반등
      - 최근 봉 반등(close > open)
    """
    wick_ratio = (last["open"] - last["low"]) / (last["high"] - last["low"] + 1e-9)

    cond_wick = wick_ratio >= 0.45             # 아래꼬리 길이
    cond_rsi = last["rsi"] <= 40               # RSI 침체 후 반등
    cond_bull = last["close"] > last["open"]   # 양봉

    return cond_wick and cond_rsi and cond_bull


# -----------------------------
# 🔥 3) CR Enhanced Momentum
# -----------------------------
def detect_cr_momentum(df_seq: pd.DataFrame, last):
    """
    CR 전용 모멘텀 강화:
      - 20/60 정배열 + RSI 강세(55~80)
      - 20봉 평균 대비 거래량 폭발 (vol > vol_ma20 * 1.2)
      - 최근 10봉 고점 돌파(high > recent_high)
    """

    recent_high = df_seq["high"].iloc[-10:].max()

    cond_ma = (last["close"] > last["ma20"]) and (last["ma20"] > last["ma60"])
    cond_rsi = 55 <= last["rsi"] <= 80
    cond_vol = last["volume"] > last["vol_ma20"] * 1.2
    cond_break = last["high"] >= recent_high

    return cond_ma and cond_rsi and cond_vol and cond_break


# -----------------------------
# 🔥 4) CR 전용 엔트리 필터
# -----------------------------
def coin_entry_filters(last, atr, params):
    """
    변동성 필터:
      - ATR 비율( ATR / close )이 너무 높으면 진입 금지
      - 직전 봉 변동폭이 너무 높아도 금지
    """
    close = last["close"]
    atr_ratio = atr / close if close > 0 else 999

    # 변동성이 지나치게 큰 상황 차단
    if atr_ratio > params.get("atr_max_ratio", 0.025):
        return False

    # 고가-저가 변동폭 기준(폭등/폭락 봉 차단)
    if (last["high"] - last["low"]) / close > params.get("hl_max_ratio", 0.035):
        return False

    return True


# -----------------------------
# 🔥 5) 최종 CR 엔트리 판단
# -----------------------------
def make_entry_signal_coin(df: pd.DataFrame, params: dict) -> Dict[str, Any]:
    """
    CR 전용 강화 엔트리 전략
    """

    if df is None or len(df) < SEQ_LEN:
        return {
            "entry_signal": False,
            "strategy_name": "NONE",
            "note": "NO_DATA",
        }

    # 최근 시퀀스 뽑기
    df_seq = df.iloc[-SEQ_LEN:].copy()

    # 공통 지표/컬럼 붙이기 (RSI, MA20, MA60 등)
    df2 = add_common_entry_columns(df_seq, params)
    last = df2.iloc[-1]

    # ATR 계산
    atr = calculate_atr(df2, 14).iloc[-1]

    # 공통 엔트리 로직 (REVERSAL + MOMENTUM)
    base = make_common_entry_signal(df_seq, params)
    base_signal = base["entry_signal"]

    # -----------------------------
    # 🔥 CR 전용 조건 계산
    # -----------------------------
    cr_reversal = detect_cr_reversal(last, params)
    cr_momentum = detect_cr_momentum(df2, last)

    # -----------------------------
    # 🔥 변동성 필터 적용
    # -----------------------------
    if not coin_entry_filters(last, atr, params):
        return {
            "entry_signal": False,
            "strategy_name": "NONE",
            "note": "FILTER_VOLATILITY",
        }

    # -----------------------------
    # 🔥 최종 엔트리 결정
    # -----------------------------
    if cr_reversal:
        return {
            "entry_signal": True,
            "strategy_name": "CR_REVERSAL",
            "note": "CR_REVERSAL",
        }

    if cr_momentum:
        return {
            "entry_signal": True,
            "strategy_name": "CR_MOMENTUM_ENHANCED",
            "note": "CR_MOMENTUM",
        }

    if base_signal:
        # 기존 base 전략 허용
        return base

    # 아무 것도 아니면 NO ENTRY
    return {
        "entry_signal": False,
        "strategy_name": "NONE",
        "note": "NO_MATCH",
    }
