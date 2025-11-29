# st_entry_coin.py
"""
코인(CR) 전용 강화 엔트리 전략

공통 엔트리(base) + CR 필터 + CR 전용 시그널(CR_ENHANCED) 조합
"""

from typing import Dict, Any
import pandas as pd
import numpy as np
from swing_infer_cr import predict_cr_swing
from ml_features import SEQ_LEN
from st_entry_common import make_common_entry_signal, add_common_entry_columns
from utils import calculate_atr

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
    # 🔥 1) 우선 기존 CR 전용 룰로 entry 후보 결정
    entry_decision = None

    if cr_reversal:
        entry_decision = {
            "entry_signal": True,
            "strategy_name": "CR_REVERSAL",
            "note": "CR_REVERSAL",
        }
    elif cr_momentum:
        entry_decision = {
            "entry_signal": True,
            "strategy_name": "CR_MOMENTUM_ENHANCED",
            "note": "CR_MOMENTUM",
        }
    elif base_signal:
        entry_decision = base  # 기존 base 전략 허용

    if not entry_decision:
        # 아무 전략도 안 걸리면 그냥 NO_ENTRY
        return {
            "entry_signal": False,
            "strategy_name": "NONE",
            "note": "NO_MATCH",
        }

    # 🔥 2) 여기서 Swing 모델 예측 호출
    swing_pred = predict_cr_swing(df)
    if swing_pred is None:
        # 모델/데이터 문제로 예측 실패하면 일단 기존 결정 그대로 사용
        entry_decision["swing_pred"] = None
        return entry_decision

    r3 = swing_pred["r_3"]
    r6 = swing_pred["r_6"]
    r12 = swing_pred["r_12"]

    entry_decision["swing_pred"] = {
        "r_3": r3,
        "r_6": r6,
        "r_12": r12,
    }

    # 🔥 3) 간단한 필터 룰 예시 (나중에 조정 가능)
    min_r6 = params.get("swing_min_r6", 0.003)     # +0.3% 이상 기대
    min_r12 = params.get("swing_min_r12", 0.005)   # +0.5% 이상 기대

    # 예: r6 또는 r12 둘 다 기준보다 낮으면 진입 차단
    if (r6 < min_r6) and (r12 < min_r12):
        return {
            "entry_signal": False,
            "strategy_name": "NONE",
            "note": f"SWING_FILTER_BLOCK(r6={r6:.4f}, r12={r12:.4f})",
            "swing_pred": entry_decision["swing_pred"],
        }

    # 통과하면 기존 entry 유지 + note에 ML 정보 추가
    entry_decision["note"] = (entry_decision.get("note", "") + 
                              f"|SWING_OK(r6={r6:.4f},r12={r12:.4f})")
    return entry_decision