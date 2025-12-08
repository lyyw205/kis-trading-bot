# st_entry_common.py
"""
공통 엔트리(진입) 전략 로직.
- REVERSAL (역추세)
- MOMENTUM_STRONG (추세추종)

trader.py / build_ml_seq_samples.py 에서 모두 이 로직을 기준으로 사용.
"""

from typing import Dict, Any
import pandas as pd
from c_ml_features import calculate_rsi


def add_common_entry_columns(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    """
    공통 엔트리 전략을 DataFrame 전체에 벡터화해서 적용.

    추가 컬럼:
      - support, ma20, ma60, rsi, vol_ma20
      - close_prev, vol_prev
      - is_bullish, price_up
      - at_support
      - sig_reversal, sig_momentum
      - entry_signal

    params 예:
      {
        "lookback": 20,
        "band_pct": 0.01,
      }
    """
    lookback = params.get("lookback", 20)
    band_pct = params.get("band_pct", 0.005)

    df = df.copy()

    # ------------------------------
    # 1) 보조지표 계산
    # ------------------------------
    df["support"] = df["low"].rolling(lookback).min()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["rsi"] = calculate_rsi(df["close"], 14)
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    # 이전 봉 값
    df["close_prev"] = df["close"].shift(1)
    df["vol_prev"] = df["volume"].shift(1)

    # ------------------------------
    # 2) 기본 상태 플래그
    # ------------------------------
    df["is_bullish"] = df["close"] > df["open"]
    df["price_up"] = df["close"] > df["close_prev"]

    # ------------------------------
    # 3) Reversal (역추세)
    # ------------------------------
    df["at_support"] = df["low"] <= df["support"] * (1 + band_pct)
    df["sig_reversal"] = df["at_support"] & df["is_bullish"] & df["price_up"]

    # ------------------------------
    # 4) Momentum Strong (추세추종)
    # ------------------------------
    cond_align = (df["close"] > df["ma20"]) & (df["ma20"] > df["ma60"])
    # trader.py 기준: RSI 50 ~ 75
    cond_rsi = (df["rsi"] >= 50) & (df["rsi"] <= 75)
    # trader.py 기준: (prev_vol > vol_ma20) or (last_vol > vol_ma20 * 0.4)
    cond_vol = (df["vol_prev"] > df["vol_ma20"]) | (
        df["volume"] > df["vol_ma20"] * 0.4
    )

    df["sig_momentum"] = cond_align & cond_rsi & cond_vol & df["is_bullish"]

    # ------------------------------
    # 5) 최종 엔트리 플래그
    # ------------------------------
    df["entry_signal"] = df["sig_reversal"] | df["sig_momentum"]

    return df


def make_common_entry_signal_from_seq(
    df_seq: pd.DataFrame,
    params: dict,
) -> Dict[str, Any]:
    """
    최근 SEQ_LEN 캔들 시퀀스를 받아서,
    '현재 시점(시퀀스의 마지막 캔들)'이 진입 조건에 해당하는지 판정.

    df_seq:
      - 시간 오름차순 정렬된 최근 N개 캔들 (예: 30개)
      - columns: [open, high, low, close, volume] 필수

    반환:
      {
        "entry_signal": bool,
        "strategy_name": "REVERSAL" / "MOMENTUM_STRONG" / "NONE",
        "at_support": bool,
        "is_bullish": bool,
        "price_up": bool,
      }
    """
    if df_seq is None or len(df_seq) < 2:
        return {
            "entry_signal": False,
            "strategy_name": "NONE",
            "at_support": False,
            "is_bullish": False,
            "price_up": False,
        }

    df2 = add_common_entry_columns(df_seq, params)

    # ✅ 시퀀스 전체를 사용해 보조지표/패턴을 계산하고,
    #    그 중 "마지막 캔들 위치"에서 진입 여부를 판단.
    last = df2.iloc[-1]

    entry_signal = bool(last["entry_signal"])
    at_support = bool(last["at_support"])
    is_bullish = bool(last["is_bullish"])
    price_up = bool(last["price_up"])

    strategy_name = "NONE"
    if bool(last["sig_reversal"]):
        strategy_name = "REVERSAL"
    elif bool(last["sig_momentum"]):
        strategy_name = "MOMENTUM_STRONG"

    return {
        "entry_signal": entry_signal,
        "strategy_name": strategy_name,
        "at_support": at_support,
        "is_bullish": is_bullish,
        "price_up": price_up,
    }


# 기존 이름을 그대로 쓰고 싶으면 alias로 한번 더 감싸줘도 됨
def make_common_entry_signal(df_seq: pd.DataFrame, params: dict) -> Dict[str, Any]:
    """
    backward-compat 용 래퍼.
    기존 코드에서 쓰던 이름을 그대로 유지.
    """
    return make_common_entry_signal_from_seq(df_seq, params)
