# bi_entry_lib.py
# BI 코인 엔트리용 공통 Core (TCN + Transformer 앙상블 + 공통 필터)

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from c_ml_features import calculate_rsi

# 기존 TCN 기반 예측
from bi_infer import predict_bi_swing


ENTRY_VERSION = "BI_MS_ENTRY_v3_TRANS_2025-12-07"


# -------------------------------------------------------------
# 1) 공통: DF 준비 + 인덱스 정리 + 인디케이터 추가
# -------------------------------------------------------------
def prepare_ohlcv_with_indicators(
    df_5m: pd.DataFrame,
    params: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """
    - DatetimeIndex 정리
    - add_common_entry_columns 적용
    - min_len 체크는 params["min_len"]에 위임
    """
    if df_5m is None or len(df_5m) < params.get("min_len", 0):
        return None

    df = df_5m.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "dt" in df.columns:
            df["dt"] = pd.to_datetime(df["dt"])
            df = df.set_index("dt")
        else:
            raise ValueError("DatetimeIndex or 'dt' column required")
    df = df.sort_index()

    df_ind = add_common_entry_columns(df.copy(), params)
    return df_ind

def add_common_entry_columns(df: pd.DataFrame, params: Dict[str, Any]) -> pd.DataFrame:
    """
    BI(코인) 전용 공통 인디케이터 세트.

    엔트리/익절 로직에서 실제로 사용하는 것들만 계산:
      - ma20 / ma50 / ma60
      - rsi (14)
      - vol_ma20 (거래량 평균)
    """
    lookback = params.get("lookback", 20)

    df = df.copy()

    # 이동평균
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma60"] = df["close"].rolling(60).mean()

    # RSI
    df["rsi"] = calculate_rsi(df["close"], 14)

    # 거래량 평균 (MOMO, MS 등에서 사용 가능)
    df["vol_ma20"] = df["volume"].rolling(20).mean()

    return df
# -------------------------------------------------------------
# 2) ATR, 변동성 필터
# -------------------------------------------------------------
def calculate_atr_basic(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def coin_volatility_filter(last: pd.Series, atr: float, params: dict):
    """
    - 공통 변동성 + (선택적) RSI + (선택적) 추세 필터 (롱용)
    """
    close = float(last["close"])
    if close <= 0:
        return False, "INVALID_CLOSE"

    atr_ratio = atr / close if atr > 0 else 999.0
    hl_ratio = (float(last["high"]) - float(last["low"])) / close

    if atr_ratio > params["atr_max_ratio"]:
        return False, f"FILTER_ATR({atr_ratio:.4f}>{params['atr_max_ratio']:.4f})"
    if hl_ratio > params["hl_max_ratio"]:
        return False, f"FILTER_HL({hl_ratio:.4f}>{params['hl_max_ratio']:.4f})"

    if params.get("use_rsi_filter", False):
        rsi = float(last.get("rsi", np.nan))
        if not (params["rsi_min"] <= rsi <= params["rsi_max"]):
            return False, f"FILTER_RSI({rsi:.1f} not in [{params['rsi_min']},{params['rsi_max']}])"

    if params.get("use_trend_filter", False):
        ma20 = float(last.get("ma20", np.nan))
        ma60 = float(last.get("ma60", np.nan))
        if not (ma20 > ma60):
            return False, "FILTER_TREND(MA20<=MA60)"
        if params.get("require_price_above_ma20", False) and not (close >= ma20):
            return False, "FILTER_TREND(CLOSE<MA20)"

    return True, ""

# -------------------------------------------------------------
# 3) 변동성 필터 (숏용)
# -------------------------------------------------------------
def coin_volatility_filter_short(last: pd.Series, atr: float, params: dict):
    """
    [숏 전용] 변동성/RSI/추세 필터
    - short_* 파라미터는 각 숏 전략에서 넘겨주는 params에 포함
    """
    close = float(last["close"])
    if close <= 0:
        return False, "INVALID_CLOSE"

    atr_ratio = atr / close if atr > 0 else 999.0
    hl_ratio = (float(last["high"]) - float(last["low"])) / close

    # 1. 과도한 변동성 제한
    if atr_ratio > params["atr_max_ratio"]:
        return False, f"FILTER_ATR({atr_ratio:.4f}>{params['atr_max_ratio']:.4f})"
    if hl_ratio > params["hl_max_ratio"]:
        return False, f"FILTER_HL({hl_ratio:.4f}>{params['hl_max_ratio']:.4f})"

    # 2. RSI 필터 (옵션)
    if params.get("use_rsi_filter", False):
        rsi = float(last.get("rsi", np.nan))
        short_rsi_min = params.get("short_rsi_min", 35.0)
        short_rsi_max = params.get("short_rsi_max", 60.0)
        if not (short_rsi_min <= rsi <= short_rsi_max):
            return False, f"FILTER_RSI_SHORT({rsi:.1f} not in [{short_rsi_min},{short_rsi_max}])"

    # 3. 추세 필터 (DeathCross or Breakdown)
    if params.get("use_trend_filter", False):
        ma20 = float(last.get("ma20", np.nan))
        ma60 = float(last.get("ma60", np.nan))

        is_death_cross = (ma20 < ma60)
        is_price_breakdown = (close < ma60) and (close < ma20)

        if not (is_death_cross or is_price_breakdown):
            return False, "FILTER_TREND_SHORT(Not DeathCross & Not Breakdown)"

        if params.get("require_price_below_ma20", True) and not (close <= ma20):
            return False, "FILTER_TREND_SHORT(CLOSE>MA20)"

    return True, ""


# -------------------------------------------------------------
# 4) 공통: ML 스코어 계산 (r3/r6/r12 → score/worst/pos_ratio)
# -------------------------------------------------------------
def compute_ml_score(r3: float, r6: float, r12: float, params: dict):
    """
    각 전략에서 넘긴 params["ml_horizon_weights"]를 사용해
    종합 score, worst, best, pos_ratio, neg_ratio 계산
    """
    w3, w6, w12 = params.get("ml_horizon_weights", [0.4, 0.35, 0.25])
    preds = np.array([r3, r6, r12], dtype=float)
    weights = np.array([w3, w6, w12], dtype=float)

    if weights.sum() <= 0:
        weights = np.ones_like(weights) / len(weights)
    else:
        weights = weights / weights.sum()

    score = float(np.dot(weights, preds))
    pos_ratio = float((preds > 0).mean())
    neg_ratio = float((preds < 0).mean())

    return {
        "score": score,
        "worst": float(preds.min()),
        "best": float(preds.max()),
        "pos_ratio": pos_ratio,
        "neg_ratio": neg_ratio,
    }



# -------------------------------------------------------------
# 5) 최종: 하이브리드 모델 기반 ML 래퍼 (엔트리/익절에서 호출)
# -------------------------------------------------------------
def run_bi_swing_ml(
    df: pd.DataFrame,
    params: Dict[str, Any],
) -> Optional[Dict[str, float]]:
    """
    - bi_infer.predict_bi_swing(df)를 호출해서 r_3, r_6, r_12, r_24를 받고
    - compute_ml_score로 score/worst/pos_ratio 등 계산 후 리턴

    반환 예:
        {
          "r_3": ...,
          "r_6": ...,
          "r_12": ...,
          "score": ...,
          "worst": ...,
          "best": ...,
          "pos_ratio": ...,
          "neg_ratio": ...,
        }
    """
    swing = predict_bi_swing(df)
    if swing is None:
        return None

    r3 = float(swing["r_3"])
    r6 = float(swing["r_6"])
    r12 = float(swing["r_12"])

    stats = compute_ml_score(r3, r6, r12, params)
    stats.update({
        "r_3": r3,
        "r_6": r6,
        "r_12": r12,
    })
    return stats