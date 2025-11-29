# st_entry_coin_ms.py
# 실전 엔트리 로직 (리얼타임 / 백테스트 공통으로 쓰는 핵심 전략 모듈)
"""
CR(코인) 전용 Multi-Scale TCN + Transformer 최적화 엔트리 모듈

- 입력: 5분봉 OHLCV DataFrame (df_5m)
- 내부:
    1) 공통 지표(RSI, MA20/60, 거래량 평균 등) 추가
    2) ATR/변동성 필터
    3) Multi-Scale 모델 예측 (r_3, r_6, r_12) 호출
    4) 예측 기반 ML Score 계산 + 간단 필터
- 출력: dict
    {
      "entry_signal": bool,
      "strategy_name": str,
      "note": str,
      "ml_pred": {...},
      "risk": {...},
    }

기존 st_entry_coin.py와 별개로 사용 가능:
- core_trade_brain_cr / trader_cr 등에서
  from st_entry_coin_ms import make_entry_signal_coin_ms
  형태로 import 해서 사용하면 된다.
"""

from typing import Dict, Any

import numpy as np
import pandas as pd

from st_entry_common import add_common_entry_columns
from tcn_infer import predict_cr_swing


# -------------------------------------------------------------------
# 0) ATR 계산 (코인 변동성 필터 전용, Multi-Scale 모델용 간단 버전)
# -------------------------------------------------------------------
def calculate_atr_basic(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    df: open, high, low, close 컬럼 포함된 DataFrame (시간순 정렬)
    반환: ATR 시리즈
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()
    return atr


# -------------------------------------------------------------------
# 1) 변동성 / 기본 필터
# -------------------------------------------------------------------
def coin_volatility_filter(last: pd.Series, atr: float, params: dict) -> tuple[bool, str]:
    """
    - ATR 비율이 너무 크면 진입 금지
    - 직전 봉 고저 폭이 너무 크면 진입 금지
    - 필요시 RSI/추세 필터도 함께 적용
    """
    close = float(last["close"])
    if close <= 0:
        return False, "INVALID_CLOSE"

    atr_ratio = atr / close if atr > 0 else 999.0
    hl_ratio = (float(last["high"]) - float(last["low"])) / close

    atr_max_ratio = params.get("atr_max_ratio", 0.03)   # 예: 3%
    hl_max_ratio = params.get("hl_max_ratio", 0.05)     # 예: 5%

    if atr_ratio > atr_max_ratio:
        return False, f"FILTER_ATR({atr_ratio:.4f}>{atr_max_ratio:.4f})"

    if hl_ratio > hl_max_ratio:
        return False, f"FILTER_HL({hl_ratio:.4f}>{hl_max_ratio:.4f})"

    # (선택) RSI / 추세 필터
    rsi = float(last.get("rsi", np.nan))
    ma20 = float(last.get("ma20", np.nan))
    ma60 = float(last.get("ma60", np.nan))

    use_rsi_filter = params.get("use_rsi_filter", True)
    use_trend_filter = params.get("use_trend_filter", True)

    if use_rsi_filter and not np.isnan(rsi):
        rsi_min = params.get("rsi_min", 35.0)
        rsi_max = params.get("rsi_max", 75.0)
        if not (rsi_min <= rsi <= rsi_max):
            return False, f"FILTER_RSI({rsi:.1f} not in [{rsi_min},{rsi_max}])"

    if use_trend_filter and not (np.isnan(ma20) or np.isnan(ma60)):
        # 단순 우상향 필터: MA20 > MA60
        if not (ma20 > ma60):
            return False, "FILTER_TREND(MA20<=MA60)"

    return True, ""


# -------------------------------------------------------------------
# 2) ML 예측 기반 스코어 계산
# -------------------------------------------------------------------
def compute_ml_score(r3: float, r6: float, r12: float, params: dict) -> Dict[str, float]:
    """
    Multi-horizon 수익률 예측(r3,r6,r12)을 하나의 스코어로 압축.
    - 가중 평균
    - worst-case / best-case
    - 양수 비율 등
    """
    weights = params.get("ml_horizon_weights", [0.4, 0.35, 0.25])
    if len(weights) != 3:
        weights = [0.4, 0.35, 0.25]

    w3, w6, w12 = weights
    score = w3 * r3 + w6 * r6 + w12 * r12

    preds = np.array([r3, r6, r12], dtype=float)
    worst = float(preds.min())
    best = float(preds.max())
    pos_ratio = float((preds > 0).mean())

    return {
        "score": float(score),
        "worst": worst,
        "best": best,
        "pos_ratio": pos_ratio,
    }


# -------------------------------------------------------------------
# 3) 최종 엔트리 판단 (Multi-Scale 모델 최적화)
# -------------------------------------------------------------------
def make_entry_signal_coin_ms(df_5m: pd.DataFrame, params: dict) -> Dict[str, Any]:
    """
    CR 전용 Multi-Scale TCN + Transformer 기반 엔트리 로직.

    입력:
        df_5m: 5분봉 OHLCV DataFrame (최신 캔들까지 포함, 시간순 정렬)
        params: 설정 딕셔너리

    반환:
        {
          "entry_signal": bool,
          "strategy_name": str,
          "note": str,
          "ml_pred": {...} or None,
          "risk": {...},
        }
    """

    # 최소 길이 체크 (모델/리샘플/지표에 필요한 최소 캔들 수)
    min_len = params.get("min_len", 120)   # 120개(=10시간) 정도로 가볍게
    if df_5m is None or len(df_5m) < min_len:
        return {
            "entry_signal": False,
            "strategy_name": "CR_MS_NONE",
            "note": "NO_DATA",
            "ml_pred": None,
            "risk": {},
        }

    # 시간/정렬 정리
    df = df_5m.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        if "dt" in df.columns:
            df["dt"] = pd.to_datetime(df["dt"])
            df = df.set_index("dt")
        else:
            raise ValueError("df_5m에는 DatetimeIndex 또는 'dt' 컬럼이 필요합니다.")
    df = df.sort_index()

    # 1) 공통 지표 붙이기 (RSI, MA20/60 등)
    df_ind = add_common_entry_columns(df.copy(), params)
    df_ind = df_ind.dropna(subset=["close"])
    last = df_ind.iloc[-1]

    # 2) ATR 및 변동성 필터
    atr_series = calculate_atr_basic(df_ind, period=params.get("atr_period", 14))
    atr = float(atr_series.iloc[-1])
    vol_ok, vol_reason = coin_volatility_filter(last, atr, params)
    if not vol_ok:
        return {
            "entry_signal": False,
            "strategy_name": "CR_MS_FILTER",
            "note": vol_reason,
            "ml_pred": None,
            "risk": {
                "atr": atr,
                "atr_ratio": atr / float(last["close"]) if last["close"] > 0 else None,
            },
        }

    # 3) Multi-Scale 모델 예측 호출
    swing_pred = predict_cr_swing(df)
    if swing_pred is None:
        # 모델 파일 없음 / 입력부족 등
        return {
            "entry_signal": False,
            "strategy_name": "CR_MS_NO_ML",
            "note": "ML_PREDICT_FAIL",
            "ml_pred": None,
            "risk": {
                "atr": atr,
                "atr_ratio": atr / float(last["close"]) if last["close"] > 0 else None,
            },
        }

    r3 = float(swing_pred["r_3"])
    r6 = float(swing_pred["r_6"])
    r12 = float(swing_pred["r_12"])

    ml_stats = compute_ml_score(r3, r6, r12, params)

    score = ml_stats["score"]
    worst = ml_stats["worst"]
    pos_ratio = ml_stats["pos_ratio"]

    # 4) ML 기반 진입 필터 (하드컷)
    min_r3 = params.get("ml_min_r3", 0.0010)       # +0.10%
    min_r6 = params.get("ml_min_r6", 0.0015)       # +0.15%
    min_r12 = params.get("ml_min_r12", 0.0020)     # +0.20%
    min_score = params.get("ml_min_score", 0.0015) # 가중 평균 최소 +0.15%
    max_worst = params.get("ml_max_worst", -0.01)  # 최악 horizon이 -1% 미만이면 컷
    min_pos_ratio = params.get("ml_min_pos_ratio", 0.67)  # 3개 중 2개 이상 양수

    # horizon별 기본 조건
    if r3 < min_r3 or r6 < min_r6 or r12 < min_r12:
        note = (
            f"ML_BLOCK_HORIZON(r3={r3:.4f},r6={r6:.4f},r12={r12:.4f};"
            f"min={min_r3:.4f}/{min_r6:.4f}/{min_r12:.4f})"
        )
        return {
            "entry_signal": False,
            "strategy_name": "CR_MS_BLOCK",
            "note": note,
            "ml_pred": {
                "r_3": r3,
                "r_6": r6,
                "r_12": r12,
                "score": score,
                "worst": worst,
                "pos_ratio": pos_ratio,
            },
            "risk": {
                "atr": atr,
                "atr_ratio": atr / float(last["close"]) if last["close"] > 0 else None,
            },
        }

    # 가중 평균 스코어 / worst / pos_ratio 조건
    if score < min_score:
        return {
            "entry_signal": False,
            "strategy_name": "CR_MS_BLOCK",
            "note": f"ML_BLOCK_SCORE(score={score:.4f} < {min_score:.4f})",
            "ml_pred": {
                "r_3": r3,
                "r_6": r6,
                "r_12": r12,
                "score": score,
                "worst": worst,
                "pos_ratio": pos_ratio,
            },
            "risk": {
                "atr": atr,
                "atr_ratio": atr / float(last["close"]) if last["close"] > 0 else None,
            },
        }

    if worst < max_worst:
        return {
            "entry_signal": False,
            "strategy_name": "CR_MS_BLOCK",
            "note": f"ML_BLOCK_WORST(worst={worst:.4f} < {max_worst:.4f})",
            "ml_pred": {
                "r_3": r3,
                "r_6": r6,
                "r_12": r12,
                "score": score,
                "worst": worst,
                "pos_ratio": pos_ratio,
            },
            "risk": {
                "atr": atr,
                "atr_ratio": atr / float(last["close"]) if last["close"] > 0 else None,
            },
        }

    if pos_ratio < min_pos_ratio:
        return {
            "entry_signal": False,
            "strategy_name": "CR_MS_BLOCK",
            "note": f"ML_BLOCK_POS_RATIO(pos_ratio={pos_ratio:.2f} < {min_pos_ratio:.2f})",
            "ml_pred": {
                "r_3": r3,
                "r_6": r6,
                "r_12": r12,
                "score": score,
                "worst": worst,
                "pos_ratio": pos_ratio,
            },
            "risk": {
                "atr": atr,
                "atr_ratio": atr / float(last["close"]) if last["close"] > 0 else None,
            },
        }

    # 5) 포지션 강도(스케일링) 힌트 (선택)
    #    - score가 높고 atr_ratio가 낮을수록 강하게
    close = float(last["close"])
    atr_ratio = atr / close if close > 0 else 0.0

    strong_thr = params.get("ml_strong_score", 0.003)  # 0.3%
    weak_thr = params.get("ml_weak_score", 0.0015)     # 0.15%

    if score >= strong_thr and atr_ratio <= params.get("atr_for_strong", 0.02):
        strength = "STRONG"
    elif score >= weak_thr:
        strength = "NORMAL"
    else:
        strength = "LIGHT"

    note = (
        f"CR_MS_ML_OK(score={score:.4f},worst={worst:.4f},"
        f"pos={pos_ratio:.2f},str={strength})"
    )

    return {
        "entry_signal": True,
        "strategy_name": f"CR_MS_{strength}",
        "note": note,
        "ml_pred": {
            "r_3": r3,
            "r_6": r6,
            "r_12": r12,
            "score": score,
            "worst": worst,
            "pos_ratio": pos_ratio,
        },
        "risk": {
            "atr": atr,
            "atr_ratio": atr_ratio,
        },
    }
