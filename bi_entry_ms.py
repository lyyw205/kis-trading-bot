# bi_entry_ms.py
#BI 멀티 엔트리 - 전략1: MS + ML 기반 엔트리

from typing import Dict, Any, Optional
import pandas as pd

from bi_entry_lib import (
    prepare_ohlcv_with_indicators,
    calculate_atr_basic,
    coin_volatility_filter,
    run_bi_swing_ml,
)

# MS 전용 파라미터 (롱)
DEFAULT_ENTRY_PARAMS_MS: Dict[str, Any] = {
    "min_len": 120,
    "atr_period": 14,

    # 변동성 필터
    "atr_max_ratio": 0.03,
    "hl_max_ratio": 0.05,

    # RSI / 추세 필터
    "use_rsi_filter": True,
    "use_trend_filter": True,
    "require_price_above_ma20": True,
    "rsi_min": 40.0,
    "rsi_max": 65.0,

    # ML 하드컷 필터
    "ml_min_r3": 0.0020,
    "ml_min_r6": 0.0025,
    "ml_min_r12": 0.0030,
    "ml_min_score": 0.0040,
    "ml_max_worst": -0.005,
    "ml_min_pos_ratio": 0.67,

    # ML horizon 가중치 (MS는 단기/중기 비슷하게)
    "ml_horizon_weights": [0.4, 0.35, 0.25],

    # 전략 강도 기준
    "ml_strong_score": 0.0040,
    "ml_weak_score": 0.0030,
    "atr_for_strong": 0.015,

    # 유니버스 전체 최종 진입 최소 점수 (허브 기본값 용도)
    "ms_min_final_score": 0.010,
}


def make_entry_signal_coin_ms(
    df_5m: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:

    # 파라미터 병합 (MS 전략 전용 튜닝값이 필요하면 여기 override)
    if params is None:
        params = DEFAULT_ENTRY_PARAMS_MS.copy()
    else:
        merged = DEFAULT_ENTRY_PARAMS_MS.copy()
        merged.update(params)
        params = merged

    # 인디케이터 포함 DF 준비
    df_ind = prepare_ohlcv_with_indicators(df_5m, params)
    if df_ind is None:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MS_NONE",
            "note": "NO_DATA",
            "ml_pred": None,
            "risk": {},
            "entry_score": None,
        }

    last = df_ind.iloc[-1]
    close = float(last["close"])

    # ATR / 변동성 필터
    atr_series = calculate_atr_basic(df_ind, params["atr_period"])
    atr = float(atr_series.iloc[-1])
    ok, reason = coin_volatility_filter(last, atr, params)
    if not ok:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MS_FILTER",
            "note": reason,
            "ml_pred": None,
            "risk": {"atr": atr, "atr_ratio": atr / close if close > 0 else None},
            "entry_score": None,
        }

    # ML 예측 (공통 허브)
    ml = run_bi_swing_ml(df_ind, params)
    if ml is None:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MS_NO_ML",
            "note": "ML_PREDICT_FAIL",
            "ml_pred": None,
            "risk": {"atr": atr, "atr_ratio": atr / close if close > 0 else None},
            "entry_score": None,
        }

    r3, r6, r12 = ml["r_3"], ml["r_6"], ml["r_12"]
    score, worst, pos_ratio = ml["score"], ml["worst"], ml["pos_ratio"]

    # 하드컷들 (전략1 전용)
    if r3 < params["ml_min_r3"] or r6 < params["ml_min_r6"] or r12 < params["ml_min_r12"]:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MS_BLOCK",
            "note": "ML_BLOCK_HORIZON",
            "ml_pred": ml,
            "risk": {"atr": atr, "atr_ratio": atr / close if close > 0 else None},
            "entry_score": score,
        }

    if score < params["ml_min_score"]:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MS_BLOCK",
            "note": "ML_BLOCK_SCORE",
            "ml_pred": ml,
            "risk": {"atr": atr, "atr_ratio": atr / close if close > 0 else None},
            "entry_score": score,
        }

    if worst < params["ml_max_worst"]:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MS_BLOCK",
            "note": "ML_BLOCK_WORST",
            "ml_pred": ml,
            "risk": {"atr": atr, "atr_ratio": atr / close if close > 0 else None},
            "entry_score": score,
        }

    if pos_ratio < params["ml_min_pos_ratio"]:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MS_BLOCK",
            "note": "ML_BLOCK_POS_RATIO",
            "ml_pred": ml,
            "risk": {"atr": atr, "atr_ratio": atr / close if close > 0 else None},
            "entry_score": score,
        }

    # 강도 분류
    atr_ratio = atr / close if close > 0 else None

    if score >= params["ml_strong_score"] and (atr_ratio is not None and atr_ratio <= params["atr_for_strong"]):
        strength = "STRONG"
        entry_signal = True          # STRONG만 진입 허용
    elif score >= params["ml_weak_score"]:
        strength = "NORMAL"
        entry_signal = False         # 진입 안 함 (신호만 참고용)
    else:
        strength = "LIGHT"
        entry_signal = False         # 진입 안 함

    return {
        "entry_signal": entry_signal,
        "strategy_name": f"BI_MS_{strength}",
        "note": f"BI_MS_ML_OK(score={score:.4f}, worst={worst:.4f}, pos={pos_ratio:.2f}, str={strength})",
        "ml_pred": ml,
        "risk": {"atr": atr, "atr_ratio": atr_ratio},
        "entry_score": score,  # 허브에서 비교할 점수
        "direction": "LONG",
        "side": "BUY",
    }