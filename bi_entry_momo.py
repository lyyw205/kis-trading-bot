# st_entry_coin_momo_strategy.py
# CR 멀티 엔트리 - 전략3: 모멘텀 돌파 + ML 결합

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from bi_entry_lib import (
    prepare_ohlcv_with_indicators,
    run_bi_swing_ml,
)


DEFAULT_ENTRY_PARAMS_MOMO: Dict[str, Any] = {
    # 최소 데이터 길이
    "min_len": 60,

    # 거래량 스파이크 판단용
    "vol_window": 20,
    "vol_ratio_min": 2.0,   # 최근 거래량 / 과거 평균 >= 1.5배

    # RSI 구간 (모멘텀 영역)
    "rsi_min": 50.0,
    "rsi_max": 70.0,

    # 최종 MOMO 스코어 컷
    "momo_min_score": 0.006,

    # 룰/ML 가중치
    "rule_weight": 0.3,
    "ml_weight": 0.7,

    # ML horizon 가중치 (r3/r6/r12 비중)
    "ml_horizon_weights": [0.4, 0.35, 0.25],

    # 돌파 강도 / MA20 기울기 필터용
    "min_breakout_strength": 0.002,   # 0.2% 이상 돌파만
    "ma20_slope_lookback": 5,         # 몇 봉 전 MA20과 비교할지
    "ma20_slope_min": 0.0,            # MA20이 평평/하락이면 컷
}


def make_entry_signal_coin_momo(
    df_5m: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    if params is None:
        params = DEFAULT_ENTRY_PARAMS_MOMO.copy()
    else:
        merged = DEFAULT_ENTRY_PARAMS_MOMO.copy()
        merged.update(params)
        params = merged

    ml_params = params
    # 1) 인디케이터 포함 DF 준비
    df_ind = prepare_ohlcv_with_indicators(df_5m, {"min_len": params["min_len"]})
    if df_ind is None or len(df_ind) < params["vol_window"] + 1:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MOMO_NONE",
            "note": "NO_DATA",
            "ml_pred": None,
            "risk": {},
            "entry_score": None,
        }

    last = df_ind.iloc[-1]
    prev = df_ind.iloc[-2]

    close = float(last["close"])
    if close <= 0:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MOMO_NONE",
            "note": "INVALID_CLOSE",
            "ml_pred": None,
            "risk": {},
            "entry_score": None,
        }

    # -------------------------------------------------
    # 2) 모멘텀 룰 기반 스코어
    #    (RSI, 추세, 거래량 스파이크, 직전 고가 돌파)
    # -------------------------------------------------
    # (1) RSI 필터: 과매수까지는 아니지만 상승 모멘텀 구간
    rsi = float(last.get("rsi", np.nan))
    if np.isnan(rsi) or not (params["rsi_min"] <= rsi <= params["rsi_max"]):
        return {
            "entry_signal": False,
            "strategy_name": "BI_MOMO_FILTER",
            "note": f"RSI_OUT({rsi:.1f} not in [{params['rsi_min']},{params['rsi_max']}])",
            "ml_pred": None,
            "risk": {},
            "entry_score": None,
        }

    # (2) 추세 필터: MA20 > MA60 (상승 추세)
    ma20 = float(last.get("ma20", np.nan))
    ma60 = float(last.get("ma60", np.nan))
    if not (ma20 > ma60):
        return {
            "entry_signal": False,
            "strategy_name": "BI_MOMO_FILTER",
            "note": "TREND_BAD(MA20<=MA60)",
            "ml_pred": None,
            "risk": {},
            "entry_score": None,
        }
    
    # MA20 기울기(상승 중인지) 확인
    lookback = int(params.get("ma20_slope_lookback", 5))
    if len(df_ind) > lookback:
        ma20_past = float(df_ind["ma20"].iloc[-lookback])
        ma20_slope = ma20 - ma20_past
        if ma20_slope <= params.get("ma20_slope_min", 0.0):
            return {
                "entry_signal": False,
                "strategy_name": "BI_MOMO_FILTER",
                "note": f"MA20_FLAT_OR_DOWN(slope={ma20_slope:.6f})",
                "ml_pred": None,
                "risk": {},
                "entry_score": None,
            }

    # (3) 거래량 스파이크: 최근 거래량이 과거 평균 대비 n배 이상
    vol = df_ind["volume"]
    vol_ma = vol.rolling(params["vol_window"]).mean()

    last_vol = float(vol.iloc[-1])
    last_vol_ma = float(vol_ma.iloc[-1]) if not np.isnan(vol_ma.iloc[-1]) else 0.0
    vol_ratio = last_vol / last_vol_ma if last_vol_ma > 0 else 0.0

    if vol_ratio < params["vol_ratio_min"]:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MOMO_FILTER",
            "note": f"VOL_LOW({vol_ratio:.2f}<{params['vol_ratio_min']:.2f})",
            "ml_pred": None,
            "risk": {"vol_ratio": vol_ratio},
            "entry_score": None,
        }

    # (4) 직전 고가 돌파 여부 (순수 모멘텀 브레이크아웃)
    prev_high = float(prev["high"])
    breakout = close > prev_high
    if not breakout:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MOMO_FILTER",
            "note": "NO_BREAKOUT",
            "ml_pred": None,
            "risk": {"vol_ratio": vol_ratio},
            "entry_score": None,
        }
    
    # 🔴 추가 1: 돌파 강도 최소 기준
    breakout_strength = (close - prev_high) / prev_high  # 얼마나 세게 돌파했는지
    min_bs = float(params.get("min_breakout_strength", 0.002))  # 기본 0.2%
    if breakout_strength < min_bs:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MOMO_FILTER",
            "note": f"BREAKOUT_WEAK({breakout_strength:.4f}<{min_bs:.4f})",
            "ml_pred": None,
            "risk": {"vol_ratio": vol_ratio},
            "entry_score": None,
        }

    # 🔴 추가 2: 윗꼬리 과도한 가짜 돌파 컷
    high = float(last["high"])
    open_ = float(last["open"])
    upper_shadow = high - max(close, open_)
    body = abs(close - open_)
    if body > 0 and upper_shadow > body * 0.5:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MOMO_FILTER",
            "note": f"LONG_WICK(upper={upper_shadow:.4f}, body={body:.4f})",
            "ml_pred": None,
            "risk": {"vol_ratio": vol_ratio},
            "entry_score": None,
        }

    # (5) 룰 기반 스코어 계산
    breakout_strength = (close - prev_high) / prev_high  # 얼마나 세게 돌파했는지
    vol_score = vol_ratio / params["vol_ratio_min"]      # 1.0 이상일수록 좋음

    rule_score = (
        breakout_strength * 0.6 +
        max(0.0, vol_score - 1.0) * 0.4
    )

    # -------------------------------------------------
    # 3) ML 스코어 (단기 수익 기대값)
    # -------------------------------------------------
    ml = run_bi_swing_ml(df_ind, ml_params)
    if ml is None:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MOMO_NO_ML",
            "note": "ML_PREDICT_FAIL",
            "ml_pred": None,
            "risk": {"vol_ratio": vol_ratio},
            "entry_score": rule_score,
        }

    ml_score = ml["score"]

    # -------------------------------------------------
    # 4) 최종 스코어 = 룰 + ML 가중합
    # -------------------------------------------------
    w_rule = params["rule_weight"]
    w_ml = params["ml_weight"]
    final_score = w_rule * rule_score + w_ml * ml_score

    if final_score < params["momo_min_score"]:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MOMO_BLOCK",
            "note": (
                f"MOMO_SCORE_LOW({final_score:.4f}"
                f"<{params['momo_min_score']:.4f})"
            ),
            "ml_pred": ml,
            "risk": {"vol_ratio": vol_ratio},
            "entry_score": final_score,
        }

    # -------------------------------------------------
    # 5) 진입 OK
    # -------------------------------------------------
    return {
        "entry_signal": True,
        "strategy_name": "BI_MOMO_ENTRY",
        "note": (
            "MOMO_OK("
            f"RSI={rsi:.1f}, "
            f"vol_ratio={vol_ratio:.2f}, "
            f"rule={rule_score:.4f}, "
            f"ml={ml_score:.4f}, "
            f"final={final_score:.4f}"
            ")"
        ),
        "ml_pred": ml,
        "risk": {"vol_ratio": vol_ratio},
        "entry_score": final_score,   # ✅ 메인 허브에서 비교할 점수
    }