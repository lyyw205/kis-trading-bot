# st_entry_coin_rev_strategy.py
# CR 멀티 엔트리 - 전략2: 눌림목 리버설 + ML 결합 (업그레이드 버전)
#
# 개념:
# 1) 상승 추세(Uptrend) 필터
#    - 가격이 MA50 위
#    - MA20 우상향
# 2) 눌림목(Pullback) 필터
#    - 가격이 MA20 근처 밴드 안에 있을 때만 (너무 위/아래면 제외)
# 3) 과매도 + 리버설 패턴
#    - RSI가 낮은 구간 (과매도 근처)
#    - 직전 봉 음봉, 현재 봉 양봉 + 이전 종가 돌파
# 4) ML 스코어 비중 ↑ : rule_weight 0.3, ml_weight 0.7

from typing import Dict, Any, Optional
import numpy as np
import pandas as pd

from bi_entry_lib import (
    prepare_ohlcv_with_indicators,
    run_bi_swing_ml,
)


DEFAULT_ENTRY_PARAMS_REV = {
    # 최소 데이터 길이
    "min_len": 120,

    # 과매도 상한 (이 값보다 RSI가 작을수록 더 점수 ↑)
    "rsi_max": 40.0,

    # 최종 스코어 최소값
    "rev_min_score": 0.004,

    # 룰/ML 가중치
    "rule_weight": 0.3,
    "ml_weight": 0.7,

    # 눌림목용: MA20 근처 허용 밴드 (±5%)
    "pullback_band_pct": 0.05,

    # 추세 필터: MA20 우상향 요구 여부 + 기준 길이
    "require_ma20_uptrend": True,
    "ma_trend_lookback": 5,
}


def make_entry_signal_coin_rev(
    df_5m: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    # 0) 파라미터 머지
    if params is None:
        params = DEFAULT_ENTRY_PARAMS_REV.copy()
    else:
        merged = DEFAULT_ENTRY_PARAMS_REV.copy()
        merged.update(params)
        params = merged

    # -----------------------------
    # 1) 인디케이터 준비
    # -----------------------------
    df_ind = prepare_ohlcv_with_indicators(df_5m, {"min_len": params["min_len"]})
    if df_ind is None or len(df_ind) < 2:
        return {
            "entry_signal": False,
            "strategy_name": "BI_REV_NONE",
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
            "strategy_name": "BI_REV_NONE",
            "note": "INVALID_CLOSE",
            "ml_pred": None,
            "risk": {},
            "entry_score": None,
        }

    # 필요한 인디케이터 가져오기
    rsi = float(last.get("rsi", np.nan))
    ma20 = float(last.get("ma20", np.nan))
    ma50 = float(last.get("ma50", np.nan))

    if np.isnan(ma20) or np.isnan(ma50):
        return {
            "entry_signal": False,
            "strategy_name": "BI_REV_FILTER",
            "note": "NO_MA_DATA",
            "ml_pred": None,
            "risk": {},
            "entry_score": None,
        }

    # -----------------------------
    # 2) 추세(Trend) 필터: 진짜 눌림목이 되려면 Uptrend 전제
    # -----------------------------
    # (1) 가격이 MA50 위에 있을 것 (큰 추세는 상승)
    if close < ma50:
        return {
            "entry_signal": False,
            "strategy_name": "BI_REV_FILTER",
            "note": f"BELOW_MA50(close={close:.4f}, ma50={ma50:.4f})",
            "ml_pred": None,
            "risk": {},
            "entry_score": None,
        }

    # (2) MA20 우상향 (단기 추세 상승)
    if params.get("require_ma20_uptrend", True):
        lookback_n = int(params.get("ma_trend_lookback", 5))
        if lookback_n >= len(df_ind):
            ma20_past = float(df_ind.iloc[0].get("ma20", np.nan))
        else:
            ma20_past = float(df_ind.iloc[-lookback_n].get("ma20", np.nan))

        if np.isnan(ma20_past) or ma20 <= ma20_past:
            return {
                "entry_signal": False,
                "strategy_name": "BI_REV_FILTER",
                "note": f"MA20_NOT_UPTREND(ma20_now={ma20:.4f}, ma20_past={ma20_past:.4f})",
                "ml_pred": None,
                "risk": {},
                "entry_score": None,
            }

    # -----------------------------
    # 3) 눌림목(Pullback) 필터: 가격이 MA20 근처에 있는지
    # -----------------------------
    # diff_ratio > 0  : close가 MA20보다 위
    # diff_ratio < 0  : close가 MA20보다 아래
    diff_ratio = (close - ma20) / ma20
    band = float(params.get("pullback_band_pct", 0.05))

    # 눌림목은 "너무 떨어지지도, 너무 멀리 튀지도 않은" MA20 근처 구간만 허용
    if abs(diff_ratio) > band:
        return {
            "entry_signal": False,
            "strategy_name": "BI_REV_FILTER",
            "note": f"NOT_IN_PULLBACK_BAND(diff={diff_ratio:.3f}, band=±{band:.3f})",
            "ml_pred": None,
            "risk": {},
            "entry_score": None,
        }

    # pullback_score: MA20에 가까울수록 1, 밴드 끝에 있을수록 0
    pullback_score = max(0.0, 1.0 - abs(diff_ratio) / band)

    # -----------------------------
    # 4) RSI 기반 과매도 필터
    # -----------------------------
    if np.isnan(rsi) or rsi > params["rsi_max"]:
        return {
            "entry_signal": False,
            "strategy_name": "BI_REV_FILTER",
            "note": f"RSI_TOO_HIGH({rsi:.1f}>{params['rsi_max']})",
            "ml_pred": None,
            "risk": {},
            "entry_score": None,
        }

    oversold_score = max(0.0, (params["rsi_max"] - rsi) / params["rsi_max"])  # 0~1

    # -----------------------------
    # 5) 캔들 리버설 패턴 (음봉 → 양봉 + 이전 종가 돌파)
    # -----------------------------
    bull_reversal = (
        prev["close"] < prev["open"] and
        last["close"] > last["open"] and
        last["close"] > prev["close"]
    )

    if not bull_reversal:
        return {
            "entry_signal": False,
            "strategy_name": "BI_REV_FILTER",
            "note": "NO_REVERSAL_PATTERN",
            "ml_pred": None,
            "risk": {},
            "entry_score": None,
        }

    body = (last["close"] - last["open"]) / last["open"]
    body_score = max(0.0, body)  # 양봉이 길수록 ↑ (음봉이면 0)

    # -----------------------------
    # 6) 룰 기반 스코어 (눌림목 + 과매도 + 리버설)
    # -----------------------------
    # oversold_score : RSI 기반
    # body_score     : 캔들 강도
    # pullback_score : MA20 근처 눌림목 위치
    rule_score = (
        0.4 * oversold_score +
        0.3 * body_score +
        0.3 * pullback_score
    )

    # -----------------------------
    # 7) ML 스코어 (공통 허브)
    # -----------------------------
    ml = run_bi_swing_ml(df_ind, params)
    if ml is None:
        return {
            "entry_signal": False,
            "strategy_name": "BI_REV_NO_ML",
            "note": "ML_PREDICT_FAIL",
            "ml_pred": None,
            "risk": {},
            "entry_score": rule_score,
        }

    ml_score = ml["score"]

    # -----------------------------
    # 8) 최종 스코어 = 룰 + ML 가중합
    # -----------------------------
    w_rule = float(params["rule_weight"])
    w_ml = float(params["ml_weight"])

    final_score = w_rule * rule_score + w_ml * ml_score

    if final_score < params["rev_min_score"]:
        return {
            "entry_signal": False,
            "strategy_name": "BI_REV_BLOCK",
            "note": (
                f"REV_SCORE_LOW(final={final_score:.4f}"
                f"<{params['rev_min_score']:.4f}, "
                f"rule={rule_score:.4f}, ml={ml_score:.4f})"
            ),
            "ml_pred": ml,
            "risk": {},
            "entry_score": final_score,
        }

    # -----------------------------
    # 9) 진입 OK
    # -----------------------------
    return {
        "entry_signal": True,
        "strategy_name": "BI_REV_ENTRY",
        "note": (
            "REV_OK("
            f"RSI={rsi:.1f}, "
            f"rule={rule_score:.4f}, "
            f"ml={ml_score:.4f}, "
            f"final={final_score:.4f}, "
            f"pullback={pullback_score:.3f}, "
            f"diff={diff_ratio:.3f}"
            ")"
        ),
        "ml_pred": ml,
        "risk": {},
        "entry_score": final_score,   # ✅ 메인 허브에서 비교할 점수
    }
