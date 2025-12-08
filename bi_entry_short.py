# bi_entry_ms_short.py
from typing import Dict, Any, Optional
import pandas as pd

from bi_entry_lib import (
    prepare_ohlcv_with_indicators,
    calculate_atr_basic,
    coin_volatility_filter_short,
    run_bi_swing_ml,
    # compute_ml_score_short,  <-- 제거 (라이브러리에 없음)
)

DEFAULT_ENTRY_PARAMS_MS_SHORT: Dict[str, Any] = {
    "min_len": 120,
    "atr_period": 14,

    # 변동성 필터
    "atr_max_ratio": 0.03,
    "hl_max_ratio": 0.05,

    # SHORT용 RSI / 추세 필터
    "use_rsi_filter": True,
    "use_trend_filter": True,
    "short_rsi_min": 35.0,
    "short_rsi_max": 60.0,
    "require_price_below_ma20": True,

    # SHORT용 ML 하드컷 (예시 값: 필요하면 튜닝)
    "short_ml_max_r3": -0.0030,
    "short_ml_max_r6": -0.0040,
    "short_ml_max_r12": -0.0050,
    "short_ml_max_score": -0.0050,
    # best가 이 값보다 크면(너무 양수면) 숏 금지 → 숏 스퀴즈 방지용
    "short_ml_min_best": 0.0050,
    # 예측 중 음수 비율 (하락 예측 비율)
    "short_ml_min_neg_ratio": 0.67,

    # ML horizon 가중치 (롱/숏 공통으로 써도 무방)
    "ml_horizon_weights": [0.4, 0.35, 0.25],

    # STRONG 판단용 ATR 문턱 (너무 변동성 크면 숏 보류)
    "atr_for_strong": 0.020,
}


def make_entry_signal_coin_ms_short(
    df_5m: pd.DataFrame,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # 1) 파라미터 병합
    if params is None:
        params = DEFAULT_ENTRY_PARAMS_MS_SHORT.copy()
    else:
        merged = DEFAULT_ENTRY_PARAMS_MS_SHORT.copy()
        merged.update(params)
        params = merged

    # 2) 인디케이터 포함 DF 준비
    df_ind = prepare_ohlcv_with_indicators(df_5m, params)
    if df_ind is None:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MS_SHORT_NONE",
            "note": "NO_DATA",
            "ml_pred": None,
            "risk": {},
            "entry_score": None,
            "direction": "SHORT",
            "side": "SHORT",
        }

    last = df_ind.iloc[-1]
    close = float(last["close"])

    # 3) ATR / 변동성 & 추세 필터 (숏 버전)
    # bi_entry_lib.coin_volatility_filter_short 는 
    # 기본적으로 역배열(MA20 < MA60)과 RSI 상단을 체크함
    atr_series = calculate_atr_basic(df_ind, params["atr_period"])
    atr = float(atr_series.iloc[-1])
    ok, reason = coin_volatility_filter_short(last, atr, params)
    
    if not ok:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MS_SHORT_FILTER",
            "note": reason,
            "ml_pred": None,
            "risk": {"atr": atr, "atr_ratio": atr / close if close > 0 else None},
            "entry_score": None,
            "direction": "SHORT",
            "side": "SHORT",
        }

    # 4) ML 예측 실행
    ml = run_bi_swing_ml(df_ind, params)
    if ml is None:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MS_SHORT_NO_ML",
            "note": "ML_PREDICT_FAIL",
            "ml_pred": None,
            "risk": {"atr": atr, "atr_ratio": atr / close if close > 0 else None},
            "entry_score": None,
            "direction": "SHORT",
            "side": "SHORT",
        }

    # 5) 숏 스코어 계산 (직접 계산)
    # ML 모델은 하락을 '음수'로 예측함. 
    # 따라서 숏 입장에서는 '음수일수록' 좋은 점수임.
    # 비교 편의를 위해 -1을 곱해 '하락 강도(short_magnitude)'를 만듦.
    
    raw_score = ml["score"]      # 예: -0.0050 (하락 예측)
    neg_ratio = ml["neg_ratio"]  # 예: 0.8 (모델 80%가 하락 예측)
    
    # 하락 강도 (양수일수록 강한 하락 예측)
    short_magnitude = -1.0 * raw_score 

    r3, r6, r12 = ml["r_3"], ml["r_6"], ml["r_12"]
    best = ml["best"]   # 예측값 중 가장 높은 값 (숏 입장에선 이게 0에 가까워야 안전, 크면 위험)
    
    # 6) 숏용 하드컷 기준 적용
    
    # (A) 개별 Horizon 체크: 모든 기간에서 예측치가 기준치 이하여야 함 (즉, 확실히 떨어져야 함)
    # 예: short_ml_max_r3 가 -0.0020 이라면, r3는 -0.0030 이어야 통과 (더 작아야 함)
    if (r3 > params.get("short_ml_max_r3", -0.0020) or 
        r6 > params.get("short_ml_max_r6", -0.0025) or 
        r12 > params.get("short_ml_max_r12", -0.0030)):
        
        return {
            "entry_signal": False,
            "strategy_name": "BI_MS_SHORT_BLOCK",
            "note": "ML_BLOCK_HORIZON_SHORT",
            "ml_pred": ml,
            "risk": {"atr": atr, "atr_ratio": atr / close if close > 0 else None},
            "entry_score": short_magnitude, # 점수는 반환해줌 (로깅용)
            "direction": "SHORT",
            "side": "SHORT",
        }

    # (B) 종합 스코어 강도 체크
    # 예: short_ml_max_score 가 -0.0040 이라면, raw_score는 -0.0050 이어야 함.
    # 즉, short_magnitude(0.0050) > abs(-0.0040)
    target_score_abs = abs(params.get("short_ml_max_score", -0.0040))
    
    if short_magnitude < target_score_abs:
        return {
            "entry_signal": False,
            "strategy_name": "BI_MS_SHORT_BLOCK",
            "note": f"ML_BLOCK_SCORE_SHORT({short_magnitude:.4f}<{target_score_abs:.4f})",
            "ml_pred": ml,
            "risk": {"atr": atr, "atr_ratio": atr / close if close > 0 else None},
            "entry_score": short_magnitude,
            "direction": "SHORT",
            "side": "SHORT",
        }

    # (C) 튀는 값(Best) 체크 (중요: 숏 스퀴즈 방지)
    # 예측값 중 하나라도 너무 양수(급등)를 가리키면 진입 금지
    if best > params.get("short_ml_min_best", 0.0050): 
        return {
            "entry_signal": False,
            "strategy_name": "BI_MS_SHORT_BLOCK",
            "note": f"ML_BLOCK_BEST_SHORT({best:.4f})",
            "ml_pred": ml,
            "risk": {"atr": atr, "atr_ratio": atr / close if close > 0 else None},
            "entry_score": short_magnitude,
            "direction": "SHORT",
            "side": "SHORT",
        }

    # (D) 만장일치 여부 (neg_ratio)
    if neg_ratio < params.get("short_ml_min_neg_ratio", 0.67):
        return {
            "entry_signal": False,
            "strategy_name": "BI_MS_SHORT_BLOCK",
            "note": f"ML_BLOCK_NEG_RATIO({neg_ratio:.2f})",
            "ml_pred": ml,
            "risk": {"atr": atr, "atr_ratio": atr / close if close > 0 else None},
            "entry_score": short_magnitude,
            "direction": "SHORT",
            "side": "SHORT",
        }

    # 7) 최종 진입 결정 (강도 분류)
    atr_ratio = atr / close if close > 0 else 0.0
    
    # 숏 전략은 기본적으로 하락장(BEAR)에서만 켜지므로 
    # 문턱을 넘으면 대부분 STRONG으로 봐도 무방하나, ATR로 안전장치 한 번 더 검
    
    atr_for_strong = params.get("atr_for_strong", 0.020) # 변동성이 너무 크면 위험

    # Hub에서 사용할 최종 점수는 short_magnitude (양수화된 값)
    
    if atr_ratio <= atr_for_strong:
        strength = "STRONG"
        entry_signal = True
    else:
        strength = "LIGHT" # 변동성이 너무 크면 진입 보류 or 소액
        entry_signal = False # 여기선 보수로 False 처리 (원하면 True로 변경 가능)

    return {
        "entry_signal": entry_signal,
        "strategy_name": f"BI_MS_SHORT_{strength}",
        "note": f"SHORT_OK(mag={short_magnitude:.4f}, neg={neg_ratio:.2f})",
        "ml_pred": ml,
        "risk": {"atr": atr, "atr_ratio": atr_ratio},
        "entry_score": short_magnitude,  # Hub는 점수가 높을수록 좋다고 판단하므로 변환된 값 리턴
        "direction": "SHORT",
        "side": "SHORT", # [중요] Spot 봇이 이 값을 보고 매수 방지함
    }