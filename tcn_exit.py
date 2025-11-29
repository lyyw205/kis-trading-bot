"""
st_exit_cr_ms.py

CR(코인) 전용 Multi-Scale TCN 기반 청산 모듈.

- 엔트리: tcn_entry_cr.make_entry_signal_coin_ms 에 맞춰 설계
- 실시간용:
    CoinRealTimeTrader에서 이 모듈의 decide_exit_cr_ms()를 호출해서
    개별 포지션의 청산 여부를 판단하도록 연결하면 된다.

기본 아이디어:
1) 하드 TP/SL (퍼센트 기준)
2) 최대 보유 봉 수 (timeout)
3) TCN 모델 재예측 기반 조기 청산:
   - 수익 중인데 ML score/최악(worst)이 많이 나빠짐 → 수익 보호 차원 조기 익절
   - 손실 중인데 ML이 더 악화 → 손절 가속
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

from tcn_infer import predict_cr_swing 


# ------------------------------------------------------
# 포지션 정보 구조 (편의용 dataclass, dict로 써도 상관 없음)
# ------------------------------------------------------
@dataclass
class CrPosition:
    region: str          # "CR"
    symbol: str          # "KRW-BTC" 등
    side: str            # "BUY"만 쓴다고 가정
    qty: float
    entry_price: float
    entry_time: datetime
    # 선택: 엔트리 시점 ML/리스크 정보
    ml_score_entry: Optional[float] = None
    ml_worst_entry: Optional[float] = None
    atr_ratio_entry: Optional[float] = None


# ------------------------------------------------------
# 기본 파라미터 (실전에서 dict로 override 가능)
# ------------------------------------------------------
DEFAULT_EXIT_PARAMS: Dict[str, Any] = {
    # 퍼센트 기준 TP/SL
    "tp_rate": 0.03,           # +2% 이익 시 기본 TP 후보
    "sl_rate": -0.02,          # -2% 손실 시 기본 SL 후보

    # 최대 보유 봉 수 (5분봉 기준)
    "max_hold_bars": 12,       # 24개 * 5m = 약 2시간

    # TCN 재예측 관련
    "use_ml_exit": True,
    "ml_recheck_min_bars": 2,  # 엔트리 후 최소 3봉 이상 지나야 ML 기반 조기 청산 체크
    "ml_recheck_interval": 2,  # 3봉마다 ML 재예측 체크

    # 수익 보호용 ML 조건
    "ml_profit_score_drop": 0.0015,   # 엔트리 대비 score가 이만큼 이상 떨어지면 조기익절
    "ml_profit_worst_floor": -0.01,   # worst가 -1% 보다 더 나빠지면 조기익절 고려

    # 손실 제한용 ML 조건
    "ml_loss_worst_cut": -0.03,       # worst가 -3% 이하로 악화되면 손절 가속
}


# ------------------------------------------------------
# 도우미: 포지션 기준으로 현재 PnL, 경과 봉 수 계산
# ------------------------------------------------------
def _calc_pnl_and_bars(
    pos: CrPosition,
    df_5m: pd.DataFrame,
    cur_price: float,
) -> tuple[float, int]:
    """
    df_5m: datetime index, 5분봉 (엔트리 시점 이후 데이터 포함)
    cur_price: 현재 가격
    반환: (pnl_pct, holding_bars)
    """
    if pos.entry_price <= 0:
        return 0.0, 0

    pnl_pct = (cur_price - pos.entry_price) / pos.entry_price

    # entry_time 이후 봉 개수를 세서 holding_bars 추정
    if not isinstance(df_5m.index, pd.DatetimeIndex):
        if "dt" in df_5m.columns:
            df_5m = df_5m.copy()
            df_5m["dt"] = pd.to_datetime(df_5m["dt"])
            df_5m = df_5m.set_index("dt")
        else:
            holding_bars = 0
        # 위에서 변형된 df_5m는 바깥에는 영향을 주지 않도록 함수 내에서만 쓴다.

    if isinstance(df_5m.index, pd.DatetimeIndex):
        after_entry = df_5m[df_5m.index >= pos.entry_time]
        holding_bars = max(0, len(after_entry))
    else:
        holding_bars = 0

    return float(pnl_pct), int(holding_bars)


# ------------------------------------------------------
# 메인: CR TCN 전용 Exit 판단
# ------------------------------------------------------
def decide_exit_cr_ms(
    pos: CrPosition,
    df_5m: pd.DataFrame,
    cur_price: float,
    now_dt: datetime,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    CR(코인) 멀티스케일 TCN 전용 청산 판단 함수.

    입력:
      - pos: CrPosition (또는 같은 필드 가진 object/dict)
      - df_5m: 최근까지의 5분봉 DataFrame (datetime index or 'dt' column 포함)
      - cur_price: 현재 가격 (ticker의 last or 현재 5분봉 close 등)
      - now_dt: 현재 시각
      - params: 설정 dict (없으면 DEFAULT_EXIT_PARAMS 사용)

    반환:
      {
        "should_exit": bool,
        "exit_price": float or None,
        "reason": str,   # "TP", "SL", "TIMEOUT", "ML_TAKE_PROFIT", "ML_CUT_LOSS" 등
        "note": str,     # 디버깅용 코멘트
      }
    """
    if params is None:
        params = DEFAULT_EXIT_PARAMS
    else:
        merged = DEFAULT_EXIT_PARAMS.copy()
        merged.update(params)
        params = merged

    if pos.side.upper() != "BUY":
        # 현재는 롱만 지원
        return {"should_exit": False, "exit_price": None, "reason": "UNSUPPORTED_SIDE", "note": ""}

    pnl_pct, holding_bars = _calc_pnl_and_bars(pos, df_5m, cur_price)

    tp_rate = params["tp_rate"]
    sl_rate = params["sl_rate"]
    max_hold_bars = params["max_hold_bars"]

    # ------------------------------
    # 1) 하드 TP / SL
    # ------------------------------
    if pnl_pct >= tp_rate:
        return {
            "should_exit": True,
            "exit_price": cur_price,
            "reason": "TP",
            "note": f"TP_HARD(pnl={pnl_pct:.4f} >= {tp_rate:.4f})",
        }

    if pnl_pct <= sl_rate:
        return {
            "should_exit": True,
            "exit_price": cur_price,
            "reason": "SL",
            "note": f"SL_HARD(pnl={pnl_pct:.4f} <= {sl_rate:.4f})",
        }

    # ------------------------------
    # 2) TIMEOUT (최대 보유 봉 수)
    # ------------------------------
    if holding_bars >= max_hold_bars:
        return {
            "should_exit": True,
            "exit_price": cur_price,
            "reason": "TIMEOUT",
            "note": f"TIMEOUT(bars={holding_bars} >= {max_hold_bars})",
        }

    # ------------------------------
    # 3) TCN 재예측 기반 조기 청산 (선택)
    # ------------------------------
    if not params.get("use_ml_exit", True):
        return {
            "should_exit": False,
            "exit_price": None,
            "reason": "HOLD",
            "note": f"HOLD(pnl={pnl_pct:.4f},bars={holding_bars})",
        }

    # 최소/간격 조건 체크
    min_bars = params["ml_recheck_min_bars"]
    interval = params["ml_recheck_interval"]

    if holding_bars < min_bars:
        return {
            "should_exit": False,
            "exit_price": None,
            "reason": "HOLD",
            "note": f"HOLD_FEW_BARS(pnl={pnl_pct:.4f},bars={holding_bars} < {min_bars})",
        }

    # 예: 엔트리 이후 3, 6, 9, ... 봉마다만 ML 재예측해서 너무 자주 돌지 않도록
    if holding_bars % interval != 0:
        return {
            "should_exit": False,
            "exit_price": None,
            "reason": "HOLD",
            "note": f"HOLD_NO_ML_CHECK(bars={holding_bars}, interval={interval})",
        }

    # ------------------------------
    # 3-1) TCN 재예측 실행
    # ------------------------------
    swing_pred = predict_cr_swing(df_5m)
    if swing_pred is None:
        return {
            "should_exit": False,
            "exit_price": None,
            "reason": "HOLD",
            "note": f"HOLD_ML_FAIL(pnl={pnl_pct:.4f})",
        }

    r3 = float(swing_pred.get("r_3", 0.0))
    r6 = float(swing_pred.get("r_6", 0.0))
    r12 = float(swing_pred.get("r_12", 0.0))

    preds = np.array([r3, r6, r12], dtype=float)
    score_now = float(preds.mean())
    worst_now = float(preds.min())

    # 엔트리 시점의 score/worst (없으면 0 기준으로 본다)
    score_entry = pos.ml_score_entry if pos.ml_score_entry is not None else 0.0
    worst_entry = pos.ml_worst_entry if pos.ml_worst_entry is not None else 0.0

    score_drop = score_entry - score_now

    # 수익 보호용 설정
    profit_score_drop = params["ml_profit_score_drop"]
    profit_worst_floor = params["ml_profit_worst_floor"]

    # 손실 제한용 설정
    loss_worst_cut = params["ml_loss_worst_cut"]

    # ------------------------------
    # 3-2) 수익 상태에서 ML이 꺾이면 조기 익절
    # ------------------------------
    if pnl_pct > 0:
        if score_drop >= profit_score_drop or worst_now <= profit_worst_floor:
            note = (
                f"ML_TAKE_PROFIT(pnl={pnl_pct:.4f},score_now={score_now:.4f},"
                f"score_entry={score_entry:.4f},drop={score_drop:.4f},"
                f"worst_now={worst_now:.4f},worst_entry={worst_entry:.4f})"
            )
            return {
                "should_exit": True,
                "exit_price": cur_price,
                "reason": "ML_TAKE_PROFIT",
                "note": note,
            }

    # ------------------------------
    # 3-3) 손실 상태에서 ML이 더 악화되면 손절 가속
    # ------------------------------
    if pnl_pct < 0:
        if worst_now <= loss_worst_cut:
            note = (
                f"ML_CUT_LOSS(pnl={pnl_pct:.4f},score_now={score_now:.4f},"
                f"worst_now={worst_now:.4f} <= {loss_worst_cut:.4f})"
            )
            return {
                "should_exit": True,
                "exit_price": cur_price,
                "reason": "ML_CUT_LOSS",
                "note": note,
            }

    # ------------------------------
    # 4) 특별한 조건 없으면 홀드
    # ------------------------------
    return {
        "should_exit": False,
        "exit_price": None,
        "reason": "HOLD",
        "note": f"HOLD(pnl={pnl_pct:.4f},bars={holding_bars},score_now={score_now:.4f},worst_now={worst_now:.4f})",
    }
