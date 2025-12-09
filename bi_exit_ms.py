# bi_exit_ms.py
# 전략1: MS 전용 청산 로직

from typing import Dict, Any, Optional
from datetime import datetime

import pandas as pd

from bi_exit_lib import (
    DEFAULT_EXIT_PARAMS_BASE,
    calc_pnl_and_bars,
    check_ml_based_exit,
    CrPosition,
    update_trailing_and_check_exit,
)

DEFAULT_EXIT_PARAMS_MS: Dict[str, Any] = {
    "tp_rate": 0.03,                 # 이 수치는 '기본 TP 레벨'이며 Trailing의 기준에도 활용
    "sl_rate": -0.02,                # -2% 손절 (최우선)

    "max_hold_bars": 36,             # 3시간 (5m * 36)
    "min_hold_bars_for_timeout": 6,  # 최소 6봉은 버틴 후에야 TIMEOUT 허용
    "timeout_deadband": 0.015,        # ±1% 이내면 TIMEOUT 컷 허용

    # Trailing 설정
    "use_trailing": True,            # MS 전략은 Trailing 사용
    "tp_start_rate": 0.03,           # +3% 이상 수익부터 Trailing Zone 돌입
    "trail_gap": 0.015,              # 최고 수익 대비 1.5% 되돌림에서 청산
    "min_bars_for_trailing": 0,      # 원하면 3~6 같은 값으로 늘릴 수 있음

    # ML EXIT 설정 (조기청산 과도 방지)
    "use_ml_exit": True,
    "ml_recheck_min_bars": 6,        # 최소 6봉 이상 보유 후에만 ML 재예측
    "ml_recheck_interval": 3,        # 3봉마다 다시 보는 형태

    # (선택) ML EXIT 허용 PnL 범위 (tcn_exit_lib에서 기본값이 있다면 생략해도 무방)
    # "ml_pnl_min": -0.05,
    # "ml_pnl_max": 0.08,
}

def decide_exit_ms(
    pos: CrPosition,
    df_5m: pd.DataFrame,
    cur_price: float,
    now_dt: datetime,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    CR(코인) 멀티스케일 TCN 전용 청산 판단 함수.
    (MS 전략용)

    공통 순서:
      1) SL (최우선, -2% 손절)
      2) Trailing TP (3% 이상 수익 구간에서 max_pnl 기준 trailing)
      3) TIMEOUT (holding_bars 기반, deadband 내면 강제 컷)
      4) ML 재예측 기반 조기 청산 (min_bars + interval 조건 충족 시)
      5) 아무 조건도 안 맞으면 HOLD
    """
    # 0) 파라미터 병합 (Base → MS 전용 → 사용자 지정)
    if params is None:
        params = DEFAULT_EXIT_PARAMS_BASE.copy()
        params.update(DEFAULT_EXIT_PARAMS_MS)
    else:
        merged = DEFAULT_EXIT_PARAMS_BASE.copy()
        merged.update(DEFAULT_EXIT_PARAMS_MS)
        merged.update(params)
        params = merged

    # 1) side 정규화: LONG/SHORT → BUY/SELL 매핑
    side = str(getattr(pos, "side", "BUY")).upper()
    if side == "LONG":
        side = "BUY"
    elif side == "SHORT":
        side = "SELL"

    if side not in ("BUY", "SELL"):
        return {
            "should_exit": False,
            "exit_price": None,
            "reason": "UNSUPPORTED_SIDE",
            "note": f"UNSUPPORTED_SIDE(side={side})",
            "debug": {"side": side},
        }

    # 2) PnL / 보유봉 계산 (공통 헬퍼 사용)
    pnl_pct, holding_bars = calc_pnl_and_bars(
        pos=pos,
        df_5m=df_5m,
        cur_price=cur_price,
        now_dt=now_dt,
    )

    # 🔹 Decimal 섞임 방지: 전부 float/int로 캐스팅
    tp_rate = float(params.get("tp_rate", 0.03))
    sl_rate = float(params.get("sl_rate", -0.02))
    max_hold_bars = int(params.get("max_hold_bars", 36))
    min_hold_bars_for_timeout = int(params.get("min_hold_bars_for_timeout", 0))
    timeout_deadband = float(params.get("timeout_deadband", 0.0))

    ml_info = None  # 마지막 return에서 쓸 수 있게 초기화

    # 공통 디버그 베이스
    debug_base: Dict[str, Any] = {
        "side": side,
        "pnl_pct": float(pnl_pct),
        "holding_bars": int(holding_bars),
        "tp_rate": tp_rate,
        "sl_rate": sl_rate,
        "max_hold_bars": max_hold_bars,
        "min_hold_bars_for_timeout": min_hold_bars_for_timeout,
        "timeout_deadband": timeout_deadband,
    }

    # --------------------------------------------------
    # 3) SL (손절) — 최우선
    # --------------------------------------------------
    if pnl_pct <= sl_rate:
        return {
            "should_exit": True,
            "exit_price": cur_price,
            "reason": "SL",
            "note": f"SL_HARD(pnl={pnl_pct:.4f} <= {sl_rate:.4f})",
            "debug": {**debug_base, "exit_trigger": "SL_HARD"},
        }

    # --------------------------------------------------
    # 4) TP/Trailing — +3% 이상 수익 구간에서 max_pnl 기준 trailing
    # --------------------------------------------------
    use_trailing = bool(params.get("use_trailing", False))
    if use_trailing:
        min_trail_bars = int(params.get("min_bars_for_trailing", 0))
        if holding_bars >= min_trail_bars:
            t_exit, t_reason, t_note = update_trailing_and_check_exit(
                pos=pos,
                pnl_pct=pnl_pct,
                holding_bars=holding_bars,
                params=params,
            )
            if t_exit:
                return {
                    "should_exit": True,
                    "exit_price": cur_price,
                    "reason": t_reason,
                    "note": t_note,
                    "debug": {**debug_base, "exit_trigger": t_reason},
                }

    # --------------------------------------------------
    # 5) TIMEOUT — 최대 보유 봉 + deadband
    # --------------------------------------------------
    if holding_bars >= max_hold_bars:
        # deadband 안에 있다면 그냥 시간으로 컷
        if (
            abs(pnl_pct) <= timeout_deadband
            and holding_bars >= min_hold_bars_for_timeout
        ):
            return {
                "should_exit": True,
                "exit_price": cur_price,
                "reason": "TIMEOUT",
                "note": (
                    f"TIMEOUT_FORCE(bars={holding_bars} >= {max_hold_bars}, "
                    f"pnl={pnl_pct:.4f})"
                ),
                "debug": {**debug_base, "exit_trigger": "TIMEOUT_FORCE"},
            }
        # deadband 밖(큰 수익/손실 구간)이면 Trailing/ML에 더 맡기고 계속 진행

    # --------------------------------------------------
    # 6) ML 재예측 기반 조기 청산 (선택)
    #     - 너무 빨리 나가지 않도록:
    #       ml_recheck_min_bars, ml_recheck_interval 로 gating
    # --------------------------------------------------
    if not params.get("use_ml_exit", True):
        return {
            "should_exit": False,
            "exit_price": None,
            "reason": "HOLD",
            "note": f"HOLD_NO_ML(pnl={pnl_pct:.4f},bars={holding_bars})",
            "debug": {**debug_base, "exit_trigger": "NO_ML_EXIT"},
        }

    min_bars = int(params.get("ml_recheck_min_bars", 3))
    interval = int(params.get("ml_recheck_interval", 3))

    if holding_bars < min_bars:
        return {
            "should_exit": False,
            "exit_price": None,
            "reason": "HOLD",
            "note": (
                f"HOLD_FEW_BARS(pnl={pnl_pct:.4f},"
                f"bars={holding_bars} < {min_bars})"
            ),
            "debug": {**debug_base, "exit_trigger": "HOLD_FEW_BARS"},
        }

    if holding_bars % interval != 0:
        return {
            "should_exit": False,
            "exit_price": None,
            "reason": "HOLD",
            "note": (
                f"HOLD_NO_ML_CHECK(bars={holding_bars}, interval={interval})"
            ),
            "debug": {**debug_base, "exit_trigger": "HOLD_NO_ML_CHECK"},
        }

    # ML 재예측 (공통 함수 활용)
    ml_exit, ml_reason, ml_note, ml_info = check_ml_based_exit(
        df_5m=df_5m,
        cur_price=cur_price,
        entry_price=pos.entry_price,
        entry_ml_score=getattr(pos, "ml_score_entry", None),
        entry_ml_worst=getattr(pos, "ml_worst_entry", None),
        now_dt=now_dt,
        params=params,
    )

    if ml_exit:
        return {
            "should_exit": True,
            "exit_price": cur_price,
            "reason": ml_reason,
            "note": ml_note,
            "ml_info": ml_info,
            "debug": {**debug_base, "exit_trigger": ml_reason},
        }

    # --------------------------------------------------
    # 7) 특별한 조건 없으면 홀드
    # --------------------------------------------------
    return {
        "should_exit": False,
        "exit_price": None,
        "reason": "HOLD",
        "note": f"HOLD(pnl={pnl_pct:.4f},bars={holding_bars})",
        "ml_info": ml_info,
        "debug": {**debug_base, "exit_trigger": "HOLD_DEFAULT"},
    }