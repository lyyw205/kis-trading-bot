# bi_exit_rev.py
# 전략2: 리버설 전용 청산 로직

from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

from bi_exit_lib import (
    DEFAULT_EXIT_PARAMS_BASE,
    check_ml_based_exit,
    calc_pnl_and_bars,
    update_trailing_and_check_exit,
    CrPosition,
)

# REV 전용 기본값 (Base를 살짝 override)
# - 기존 의도 유지: "조금만 튕겨도 익절, 오래 안 들고간다"
DEFAULT_EXIT_PARAMS_REV: Dict[str, Any] = {
    "tp_rate": 0.02,                # +1.5% 정도에서 TP 후보
    "sl_rate": -0.02,                # -2% 손절 (통일)
    "max_hold_bars": 24,             # 1시간 정도 (5m * 12)

    "min_hold_bars_for_timeout": 3,  # 최소 3봉은 버틴 후에만 TIMEOUT
    "timeout_deadband": 0.01,        # ±1% 이내면 TIMEOUT 컷 허용

    # 기본값은 Trailing OFF (리버설은 짧게 먹고 나가는 느낌)
    "use_trailing": True,
    # 필요하면 나중에 켤 수 있음
    "tp_start_rate": 0.02,
    "trail_gap": 0.01,
    "min_bars_for_trailing": 0,

    # ML EXIT: MS보다 조금 더 공격적으로 써볼 여지는 있지만,
    # 일단은 MS와 유사한 세팅으로 시작
    "use_ml_exit": True,
    "ml_recheck_min_bars": 5,   # 4봉 이상 보유 후
    "ml_recheck_interval": 2,   # 2봉마다 재확인
}


def decide_exit_rev(
    pos: CrPosition,
    df_5m: pd.DataFrame,
    cur_price: float,
    now_dt: datetime,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    # 0) 파라미터 병합 (Base → REV 전용 → 사용자 지정)
    if params is None:
        params = DEFAULT_EXIT_PARAMS_BASE.copy()
        params.update(DEFAULT_EXIT_PARAMS_REV)
    else:
        merged = DEFAULT_EXIT_PARAMS_BASE.copy()
        merged.update(DEFAULT_EXIT_PARAMS_REV)
        merged.update(params)
        params = merged

    # 1) side 정규화
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

    # 2) PnL / 보유봉 계산 (공통)
    pnl_pct, holding_bars = calc_pnl_and_bars(
        pos=pos,
        df_5m=df_5m,
        cur_price=cur_price,
        now_dt=now_dt,
    )

    tp_rate = float(params.get("tp_rate", 0.015))
    sl_rate = float(params.get("sl_rate", -0.02))
    max_hold_bars = int(params.get("max_hold_bars", 12))
    min_hold_bars_for_timeout = int(params.get("min_hold_bars_for_timeout", 0))
    timeout_deadband = float(params.get("timeout_deadband", 0.0))

    ml_info = None

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

    # ---------------------------
    # 3) SL (손절) 최우선
    # ---------------------------
    if pnl_pct <= sl_rate:
        return {
            "should_exit": True,
            "exit_price": cur_price,
            "reason": "SL",
            "note": f"SL_HARD(pnl={pnl_pct:.4f} <= {sl_rate:.4f})",
            "debug": {**debug_base, "exit_trigger": "SL_HARD"},
        }

    # ---------------------------
    # 4) (옵션) Trailing TP
    #    - 기본은 use_trailing=False라 스킵됨
    # ---------------------------
    if params.get("use_trailing", False):
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

    # ---------------------------
    # 5) TIMEOUT (짧은 홀딩)
    # ---------------------------
    if holding_bars >= max_hold_bars:
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
        # 수익이나 손실이 큰 상태면 ML에 맡기고 계속

    # ---------------------------
    # 6) ML 재예측 (조기청산/가속 손절)
    # ---------------------------
    if not params.get("use_ml_exit", True):
        return {
            "should_exit": False,
            "exit_price": None,
            "reason": "HOLD",
            "note": f"HOLD_NO_ML(pnl={pnl_pct:.4f},bars={holding_bars})",
            "debug": {**debug_base, "exit_trigger": "NO_ML_EXIT"},
        }

    min_bars = int(params.get("ml_recheck_min_bars", 4))
    interval = int(params.get("ml_recheck_interval", 2))

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

    # ---------------------------
    # 7) 기본은 HOLD
    # ---------------------------
    return {
        "should_exit": False,
        "exit_price": None,
        "reason": "HOLD",
        "note": f"HOLD(pnl={pnl_pct:.4f},bars={holding_bars})",
        "ml_info": ml_info,
        "debug": {**debug_base, "exit_trigger": "HOLD_DEFAULT"},
    }
