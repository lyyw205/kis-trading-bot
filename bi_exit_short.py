from __future__ import annotations
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

from bi_exit_lib import (
    CrPosition,
    DEFAULT_EXIT_PARAMS_BASE,
    calc_pnl_and_bars,
    update_trailing_and_check_exit,
)

# MS 숏 전용 기본값
#  - SL: -2% (롱과 동일 × 방향만 반대라 PnL 계산으로 자동 처리)
#  - Trailing: 최고 수익에서 3% 되돌림이면 컷
#  - Timeout: 6시간까지
DEFAULT_EXIT_PARAMS_MS_SHORT: Dict[str, Any] = {
    "tp_rate": 0.03,                 # 기준 TP 레벨 (참고용)
    "sl_rate": -0.02,                # -2% 손절

    "max_hold_bars": 24,             # 6시간 (5m * 72)
    "min_hold_bars_for_timeout": 6,
    "timeout_deadband": 0.015,

    # Trailing (균형형 기본)
    "use_trailing": True,
    "tp_start_rate": 0.03,           # +3% 이상 수익부터 Trailing Zone
    "trail_gap": 0.015,               # 최고 수익 대비 3% 되돌림에서 컷
    "min_bars_for_trailing": 0,

    # 기본적으로 ML EXIT 는 사용 안 함 (필요하면 나중에 켜기)
    "use_ml_exit": False,
}


def decide_exit_ms_short(
    pos: CrPosition,
    df_5m: pd.DataFrame,
    cur_price: float,
    now_dt: datetime,
    params: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:

    # 0) 파라미터 병합
    if params is None:
        params = DEFAULT_EXIT_PARAMS_BASE.copy()
        params.update(DEFAULT_EXIT_PARAMS_MS_SHORT)
    else:
        merged = DEFAULT_EXIT_PARAMS_BASE.copy()
        merged.update(DEFAULT_EXIT_PARAMS_MS_SHORT)
        merged.update(params)
        params = merged

    # side 정규화 (기본은 숏)
    side = str(getattr(pos, "side", "SHORT")).upper()
    if side == "LONG":
        side = "BUY"
    elif side == "SHORT":
        side = "SELL"

    if side not in ("BUY", "SELL"):
        return {
            "should_exit": False,
            "reason": "UNSUPPORTED_SIDE",
            "exit_price": cur_price,
            "note": f"UNSUPPORTED_SIDE(side={side})",
            "debug": {"side": side},
        }

    # 🚨 중요한 포인트:
    # calc_pnl_and_bars 는 side 가 BUY/LONG이면 가격 상승이 +pnl,
    # side 가 SELL/SHORT 이면 가격 하락이 +pnl 이 되도록 direction 을 내부에서 처리.
    pos.side = side
    pnl_pct, holding_bars = calc_pnl_and_bars(
        pos=pos,
        df_5m=df_5m,
        cur_price=cur_price,
        now_dt=now_dt,
    )

    sl_rate = float(params.get("sl_rate", -0.02))
    max_hold_bars = int(params.get("max_hold_bars", 72))
    min_hold_bars_for_timeout = int(params.get("min_hold_bars_for_timeout", 0))
    timeout_deadband = float(params.get("timeout_deadband", 0.0))

    debug_base: Dict[str, Any] = {
        "side": side,
        "pnl_pct": float(pnl_pct),
        "holding_bars": int(holding_bars),
        "sl_rate": sl_rate,
        "max_hold_bars": max_hold_bars,
        "min_hold_bars_for_timeout": min_hold_bars_for_timeout,
        "timeout_deadband": timeout_deadband,
    }

    # 1) SL (숏 포지션이지만 PnL 기준으로 동일 처리)
    if pnl_pct <= sl_rate:
        return {
            "should_exit": True,
            "reason": "SL_SHORT",
            "exit_price": cur_price,
            "note": f"STOP LOSS {pnl_pct:.4f} <= {sl_rate:.4f}",
            "debug": {**debug_base, "exit_trigger": "SL_SHORT"},
        }

    # 2) TIMEOUT
    if holding_bars >= max_hold_bars:
        if (
            abs(pnl_pct) <= timeout_deadband
            and holding_bars >= min_hold_bars_for_timeout
        ):
            return {
                "should_exit": True,
                "reason": "TIMEOUT_SHORT",
                "exit_price": cur_price,
                "note": f"TIMEOUT_SHORT bars={holding_bars}",
                "debug": {**debug_base, "exit_trigger": "TIMEOUT_SHORT"},
            }

    # 3) Trailing Stop (수익 보호)
    t_exit, t_reason, t_note = update_trailing_and_check_exit(
        pos=pos,
        pnl_pct=pnl_pct,
        params=params,
    )
    if t_exit:
        return {
            "should_exit": True,
            "reason": t_reason,
            "exit_price": cur_price,
            "note": t_note,
            "debug": {**debug_base, "exit_trigger": t_reason},
        }

    # 4) ML EXIT 는 기본 OFF (use_ml_exit=False)
    #    필요하면 나중에 MS와 유사하게 check_ml_based_exit 를 붙이면 됨.

    # 5) HOLD
    return {
        "should_exit": False,
        "reason": "HOLD_SHORT",
        "exit_price": cur_price,
        "note": f"pnl={pnl_pct:.4f}",
        "debug": {**debug_base, "exit_trigger": "HOLD_SHORT"},
    }
