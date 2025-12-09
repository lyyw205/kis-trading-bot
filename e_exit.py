# st_exit_coin.py
from datetime import datetime
from typing import Dict, Any, Tuple, Optional


def decide_exit_coin(
    symbol: str,
    region: str,
    price: float,
    avg_price: float,
    qty: float,
    state: Dict[str, Any],
    now: Optional[datetime] = None,
) -> Tuple[float, str, Dict[str, Any], float, float]:
    """
    코인 스캘핑 전용 청산 로직.

    반환값:
      sell_qty: 매도 수량 (float)
      sell_type: 사유 문자열
      new_state: 업데이트된 상태
      profit_rate: 현재 수익률
      elapsed_min: 보유 시간 (분)
    """

    # 현재 시각
    if now is None:
        now = datetime.now()

    # 비정상 방어
    if qty <= 0 or avg_price <= 0:
        return 0.0, "", state, 0.0, 0.0

    profit_rate = (price - avg_price) / avg_price  # 소수 (%)
    tp1 = bool(state.get("tp1", False))
    tp2 = bool(state.get("tp2", False))
    entry_time = state.get("entry_time", now)
    max_profit = float(state.get("max_profit", 0.0))

    # 최고 수익률 갱신
    max_profit = max(max_profit, profit_rate)

    elapsed_min = (now - entry_time).total_seconds() / 60.0

    # ============================
    # 1) 손절 (스캘핑 핵심: 빠른 컷)
    # ============================
    # -0.25% 손절
    if profit_rate <= -0.01:
        return qty, "CUT_LOSS_-0.1%", {**state, "delete": True}, profit_rate, elapsed_min

    # ============================
    # 2) 익절 2차 (+0.6% → 전량)
    # ============================
    if profit_rate >= 0.015 and not tp2:
        return qty, "TP2_+1.5%", {
            "tp1": True,
            "tp2": True,
            "entry_time": entry_time,
            "max_profit": max_profit,
            "delete": True
        }, profit_rate, elapsed_min

    # ============================
    # 3) 익절 1차 (+0.3% → 50%)
    # ============================
    if profit_rate >= 0.01 and not tp1:
        sell_qty = qty * 0.5
        return sell_qty, "TP1_+1%", {
            "tp1": True,
            "tp2": False,
            "entry_time": now,            # 1차 익절 이후 시간 리셋
            "max_profit": max_profit
        }, profit_rate, elapsed_min

    # ============================
    # 4) 1차 익절 이후 retrace (고점 대비 40bp 반환)
    #    max_profit >= 0.005(0.5%) 찍고 → 0.002(0.2%) 이하로 밀리면 전량
    # ============================
    if tp1 and not tp2:
        if max_profit >= 0.012 and profit_rate <= 0.008:
            return qty, "RETRACE_AFTER_TP1", {**state, "delete": True}, profit_rate, elapsed_min

    # ============================
    # 5) 시간 기반 종료 (스캘핑)
    #    - 진입 후 12분 유지 → 0.2% 못 넘김 → 종료
    # ============================
    if elapsed_min >= 12 and profit_rate < 0.01:
        return qty, "TIMEOUT_12min", {**state, "delete": True}, profit_rate, elapsed_min

    # 업데이트 후 계속 보유
    return 0.0, "", {
        "tp1": tp1,
        "tp2": tp2,
        "entry_time": entry_time,
        "max_profit": max_profit,
    }, profit_rate, elapsed_min
