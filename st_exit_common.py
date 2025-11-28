# st_exit_common.py
"""
모든 자산군 공통 청산(매도) 전략.
trader.py 안의 매도 블록을 그대로 함수로 분리한 형태.
"""

from datetime import datetime
from typing import Dict, Any, Tuple, Optional


def decide_exit(
    symbol: str,
    region: str,
    price: float,
    avg_price: float,
    qty: int,
    state: Dict[str, Any],
    now: Optional[datetime] = None,
) -> Tuple[int, str, Dict[str, Any], float, float]:
    """
    trader.py의 매도 로직을 함수로 옮긴 것.

    입력:
      - symbol, region: 심볼/자산군
      - price: 현재가
      - avg_price: 평단
      - qty: 보유 수량
      - state: {
          "tp1": bool,
          "tp2": bool,
          "entry_time": datetime,
          "max_profit": float,
        }
      - now: 기준 시각 (없으면 datetime.utcnow())

    출력:
      - sell_qty: 매도 수량 (0이면 매도 없음)
      - sell_type: 매도 사유 문자열
      - new_state: 갱신된 state (삭제 필요 없다면 그대로)
      - profit_rate: 현재 수익률 (소수, 0.03 = 3%)
      - elapsed_min: 보유 시간(분)
    """
    if now is None:
        now = datetime.utcnow()

    if qty <= 0 or avg_price <= 0:
        return 0, "", state, 0.0, 0.0

    profit_rate = (price - avg_price) / avg_price  # 소수 (0.03 = 3%)

    tp1 = bool(state.get("tp1", False))
    tp2 = bool(state.get("tp2", False))
    entry_time = state.get("entry_time", now)
    max_profit = float(state.get("max_profit", 0.0))

    # 최고 수익률 갱신
    max_profit = max(max_profit, profit_rate)

    elapsed_min = (now - entry_time).total_seconds() / 60.0

    sell_qty = 0
    sell_type = ""

    # 1) 손절 (-2% 도달 시 전량 매도)
    if profit_rate <= -0.02:
        sell_qty = qty
        sell_type = "CUT_LOSS"
        # trader.py에서는 여기서 trade_state에서 삭제
        return sell_qty, sell_type, state | {"delete": True}, profit_rate, elapsed_min

    # 2) 익절 2차 (5% 수익 시 남은 것의 40% 매도)
    if profit_rate >= 0.05 and not tp2:
        sell_qty = max(1, int(qty * 0.4))
        sell_type = "PROFIT_5%"
        tp2 = True
        tp1 = True  # tp1 안 찍고 바로 5% 온 경우, tp1도 찍힌 것으로 처리
        new_state = {
            "tp1": tp1,
            "tp2": tp2,
            "entry_time": entry_time,
            "max_profit": max_profit,
        }
        return sell_qty, sell_type, new_state, profit_rate, elapsed_min

    # 3) 익절 1차 (3% 수익 시 60% 매도)
    if profit_rate >= 0.03 and not tp1:
        sell_qty = max(1, int(qty * 0.6))
        sell_type = "PROFIT_3%"
        tp1 = True
        entry_time = now  # 1차 익절 시점에 기준 시간 리셋
        new_state = {
            "tp1": tp1,
            "tp2": tp2,
            "entry_time": entry_time,
            "max_profit": max_profit,
        }
        return sell_qty, sell_type, new_state, profit_rate, elapsed_min

    # 4) 1차 익절 이후, 5% 못 가고 3%로 되돌림 → 전량 매도
    if tp1 and not tp2:
        if max_profit >= 0.04 and profit_rate <= 0.028:
            sell_qty = qty
            sell_type = "EXIT_RETRACE_TP1"
            return sell_qty, sell_type, state | {"delete": True}, profit_rate, elapsed_min

    # 5) 2차 익절 이후, 7% 이상 갔다가 5% 되돌림 → 전량 매도
    if tp2:
        if max_profit >= 0.07 and profit_rate <= 0.052:
            sell_qty = qty
            sell_type = "EXIT_RETRACE_TP2"
            return sell_qty, sell_type, state | {"delete": True}, profit_rate, elapsed_min

    # 6-1) 진입 후 60분 동안 -1.5% ~ +2.5%에서만 움직이면 전량 매도
    if elapsed_min >= 60 and -0.015 <= profit_rate <= 0.025:
        sell_qty = qty
        sell_type = "TIMEOUT_NO_TP"
        return sell_qty, sell_type, state | {"delete": True}, profit_rate, elapsed_min

    # 6-2) 1차 익절 후 45분 안에 5% 못 가고 +2.5%~+4.5%에서 횡보 → 전량 매도
    if tp1 and not tp2 and elapsed_min >= 45:
        if 0.025 <= profit_rate <= 0.045:
            sell_qty = qty
            sell_type = "TIMEOUT_AFTER_TP1"
            return sell_qty, sell_type, state | {"delete": True}, profit_rate, elapsed_min

    # 매도 없음 → 상태만 업데이트
    new_state = {
        "tp1": tp1,
        "tp2": tp2,
        "entry_time": entry_time,
        "max_profit": max_profit,
    }
    return 0, "", new_state, profit_rate, elapsed_min
