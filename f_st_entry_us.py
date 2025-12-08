# st_entry_us.py
"""
미국 주식(US) 엔트리 전략 래퍼.

역할:
- 최근 SEQ_LEN 개의 5분봉 시퀀스를 받아서
- 공통 엔트리 로직(st_entry_common.make_common_entry_signal)을 호출하고
- 향후 US 전용 필터(프리장/애프터장, 거래량 조건 등)를 추가할 수 있게 한다.
"""

from typing import Dict, Any
import pandas as pd

from c_ml_features import SEQ_LEN
from c_st_entry import make_common_entry_signal


def make_entry_signal_us(df: pd.DataFrame, params: dict) -> Dict[str, Any]:
    """
    US 종목에 대한 엔트리 판단 함수.

    입력:
      - df: 최근 캔들 전체 (시간 오름차순)
            columns: ['open','high','low','close','volume', ...]
      - params: 공통 전략 파라미터 (lookback, band_pct 등)

    반환:
      {
        "entry_signal": bool,
        "strategy_name": "REVERSAL" / "MOMENTUM_STRONG" / "NONE",
        "at_support": bool,
        "is_bullish": bool,
        "price_up": bool,
      }
    """
    if df is None or len(df) < SEQ_LEN:
        return {
            "entry_signal": False,
            "strategy_name": "NONE",
            "at_support": False,
            "is_bullish": False,
            "price_up": False,
        }

    df_seq = df.iloc[-SEQ_LEN:].copy()
    base = make_common_entry_signal(df_seq, params)

    # =====================================================
    # 🔹 (향후 확장용) US 전용 필터 예시:
    #   - 프리장/애프터마켓 시간대에는 신규 엔트리 금지
    #   - 특정 거래소(예: NASD만) 필터링 (실제 exchange 정보는 trader 쪽에서 관리)
    #   - 갭상승/갭하락 패턴 필터 등
    #
    # 필요해지면 base를 수정해서 entry_signal을 꺼버리거나
    # strategy_name을 "NONE"으로 바꾸는 식으로 튜닝하면 된다.
    # =====================================================

    return base
