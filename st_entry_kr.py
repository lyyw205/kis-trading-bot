# st_entry_kr.py
"""
한국 주식(KR) 엔트리 전략 래퍼.

역할:
- 최근 SEQ_LEN 개의 5분봉 시퀀스를 받아서
- 공통 엔트리 로직(st_entry_common.make_common_entry_signal)을 호출하고
- 향후 KR 전용 필터를 추가로 얹을 수 있는 자리를 확보한다.
"""

from typing import Dict, Any
import pandas as pd

from ml_features import SEQ_LEN
from st_entry_common import make_common_entry_signal


def make_entry_signal_kr(df: pd.DataFrame, params: dict) -> Dict[str, Any]:
    """
    KR 종목에 대한 엔트리 판단 함수.

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
    # 방어 코드: 데이터가 부족하면 바로 False
    if df is None or len(df) < SEQ_LEN:
        return {
            "entry_signal": False,
            "strategy_name": "NONE",
            "at_support": False,
            "is_bullish": False,
            "price_up": False,
        }

    # 최근 SEQ_LEN 캔들 시퀀스 잘라서 공통 로직에 넘김
    df_seq = df.iloc[-SEQ_LEN:].copy()

    base = make_common_entry_signal(df_seq, params)

    # =====================================================
    # 🔹 (향후 확장용) KR 전용 추가 필터를 넣고 싶으면 여기서 처리
    # 예시:
    #   - 특정 가격대(예: 1만 원 이상)만 매수
    #   - 거래대금(= 가격 * 거래량)이 일정 이상인 종목만 허용
    #   - 장 마감 직전 N분은 신규 진입 금지
    #
    # 지금은 공통 로직을 그대로 사용하되,
    # 확장을 위한 후킹 포인트만 열어두는 상태.
    # =====================================================

    return base