# db_ohlcv_service.py
"""
OHLCV 공통 서비스 레이어

- KIS (주식) + 업비트 (코인) + (향후) 시뮬레이터 등
  여러 브로커에서 가져온 캔들 데이터를
  하나의 진입점으로 라우팅하는 역할.

- trader / build_ohlcv_history / 백테스트 코드 등에서
  이 모듈만 바라보고, 내부에서 어떤 브로커를 쓰는지는 캡슐화.
"""

from typing import Optional

import pandas as pd

from brk_kis_client import KisDataFetcher
from brk_bithumb_client import BithumbDataFetcher


def get_ohlcv_unified(
    region: str,
    symbol: str,
    *,
    exchange: Optional[str] = None,
    interval: str = "5m",
    count: int = 200,
    kis_client: Optional[KisDataFetcher] = None,
    upbit_client: Optional[BithumbDataFetcher] = None,
) -> pd.DataFrame:
    """
    자산군(region)별로 알맞은 브로커 클라이언트에 위임해서 OHLCV 반환.

    - region:
        - "KR"   → KIS 국내주식
        - "US"   → KIS 해외주식
        - "COIN" → 업비트 코인 (symbol은 "KRW-BTC" 같은 market 코드)
    - interval:
        - 주식: "5m", "1d"
        - 코인: "minute5", "day" 등 (Upbit 스펙)
    """
    if region == "KR":
        if kis_client is None:
            raise ValueError("KIS 클라이언트(kis_client)가 필요합니다.")
        # 기존 KisDataFetcher.get_ohlcv 재사용
        return kis_client.get_ohlcv(
            region="KR",
            symbol=symbol,
            interval=interval,
            count=count,
        )

    elif region == "US":
        if kis_client is None:
            raise ValueError("KIS 클라이언트(kis_client)가 필요합니다.")
        # 해외는 exchange(excd) 필요
        return kis_client.get_ohlcv(
            region="US",
            symbol=symbol,
            exchange=exchange,
            interval=interval,
            count=count,
        )

    elif region == "CR":
        if upbit_client is None:
            raise ValueError("Upbit 클라이언트(upbit_client)가 필요합니다.")

        if interval == "5m":
            upbit_interval = "minute5"
        elif interval == "1m":
            upbit_interval = "minute1"
        elif interval in ["day", "1d"]:
            upbit_interval = "day"
        else:
            upbit_interval = interval  # 이미 Upbit 포맷일 수 있음

        return upbit_client.get_coin_ohlcv(
            market=symbol,
            interval=upbit_interval,
            count=count,
        )

    # 알 수 없는 region이면 빈 DF 반환
    return pd.DataFrame()
