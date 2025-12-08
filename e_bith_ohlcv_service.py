# "OHLCV 공통 서비스 레이어 (KIS 주식 + 빗썸/업비트 코인 통합 진입점)

#  - KIS(국내/미국 주식)과 코인 브로커(Bithumb/Upbit)를 한 함수로 감싸서,
#    region 값만으로 알맞은 브로커의 get_ohlcv / get_coin_ohlcv 를 호출해주는 라우터 역할 모듈.

# 주요 기능:
# 1) get_ohlcv_unified(region, symbol, exchange=None, interval='5m', count=200,
#                      kis_client=None, upbit_client=None) → DataFrame
#    - region 값에 따라 적절한 브로커로 위임:
#      · region == "KR"
#        - KisDataFetcher 필요 (kis_client)
#        - kis_client.get_ohlcv(region="KR", symbol, interval, count) 호출
#        - 국내 주식 5분봉/일봉 등 조회
#      · region == "US"
#        - KisDataFetcher 필요
#        - kis_client.get_ohlcv(region="US", symbol, exchange=excd, interval, count)
#        - 미국 주식 5분봉/일봉 등 조회 (exchange 파라미터 사용)
#      · region == "CR"
#        - BithumbDataFetcher 필요 (upbit_client 라고 쓰여 있지만 실질적으로 코인 브로커)
#        - interval을 주식 스타일에서 코인 스타일로 매핑:
#          · '5m' → 'minute5'
#          · '1m' → 'minute1'
#          · 'day'/'1d' → 'day'
#          · 그 외는 그대로 사용
#        - upbit_client.get_coin_ohlcv(market=symbol, interval=upbit_interval, count)
#        - 코인 5분봉/1분봉/일봉 조회
#      · 그 외 region:
#        - 인식 못 하면 빈 DataFrame 반환

# 2) 설계 의도
#    - trader, 백필 스크립트(db_backfill.py), 백테스트 등에서
#      각 브로커의 상세 호출 방식을 몰라도 되도록,
#      `get_ohlcv_unified()` 하나만 바라보게 만드는 추상화 레이어.
#    - 나중에 Binance 시뮬레이터, replay 엔진 등 다른 데이터 소스가 추가되더라도
#      이 함수 안에서만 분기 로직을 확장하면 상위 코드 수정 최소화 가능."

#### 주식，　코인　분리함　


import pandas as pd
from typing import Optional
from f_kis_client import KisDataFetcher


import pandas as pd
from typing import Optional

from e_bithumb_client import BithumbDataFetcher


def get_coin_ohlcv(
    symbol: str,
    *,
    interval: str = "5m",
    count: int = 200,
    upbit_client: Optional[BithumbDataFetcher] = None,
):

    if upbit_client is None:
        raise ValueError("upbit_client (BithumbDataFetcher)가 필요합니다.")

    # 주식식 interval → 코인 interval 매핑
    if interval == "5m":
        api_interval = "minute5"
    elif interval == "1m":
        api_interval = "minute1"
    elif interval in ("1d", "day"):
        api_interval = "day"
    else:
        api_interval = interval

    return upbit_client.get_coin_ohlcv(
        market=symbol,
        interval=api_interval,
        count=count,
    )