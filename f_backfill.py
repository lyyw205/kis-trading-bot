# "UNIVERSE OHLCV 백필 스크립트 (KR/US/CR 통합)

#  - KR / US / COIN(CR) 유니버스 전체에 대해 5분봉 OHLCV 과거 데이터를
#    각 거래소 API(KIS, Bithumb)에서 조회해서 DB에 저장하는 배치용 스크립트

# 주요 기능:
# 1) 초기 설정
#    - BotDatabase(DB_PATH="trading.db")로 로그 및 저장 기능 사용
#    - KIS 주식용 KisDataFetcher(APP_KEY, APP_SECRET, ACCOUNT_NO, mode="real")
#    - 코인용 BithumbDataFetcher(logger=db.log)
#    - 공통 인터벌: interval = "5m"
#    - 자산군별 캔들 개수 설정:
#      · KR_COUNT = 1600
#      · US_COUNT = 1600
#      · COIN_COUNT = 20000 (24시간 시장이라 더 길게)

# 2) 유니버스 합치기
#    - KR_UNIVERSE_STOCKS + US_UNIVERSE_STOCKS + CR_UNIVERSE_STOCKS 를 하나로 합쳐
#      all_universe 리스트 구성
#    - 각 요소는 {region, symbol, excd} 형태로, region 값에 따라
#      · KR: 국내 주식
#      · US: 미국 주식
#      · CR: 코인(KRW-BTC 등)

# 3) 종목별 OHLCV 백필 로직 (backfill_universe_ohlcv)
#    - all_universe 순회:
#      · region / symbol / excd 추출
#      · region별로 사용할 count 결정 (KR/US/CR)
#      · 지원하지 않는 region이면 로그 찍고 스킵
#      · 진행 로그: "⏳ 백필: {region} {symbol} ({excd})" 출력
#    - get_ohlcv_unified(...) 호출:
#      · region, symbol, exchange(excd), interval("5m"), count, kis_client, upbit_client를 넘겨
#        KIS or Bithumb API 중 적절한 쪽에서 캔들을 통합적으로 가져오도록 설계
#    - 예외 처리:
#      · 조회 중 오류 → fetch_error 로 universe_backfill_failures에 기록
#      · 데이터 없음(df None/empty) → empty_data 로 universe_backfill_failures에 기록
#    - 데이터 있을 때:
#      · 인덱스 기준 최소/최대 시각(first_ts, last_ts)을 로그로 남기고
#        "5분봉 N개 | from ~ to ~" 형태로 범위 출력
#      · BotDatabase.save_ohlcv_df(region, symbol, interval, df)로 ohlcv_data 테이블에 저장
#      · "✅ 백필 완료" 로그 후 0.2초 sleep (API 과부하 방지)

# 4) 마지막 실행 시각 기록
#    - 전체 루프 완료 후:
#      · settings 테이블에 key="last_universe_ohlcv_backfill" 로 현재 시각(YYYY-MM-DD HH:MM:SS) 저장
#    - "🎉 [UNIVERSE] OHLCV 과거 데이터 백필 전체 완료" 로그 출력

# 5) 단독 실행 엔트리
#    - __main__ 블록에서 backfill_universe_ohlcv()를 호출해
#      이 파일을 스크립트로 직접 실행하면 전체 유니버스 백필을 수행"


import time
from datetime import datetime

from c_db_manager import BotDatabase
from f_kis_client import KisDataFetcher
from e_bithumb_client import BithumbDataFetcher

from c_config import (
    APP_KEY,
    APP_SECRET,
    ACCOUNT_NO,
    KR_UNIVERSE_STOCKS,
    US_UNIVERSE_STOCKS,
    CR_UNIVERSE_STOCKS,
)

from c_ohlcv_service import get_ohlcv_unified

DB_PATH = "trading.db"


def backfill_universe_ohlcv():
    """
    KR / US / COIN 유니버스 전체에 대해 5분봉 OHLCV 과거 데이터를 백필한다.
    - 조회: get_ohlcv_unified()
    - 저장: BotDatabase.save_ohlcv_df()
    - 실패 기록: BotDatabase.log_universe_backfill_failure()
    """
    db = BotDatabase(DB_PATH)
    db.log("📦 [UNIVERSE] OHLCV 과거 데이터 백필 시작")

    kis_client = KisDataFetcher(APP_KEY, APP_SECRET, ACCOUNT_NO, mode="real", logger=db.log)
    upbit_client = BithumbDataFetcher(logger=db.log)

    interval = "5m"
    KR_COUNT = 1600
    US_COUNT = 1600
    COIN_COUNT = 20000  # 24h라 좀 더 길게

    # 유니버스를 한 번에 다루기 위해 리스트 합치고, 안에 region으로 구분
    all_universe = (
        list(KR_UNIVERSE_STOCKS)
        + list(US_UNIVERSE_STOCKS)
        + list(CR_UNIVERSE_STOCKS)
    )

    for t in all_universe:
        region = t["region"]          # "KR" / "US" / "CR"
        symbol = t["symbol"]          # "005930" / "VSME" / "KRW-BTC"
        excd = t.get("excd")          # KRX / NAS / UPBIT or None

        # 자산군별 count
        if region == "KR":
            count = KR_COUNT
        elif region == "US":
            count = US_COUNT
        elif region == "CR":
            count = COIN_COUNT
        else:
            db.log(f"⚠️ 지원하지 않는 region: {region} {symbol}, 스킵")
            continue

        db.log(f"⏳ 백필: {region} {symbol} ({excd})")

        # 1) OHLCV 조회 (통합 서비스 사용)
        try:
            df = get_ohlcv_unified(
                region=region,
                symbol=symbol,
                exchange=excd,
                interval=interval,
                count=count,
                kis_client=kis_client,
                upbit_client=upbit_client,
            )
        except Exception as e:
            db.log(f"⚠️ OHLCV 조회 실패: {region} {symbol} | {e}")
            db.log_universe_backfill_failure(
                region=region,
                symbol=symbol,
                excd=excd,
                interval=interval,
                error_type="fetch_error",
                error_message=str(e),
            )
            continue

        # 2) 데이터 없음 처리
        if df is None or df.empty:
            db.log(f"⚠️ 데이터 없음: {region} {symbol}")
            db.log_universe_backfill_failure(
                region=region,
                symbol=symbol,
                excd=excd,
                interval=interval,
                error_type="empty_data",
                error_message="no rows",
            )
            continue

        # 3) 실제 범위/개수 로그
        try:
            first_ts = df.index.min()
            last_ts = df.index.max()
            db.log(
                f"📏 {region} {symbol}: 5분봉 {len(df)}개 | "
                f"from {first_ts} → {last_ts}"
            )
        except Exception:
            db.log(f"📏 {region} {symbol}: 5분봉 {len(df)}개 (index 정보 없음)")

        # 4) DB 저장
        db.save_ohlcv_df(region, symbol, interval, df)
        db.log(f"✅ 백필 완료: {region} {symbol} ({len(df)}개 저장)")

        time.sleep(0.2)

    # 마지막 실행 시각 기록
    db.set_setting(
        "last_universe_ohlcv_backfill",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    db.log("🎉 [UNIVERSE] OHLCV 과거 데이터 백필 전체 완료")


if __name__ == "__main__":
    backfill_universe_ohlcv()
