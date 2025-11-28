# db_backfill.py
import time
from datetime import datetime

from db_manager import BotDatabase
from brk_kis_client import KisDataFetcher
from brk_bithumb_client import BithumbDataFetcher

from config import (
    APP_KEY,
    APP_SECRET,
    ACCOUNT_NO,
    KR_UNIVERSE_STOCKS,
    US_UNIVERSE_STOCKS,
    CR_UNIVERSE_STOCKS,
)

from db_ohlcv_service import get_ohlcv_unified

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
    COIN_COUNT = 2000  # 24h라 좀 더 길게

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
