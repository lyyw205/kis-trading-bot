# build_ohlcv_history.py
import time
from datetime import datetime
from db import BotDatabase
from kis_api import KisDataFetcher
from config import APP_KEY, APP_SECRET, ACCOUNT_NO, TARGET_STOCKS

DB_PATH = "trading.db"

if __name__ == "__main__":
    db = BotDatabase(DB_PATH)
    db.log("📦 OHLCV 과거 데이터 백필 시작")

    fetcher = KisDataFetcher(APP_KEY, APP_SECRET, ACCOUNT_NO, mode="virtual")

    # KR/US 모두 5분봉 1개월치 (대략 1600개)
    KR_COUNT = 1600
    US_COUNT = 1600

    for t in TARGET_STOCKS:
        region = t["region"]
        symbol = t["symbol"]
        excd = t.get("excd")

        db.log(f"⏳ 백필: {region} {symbol} ({excd})")

        try:
            # KR 5분봉
            if region == "KR":
                interval = "5m"
                df = fetcher.get_ohlcv(region, symbol, interval=interval, count=KR_COUNT)

            # US 5분봉
            else:
                interval = "5m"
                df = fetcher.get_ohlcv(region, symbol, excd, interval=interval, count=US_COUNT)

        except Exception as e:
            db.log(f"⚠️ OHLCV 조회 실패: {region} {symbol} | {e}")
            continue

        if df is None or df.empty:
            db.log(f"⚠️ 데이터 없음: {region} {symbol}")
            continue

        # 저장은 여기에서 단 한 번만
        db.save_ohlcv_df(region, symbol, interval, df)
        db.log(f"✅ 백필 완료: {region} {symbol} ({len(df)}개)")

        time.sleep(0.2)

    db.log("🎉 OHLCV 과거 데이터 백필 전체 완료")
