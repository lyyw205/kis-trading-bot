# # universe_collector.py
# 유니버스 OHLCV 모으는 스크립트 (매일 자동으로 돌아가면 표본 데이터 지속적으로 쌓임)
# 추후 윈도우 작업 스케쥴러에 추가

import time
from datetime import datetime, timedelta

from db import BotDatabase
from kis_api import KisDataFetcher  # 이미 쓰고 있는 클래스 그대로
from config import APP_KEY, APP_SECRET, ACCOUNT_NO, UNIVERSE_STOCKS

DB_PATH = "trading.db"

def collect_universe_ohlcv():
    db = BotDatabase(DB_PATH)
    db.log("📦 유니버스 OHLCV 수집 시작")

    fetcher = KisDataFetcher(APP_KEY, APP_SECRET, ACCOUNT_NO, mode="virtual", logger=db.log)

    # 5분봉 기준: 최근 3일치 정도만 매일 업데이트 (중복은 INSERT OR IGNORE로 막힘)
    interval = "5m"
    count = 300   # 5분봉 300개 ≒ 약 25시간 (장시간 기준 넉넉하게)

    total_success = 0
    total_fail = 0

    for t in UNIVERSE_STOCKS:
        region = t["region"]
        symbol = t["symbol"]
        excd = t.get("excd")

        time.sleep(0.2)  # API 부하 조절

        try:
            df = fetcher.get_ohlcv(region, symbol, excd, interval=interval, count=count)
        except Exception as e:
            db.log(f"❌ [유니버스수집 실패] {region} {symbol}: API 예외 {e}")
            total_fail += 1
            continue

        if df is None or df.empty:
            db.log(f"🚫 [유니버스수집 스킵] {region} {symbol}: 데이터 없음")
            total_fail += 1
            continue

        try:
            db.save_ohlcv_df(region, symbol, interval, df)
            db.log(f"✅ [유니버스수집 완료] {region} {symbol} 5m {len(df)}개 저장")
            total_success += 1
        except Exception as e:
            db.log(f"⚠️ [유니버스 저장 실패] {region} {symbol}: {e}")
            total_fail += 1

    db.log(f"📊 [유니버스 수집완료] 성공:{total_success} 실패:{total_fail} / 총:{len(UNIVERSE_STOCKS)}")


if __name__ == "__main__":
    collect_universe_ohlcv()
