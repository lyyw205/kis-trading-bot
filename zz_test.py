# db_backfill.py
import time
from datetime import datetime
import pandas as pd
import pybithumb  # [변경] 빗썸 라이브러리 사용

from db_manager import BotDatabase
from brk_kis_client import KisDataFetcher

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
    
    [변경사항]
    - CR(코인): pybithumb를 사용하여 빗썸 데이터 수집
    - KR/US: 기존 통합 서비스(get_ohlcv_unified) 사용
    """
    db = BotDatabase(DB_PATH)
    db.log("📦 [UNIVERSE] OHLCV 과거 데이터 백필 시작 (CR source: Bithumb)")

    kis_client = KisDataFetcher(APP_KEY, APP_SECRET, ACCOUNT_NO, mode="real", logger=db.log)

    interval = "5m"
    
    # ----------------------------------------
    # 수집 개수 설정
    # ----------------------------------------
    KR_COUNT = 1600
    US_COUNT = 1600
    # 빗썸은 API 호출 시 개수 지정이 아니라 주는 대로 다 받으므로
    # 이 숫자는 KR/US 처럼 엄격하게 쓰이진 않지만 로그용으로 남겨둠
    COIN_COUNT = 2000 

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

        # 자산군별 count 설정
        if region == "KR":
            count = KR_COUNT
        elif region == "US":
            count = US_COUNT
        elif region == "CR":
            count = COIN_COUNT
        else:
            db.log(f"⚠️ 지원하지 않는 region: {region} {symbol}, 스킵")
            continue

        db.log(f"⏳ 백필 시작: {region} {symbol}")

        df = None

        # -------------------------------------------------------
        # [수정] CR(코인)인 경우 -> pybithumb 사용
        # -------------------------------------------------------
        if region == "CR":
            try:
                # 1. 심볼 변환: "KRW-BTC" -> "BTC" (pybithumb는 티커만 사용)
                ticker = symbol.replace("KRW-", "")
                
                # 2. 빗썸 API 호출 (interval="minute5")
                # pybithumb.get_ohlcv는 제공 가능한 최대 기간을 DataFrame으로 반환함
                df = pybithumb.get_ohlcv(ticker, interval="minute5")
                
                if df is not None and not df.empty:
                    # 필요한 컬럼만 선택 (pybithumb는 기본적으로 이 컬럼들을 제공함)
                    df = df[["open", "high", "low", "close", "volume"]]
                    
                    # 결측 제거 및 정렬
                    df = df.dropna().sort_index()
                    
            except Exception as e:
                error_msg = f"Bithumb Fetch Fail: {e}"
                db.log(f"⚠️ {region} {symbol} 조회 실패: {error_msg}")
                db.log_universe_backfill_failure(
                    region=region,
                    symbol=symbol,
                    excd="BITHUMB",
                    interval=interval,
                    error_type="fetch_error",
                    error_message=error_msg,
                )
                continue

        # -------------------------------------------------------
        # [기존] KR / US 인 경우 -> 통합 서비스 사용
        # -------------------------------------------------------
        else:
            try:
                df = get_ohlcv_unified(
                    region=region,
                    symbol=symbol,
                    exchange=excd,
                    interval=interval,
                    count=count,
                    kis_client=kis_client,
                    upbit_client=None, 
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

        # -------------------------------------------------------
        # 공통: 데이터 저장 처리
        # -------------------------------------------------------
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
                f"📏 {region} {symbol}: 5분봉 {len(df)}개 확보 | "
                f"{first_ts} ~ {last_ts}"
            )
        except Exception:
            db.log(f"📏 {region} {symbol}: 5분봉 {len(df)}개 (index 정보 없음)")

        # 4) DB 저장
        # save_ohlcv_df 함수가 내부적으로 region, symbol, interval 컬럼을 추가해서 저장해줌
        # df의 인덱스는 dt로 변환되어 저장됨
        db.save_ohlcv_df(region, symbol, interval, df)
        db.log(f"✅ 백필 완료: {region} {symbol}")

        # API 호출 간격 조절
        time.sleep(0.3)

    # 마지막 실행 시각 기록
    db.set_setting(
        "last_universe_ohlcv_backfill",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )

    db.log("🎉 [UNIVERSE] OHLCV 과거 데이터 백필 전체 완료")


if __name__ == "__main__":
    backfill_universe_ohlcv()