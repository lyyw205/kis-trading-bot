import time
from datetime import datetime
import pandas as pd
import pybithumb  # 빗썸 라이브러리 필수

from db_manager import BotDatabase
from config import CR_UNIVERSE_STOCKS  # 코인 리스트만 가져옴

# DB 파일 경로
DB_PATH = "trading.db"

def backfill_cr_ohlcv():
    """
    CR(코인) 유니버스에 대해 5분봉 OHLCV 데이터를 빗썸에서 가져와 DB에 백필한다.
    주식(KR, US) 로직은 포함하지 않음.
    """
    db = BotDatabase(DB_PATH)
    db.log("📦 [CR-ONLY] 코인 5분봉 백필 시작 (Source: Bithumb)")

    interval = "5m"  # DB 저장용 라벨

    # CR_UNIVERSE_STOCKS 순회
    for t in CR_UNIVERSE_STOCKS:
        region = t["region"]   # "CR"이어야 함
        symbol = t["symbol"]   # 예: "KRW-BTC"
        
        # 혹시 모를 오설정 방지
        if region != "CR":
            continue

        db.log(f"⏳ 백필 시작: {symbol}")

        try:
            # 1. 심볼 변환: "KRW-BTC" -> "BTC"
            ticker = symbol.replace("KRW-", "")
            
            # 2. 빗썸 API 호출 (5분봉)
            # pybithumb는 제공 가능한 최대 기간을 한 번에 가져옴 (보통 최근 1500~2000개)
            df = pybithumb.get_ohlcv(ticker, interval="minute5")
            
            if df is None or df.empty:
                db.log(f"⚠️ 데이터 없음: {symbol}")
                continue

            # 3. 데이터 정제
            # 필요한 컬럼만 명시적으로 선택
            df = df[["open", "high", "low", "close", "volume"]]
            
            # 결측치 제거 및 시간순 정렬
            df = df.dropna().sort_index()

            # 4. 로그 출력 (범위 확인)
            first_ts = df.index.min()
            last_ts = df.index.max()
            db.log(f"📏 {symbol}: {len(df)}개 확보 ({first_ts} ~ {last_ts})")

            # 5. DB 저장
            # BotDatabase.save_ohlcv_df가 알아서 INSERT OR IGNORE 처리함
            db.save_ohlcv_df(region, symbol, interval, df)
            db.log(f"✅ 저장 완료: {symbol}")

        except Exception as e:
            db.log(f"❌ {symbol} 처리 중 에러: {e}")
            # 에러 발생 시 멈추지 않고 다음 코인으로 넘어감
            continue
        
        # API 호출 속도 조절 (너무 빠르면 차단될 수 있음)
        time.sleep(0.3)

    # 작업 완료 시간 기록
    db.set_setting(
        "last_cr_backfill",
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    db.log("🎉 [CR-ONLY] 코인 데이터 백필 작업 완료")


if __name__ == "__main__":
    backfill_cr_ohlcv()