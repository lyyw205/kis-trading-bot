# debug_momentum.py
import pandas as pd
import time
from db import BotDatabase
from kis_api import KisDataFetcher
from config import APP_KEY, APP_SECRET, ACCOUNT_NO, MODE

# RSI 계산 함수
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)

def debug_stock(symbol):
    db = BotDatabase("trading.db")
    fetcher = KisDataFetcher(APP_KEY, APP_SECRET, ACCOUNT_NO, MODE, db)
    
    print(f"\n🔍 [{symbol}] Momentum 조건 정밀 분석 중...")
    
    # 5분봉 120개 가져오기
    df = fetcher.get_ohlcv("US", symbol, "NAS", "5m", 120) # 거래소 excd는 상황에 맞게
    if df is None or df.empty:
        print("❌ 데이터 수집 실패")
        return

    # 지표 계산
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    df["rsi"] = calculate_rsi(df["close"], 14)
    df["vol_ma20"] = df["volume"].rolling(20).mean()
    
    last = df.iloc[-1]
    
    # 조건값 확인
    price = last['close']
    ma20 = last['ma20']
    ma60 = last['ma60']
    rsi = last['rsi']
    vol = last['volume']
    vol_ma = last['vol_ma20']
    
    # 조건 체크
    cond_align = (price > ma20) and (ma20 > ma60)
    cond_rsi = (50 <= rsi <= 80) # 완화된 조건 적용
    cond_vol_strict = (vol > vol_ma)
    cond_vol_loose = (vol > vol_ma * 0.7) # 완화된 조건
    is_bullish = (price > last['open'])
    
    print(f"--------------------------------------------------")
    print(f"📊 현재가: {price} | MA20: {ma20:.2f} | MA60: {ma60:.2f}")
    print(f"📊 RSI: {rsi:.2f} (목표: 50~80)")
    print(f"📊 VOL: {vol} (평균: {vol_ma:.0f})")
    print(f"--------------------------------------------------")
    
    print(f"1. 정배열 (P>20>60) : {'✅ 통과' if cond_align else '❌ 탈락 (역배열/혼조세)'}")
    print(f"2. RSI 조건         : {'✅ 통과' if cond_rsi else '❌ 탈락 (과매수/과매도)'}")
    print(f"3. 거래량 (빡빡함)  : {'✅ 통과' if cond_vol_strict else '❌ 탈락 (평균 이하)'}")
    print(f"4. 거래량 (0.7배)   : {'✅ 통과' if cond_vol_loose else '❌ 탈락 (0.7배 미만)'}")
    print(f"5. 양봉 여부        : {'✅ 통과' if is_bullish else '❌ 탈락 (음봉)'}")
    
    final_strict = cond_align and cond_rsi and cond_vol_strict and is_bullish
    final_loose = cond_align and cond_rsi and cond_vol_loose and is_bullish
    
    print(f"--------------------------------------------------")
    print(f"결과(기존) : {'🚀 매수신호' if final_strict else '💤 대기'}")
    print(f"결과(완화) : {'🚀 매수신호' if final_loose else '💤 대기'}")

if __name__ == "__main__":
    # 테스트하고 싶은 종목 입력 (예: NVDA, TSLA)
    debug_stock("NVDA") 
    debug_stock("TSLA")
    debug_stock("QQQ")