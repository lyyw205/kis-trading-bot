import requests
import pandas as pd
import yfinance as yf
import json
import time

def get_realtime_top_100_robust():
    print("⏳ 야후 파이낸스 페이지에 직접 접속 중...")
    
    # 1. 야후 파이낸스 Most Actives 페이지 직접 크롤링 (yahoo_fin 미사용)
    url = "https://finance.yahoo.com/most-active?count=100&offset=0"
    
    # 봇 차단 방지용 헤더
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    try:
        response = requests.get(url, headers=headers)
        
        # 판다스로 HTML 내의 테이블 읽기
        dfs = pd.read_html(response.text)
        
        if not dfs:
            print("❌ 테이블을 찾을 수 없습니다.")
            return []
            
        df = dfs[0] # 첫 번째 테이블이 보통 주식 목록입니다.
        
        # Symbol 컬럼만 추출하여 리스트로 변환
        top_symbols = df['Symbol'].tolist()
        
        # 최대 100개 확보 (혹시 모르니)
        top_symbols = top_symbols[:100]
        
        print(f"✅ 1차 수집 완료: {len(top_symbols)}개 심볼 (Top 1: {top_symbols[0]})")
        
        # 2. 거래소 정보(NAS/NYS) 매핑 (yfinance 활용)
        print("⏳ 한투 API 포맷으로 변환 중 (거래소 확인)...")
        
        # 100개를 한 번에 Tickers 객체로 생성 (속도 최적화)
        tickers = yf.Tickers(" ".join(top_symbols))
        final_list = []
        
        for symbol in top_symbols:
            try:
                # fast_info를 사용해 빠르게 접근
                info = tickers.tickers[symbol].fast_info
                exchange_code = info.get('exchange', 'UNKNOWN')
                
                # 거래소 코드 변환 로직
                kis_excd = "NAS" # 기본값
                
                # 나스닥 계열
                if exchange_code in ['NMS', 'NGM', 'NCM', 'PNK', 'NAS', 'Nasdaq']:
                    kis_excd = 'NAS'
                # 뉴욕 계열
                elif exchange_code in ['NYQ', 'NYS', 'NYE', 'PCX', 'NYSE']: 
                    kis_excd = 'NYS'
                # 아멕스 계열
                elif exchange_code in ['ASE', 'AMS', 'AMEX']:
                    kis_excd = 'AMS'
                
                final_list.append({
                    "region": "US",
                    "symbol": symbol,
                    "excd": kis_excd
                })
                
            except Exception as e:
                # yfinance 정보 조회 실패 시 기본값(NAS)으로 추가하거나 스킵
                # print(f"⚠️ {symbol} 조회 실패, 기본값 적용")
                final_list.append({
                    "region": "US",
                    "symbol": symbol,
                    "excd": "NAS" # 안전장치
                })
                
        return final_list
        
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        return []

if __name__ == "__main__":
    result = get_realtime_top_100_robust()
    
    if result:
        print(f"\n✅ 최종 변환 완료: {len(result)}개")
        print(json.dumps(result, indent=4))
    else:
        print("데이터를 가져오지 못했습니다.")