import FinanceDataReader as fdr
import yfinance as yf
import pandas as pd
import time

def get_kr_top_volume(limit=100):
    print("🇰🇷 국내 주식 거래량 상위 수집 중...")
    # KRX 전체 종목 리스트 (가격/거래량 포함)
    df = fdr.StockListing('KRX')
    
    # 거래량 순 정렬
    df = df.sort_values(by='Volume', ascending=False)
    
    # 동전주(1000원 미만)나 거래정지 종목 제외하고 싶으면 필터링 가능
    # 여기서는 단순 거래량 상위만
    top = df.head(limit)
    
    result = []
    for _, row in top.iterrows():
        result.append({
            "region": "KR",
            "symbol": row['Code'],
            # "name": row['Name'] # 이름은 참고용 (봇 설정엔 불필요)
        })
    return result

def get_us_top_volume(limit=100):
    print("🇺🇸 미국 주식 거래량 상위 수집 중... (시간이 좀 걸립니다)")
    
    # 나스닥, 뉴욕 종목 리스트 가져오기
    # FDR은 기본 리스트만 주므로, yfinance로 덩어리 데이터를 가져오는게 빠름
    # 하지만 가장 쉬운 방법은 '거래량이 많은 대표 ETF'나 '기술주' 위주로 뽑는 것
    
    # 여기서는 FDR로 나스닥/뉴욕 전체 리스트를 가져와서 
    # 시가총액 상위 500개 중 거래량 순으로 정렬하는 방식을 씁니다.
    # (전체 종목 거래량 조회는 너무 오래 걸림)
    
    nas = fdr.StockListing('NASDAQ')
    nys = fdr.StockListing('NYSE')
    df = pd.concat([nas, nys])
    
    # 봇 매매에 적합하지 않은 너무 작은 주식 제외 (ETF 등 포함될 수 있음)
    # 시가총액 정보가 없으면 생략될 수 있으니, 주요 종목 위주로
    # 간단하게 yfinance의 'Most Actives' 스크레이퍼 대용으로
    # S&P500 + Nasdaq100 종목 중에서 거래량 상위를 뽑겠습니다.
    
    sp500 = fdr.StockListing('S&P500')
    nas100 = fdr.StockListing('NASDAQ') # 전체지만 일단 가져옴
    
    # 심볼만 추출해서 중복 제거
    symbols = list(set(sp500['Symbol'].tolist() + nas100['Symbol'].head(500).tolist()))
    
    # 배치 단위로 yfinance 조회 (속도 개선)
    print(f"   🔎 {len(symbols)}개 종목 실시간 거래량 조회 중...")
    
    chunk_size = 500
    valid_data = []
    
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i+chunk_size]
        try:
            tickers = yf.Tickers(" ".join(chunk))
            # 오늘(혹은 최근) 데이터 가져오기
            hist = tickers.history(period="1d")
            
            if 'Volume' in hist:
                # 마지막 날짜 거래량
                vols = hist['Volume'].iloc[-1]
                for sym, vol in vols.items():
                    valid_data.append((sym, vol))
        except:
            continue
            
    # 거래량 내림차순 정렬
    valid_data.sort(key=lambda x: x[1], reverse=True)
    top_us = valid_data[:limit]
    
    result = []
    for sym, vol in top_us:
        # 거래소 확인 (단순화: 일단 심볼만 있으면 됨, 거래소 코드는 봇이 알아서 찾으면 좋지만
        # 여기서는 KIS API 포맷인 NAS, NYS, AMS 매핑이 필요함)
        
        # 기본적으로 NAS로 설정하되, 목록에 따라 매핑
        # (봇 config에는 excd가 꼭 필요하므로 대략적으로 매핑)
        # 정확한 매핑을 위해선 복잡해지므로, 일단 NAS/NYS 구분 없이 
        # 대부분의 봇은 API 조회 시 거래소 코드를 틀려도 종목코드로 찾기도 함.
        # 여기서는 안전하게 'NAS' (나스닥) 우선으로 넣고, 
        # S&P500에 있으면 'NYS'일 확률이 높음.
        
        # *간단 매핑 로직*
        excd = "NAS" 
        if sym in sp500['Symbol'].values:
             # S&P500이고 나스닥 리스트에 없으면 NYS로 간주
             is_nasdaq = sym in nas['Symbol'].values
             if not is_nasdaq:
                 excd = "NYS"
        
        result.append({
            "region": "US",
            "symbol": sym,
            "excd": excd 
        })
        
    return result

if __name__ == "__main__":
    # 국내 100개
    kr_list = get_kr_top_volume(100)
    
    # 미국 100개
    us_list = get_us_top_volume(100)
    
    final_list = kr_list + us_list
    
    print("\n" + "="*50)
    print("👇 아래 내용을 config.py의 TARGET_STOCKS에 덮어쓰세요")
    print("="*50)
    print("TARGET_STOCKS = [")
    for item in final_list:
        if item['region'] == 'KR':
            print(f"    {{'region': 'KR', 'symbol': '{item['symbol']}'}},")
        else:
            print(f"    {{'region': 'US', 'symbol': '{item['symbol']}', 'excd': '{item['excd']}'}},")
    print("]")
    print("="*50)