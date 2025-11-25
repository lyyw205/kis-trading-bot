# make_list_yf.py
import yfinance as yf
import json
import time

def get_kis_format_list(symbols):
    print(f"🚀 야후 파이낸스로 {len(symbols)}개 종목 정보 조회 중...\n")
    
    results = []
    
    # KIS API에서 사용하는 거래소 코드 매핑
    # 야후의 거래소 코드를 KIS 포맷(NAS, NYS, AMS)으로 변환
    def map_exchange(yf_exch):
        yf_exch = yf_exch.upper()
        
        # 나스닥 계열
        if "NASDAQ" in yf_exch or yf_exch in ["NMS", "NGM", "NCM", "PNK"]:
            return "NAS"
        # 뉴욕 계열
        elif "NYSE" in yf_exch or "NEW YORK" in yf_exch or yf_exch in ["NYQ", "NYS"]:
            return "NYS"
        # 아멕스 계열
        elif "AMEX" in yf_exch or "AMERICAN" in yf_exch or yf_exch in ["ASE", "AMS"]:
            return "AMS"
        # 기타 (OTC 등) -> 일단 나스닥으로 간주하거나 에러 처리
        else:
            return "NAS" # 기본값

    for symbol in symbols:
        try:
            # 야후 파이낸스 객체 생성
            ticker = yf.Ticker(symbol)
            
            # 정보 가져오기 (여기서 시간이 조금 걸림)
            info = ticker.info
            
            # 거래소 정보 추출
            # quoteType이 'EQUITY'나 'ETF'인 경우만 처리
            exchange_name = info.get('exchange', 'Unknown')
            
            # KIS 포맷으로 변환
            kis_excd = map_exchange(exchange_name)
            
            item = {
                "region": "US",
                "symbol": symbol.upper(), # 대문자 강제 변환
                "excd": kis_excd
            }
            results.append(item)
            
            print(f"✅ 확인 완료: {symbol.ljust(6)} | 야후: {exchange_name.ljust(10)} -> KIS: {kis_excd}")
            
        except Exception as e:
            print(f"❌ 조회 실패: {symbol} (티커를 확인하세요) - {e}")

    return results

# ==========================================
# 👇 조회할 종목 리스트 (여기에 다 넣으세요)
# ==========================================
MY_SYMBOLS = [
    "IONQ", "BITF", "HIMS", "BMNR", "QUBT", "RKLB", "RR", "UUUU",
    "IRBT", "QSI", "REKR", "DVLT", "ACHR", "JOBY", "RGTI", "BURU",
    "PLUG", "IREN", "ABVE", "RZLV", "LAES"
]

if __name__ == "__main__":
    final_list = get_kis_format_list(MY_SYMBOLS)

    print("\n" + "="*40)
    print("📋 결과 JSON (config.py에 붙여넣으세요)")
    print("="*40)
    
    # JSON 포맷으로 깔끔하게 출력
    print("[\n", end="")
    for i, item in enumerate(final_list):
        comma = "," if i < len(final_list) - 1 else ""
        print(f"    {json.dumps(item)}{comma}")
    print("]")
    print("="*40)