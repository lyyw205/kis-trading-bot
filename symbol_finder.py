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
        """
        야후 파이낸스 exchange 문자열 → KIS EXCD(NAS/NYS/AMS)
        OTC / PINK / OTCQX / OTCQB 등은 None 반환
        """
        if not yf_exch:
            return None

        name = yf_exch.upper().strip()

        # 1) OTC / PINK 계열 먼저 컷
        otc_keywords = ["OTC", "PINK", "OTCQB", "OTCQX", "OTCPK", "PNK", "OBB"]
        if any(k in name for k in otc_keywords):
            print(f"⚠️ OTC 감지 → excd=None 처리: {yf_exch}")
            return None

        # 2) 나스닥 계열
        if "NASDAQ" in name or name in ["NMS", "NGM", "NCM"]:
            return "NAS"

        # 3) 뉴욕 계열
        if "NYSE" in name or "NEW YORK" in name or name in ["NYQ", "NYS"]:
            return "NYS"

        # 4) 아멕스 계열
        if "AMEX" in name or "AMERICAN" in name or name in ["ASE", "AMS"]:
            return "AMS"

        # 5) 그 외 애매한 건 None
        print(f"⚠️ 알 수 없는 거래소 코드 → excd=None 처리: {yf_exch}")
        return None

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
    "NVDA", "TSLA", "QQQ", "GOOGL", "ORCL", "AMD", "SPY", "META",
    "HOOD", "GOOG", "AVGO", "COIN", "AMZN", "MSTR", "MU", "SOFI",
    "SNDK", "PLTR", "INTC", "MSFT", "BABA", "CRCL", "BMNR", "NFLX", "AAPL", "IREN", "HIMS", "CRWV",
    "ZS", "TMC", "OPEN", "ADBE", "RGTI", "APP", "CLSK", "OKLO",
    "QUBT", "RDDT", "ONDS", "BITF", "SGBX", "RKLB","NKE","BA","QCOM","QBTS","BURU","INHD","GPUS",
    "DVLT","LOBO","AIIO","PLUG","DFLI","ASST","LAES","FMFC","MSGM","SLDP","NFE","VSME","GNS","RZLV","NUAI",
    "SES","REKR","LAC","BYND","WTO","YDKG",
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