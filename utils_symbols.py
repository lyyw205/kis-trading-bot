# utils_symbols.py

def detect_region_exchange(symbol: str):
    """
    심볼 문자열로 region / exchange 코드를 판별하는 함수.
    한국: 숫자만 → KR
    미국: 문자 포함 → US
    """
    symbol = symbol.strip().upper()

    # ---------- 한국 주식 ----------
    if symbol.isdigit():
        return {
            "region": "KR",
            "symbol": symbol
        }

    # ---------- 미국 종목 ----------
    # 기본값 = NASDAQ
    exchange = "NAS"

    # NYSE 대표 종목
    NYSE_LIST = {
        "NIO", "BABA", "VZ", "KO", "F", "T", "MRK", "PFE",
        "UBER", "PCG", "ET", "CLF", "RIO", "VALE",
    }

    # AMEX/AMS (대표 종목)
    AMEX_LIST = {"DNN", "BTG"}

    if symbol in NYSE_LIST:
        exchange = "NYS"
    elif symbol in AMEX_LIST:
        exchange = "AMS"

    return {
        "region": "US",
        "symbol": symbol,
        "excd": exchange
    }


def generate_symbol_list(symbols):
    """
    여러 심볼 문자열 입력 → 자동 region/exchange 분류 리스트 반환
    """
    return [detect_region_exchange(s) for s in symbols]


from utils_symbols import generate_symbol_list

symbols = [ "IONQ", "BITF", "HIMS", "BMNR", "QUBT", "RKLB", "RR", "UUUU", "IRBT",
    "QSI", "REKR", "DVLT", "ACHR", "JOBY", "RGTI", "BURU", "PLUG", "IREN",
    "HIMZ", "ABVE", "RZLV", "LAES"]

result = generate_symbol_list(symbols)

print(result)
