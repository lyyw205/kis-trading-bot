# config.py
import os
from dotenv import load_dotenv

load_dotenv()

APP_KEY = os.getenv("KIS_APP_KEY")
APP_SECRET = os.getenv("KIS_APP_SECRET")
ACCOUNT_NO = os.getenv("KIS_ACCOUNT_NO")
MODE = os.getenv("KIS_MODE", "virtual")


TARGET_STOCKS = [
        {"region": "KR", "symbol": "005930"},
        {"region": "KR", "symbol": "000660"},
        {"region": "KR", "symbol": "950220"},
        {"region": "KR", "symbol": "412350"},
        {"region": "KR", "symbol": "488900"},
        {"region": "KR", "symbol": "462980"},
        {"region": "KR", "symbol": "486990"},
        {"region": "KR", "symbol": "450140"},
        {"region": "KR", "symbol": "424870"},
        {"region": "KR", "symbol": "459550"},
        {"region": "KR", "symbol": "469610"},
        {"region": "KR", "symbol": "064800"},
        {"region": "KR", "symbol": "067000"},
        {"region": "KR", "symbol": "010400"},
        {"region": "KR", "symbol": "469750"},
        {"region": "KR", "symbol": "083660"},
        {"region": "KR", "symbol": "049630"},
        {"region": "KR", "symbol": "034020"},
        {"region": "KR", "symbol": "359090"},
        {"region": "KR", "symbol": "068330"},
        {"region": "KR", "symbol": "009310"},
        {"region": "KR", "symbol": "014950"},
        {"region": "KR", "symbol": "125490"},
        {"region": "KR", "symbol": "317870"},
        {"region": "KR", "symbol": "004060"},
        {"region": "KR", "symbol": "010170"},
        {"region": "KR", "symbol": "090710"},
        {"region": "KR", "symbol": "003720"},
        {"region": "KR", "symbol": "035720"},
        {"region": "KR", "symbol": "041190"},
        {"region": "KR", "symbol": "340930"},
        {"region": "KR", "symbol": "099440"},
        {"region": "KR", "symbol": "066980"},
        {"region": "KR", "symbol": "440110"},
        {"region": "KR", "symbol": "014280"},
        {"region": "KR", "symbol": "051980"},
        {"region": "KR", "symbol": "378800"},
        {"region": "KR", "symbol": "011090"},
        {"region": "KR", "symbol": "107600"},
        {"region": "KR", "symbol": "005880"},
        {"region": "KR", "symbol": "131400"},
        {"region": "KR", "symbol": "033340"},
        {"region": "KR", "symbol": "464580"},
        {"region": "KR", "symbol": "001560"},
        {"region": "KR", "symbol": "067390"},
        {"region": "KR", "symbol": "227950"},
        {"region": "KR", "symbol": "065450"},

]

UNIVERSE_STOCKS = [
    # ===== A. Mega Tech (10) =====
    {"region": "US", "symbol": "AAPL", "excd": "NAS"},
    {"region": "US", "symbol": "MSFT", "excd": "NAS"},
    {"region": "US", "symbol": "GOOGL", "excd": "NAS"},
    {"region": "US", "symbol": "AMZN", "excd": "NAS"},
    {"region": "US", "symbol": "META", "excd": "NAS"},
    {"region": "US", "symbol": "NVDA", "excd": "NAS"},
    {"region": "US", "symbol": "TSLA", "excd": "NAS"},
    {"region": "US", "symbol": "AMD", "excd": "NAS"},
    {"region": "US", "symbol": "NFLX", "excd": "NAS"},
    {"region": "US", "symbol": "AVGO", "excd": "NAS"},

    # ===== B. Semiconductor / AI Infra (12) =====
    {"region": "US", "symbol": "INTC", "excd": "NAS"},
    {"region": "US", "symbol": "MU", "excd": "NAS"},
    {"region": "US", "symbol": "ASML", "excd": "NAS"},
    {"region": "US", "symbol": "TSM", "excd": "NYS"},
    {"region": "US", "symbol": "ARM", "excd": "NAS"},
    {"region": "US", "symbol": "SMCI", "excd": "NAS"},
    {"region": "US", "symbol": "QCOM", "excd": "NAS"},
    {"region": "US", "symbol": "ON", "excd": "NAS"},
    {"region": "US", "symbol": "AMAT", "excd": "NAS"},
    {"region": "US", "symbol": "LRCX", "excd": "NAS"},
    {"region": "US", "symbol": "KLAC", "excd": "NAS"},
    {"region": "US", "symbol": "CRUS", "excd": "NAS"},

    # ===== C. Financials (10) =====
    {"region": "US", "symbol": "JPM", "excd": "NYS"},
    {"region": "US", "symbol": "BAC", "excd": "NYS"},
    {"region": "US", "symbol": "WFC", "excd": "NYS"},
    {"region": "US", "symbol": "C", "excd": "NYS"},
    {"region": "US", "symbol": "GS", "excd": "NYS"},
    {"region": "US", "symbol": "MS", "excd": "NYS"},
    {"region": "US", "symbol": "SCHW", "excd": "NYS"},
    {"region": "US", "symbol": "BLK", "excd": "NYS"},
    {"region": "US", "symbol": "COIN", "excd": "NAS"},
    {"region": "US", "symbol": "PYPL", "excd": "NAS"},

    # ===== D. Healthcare / Bio (10) =====
    {"region": "US", "symbol": "JNJ", "excd": "NYS"},
    {"region": "US", "symbol": "PFE", "excd": "NYS"},
    {"region": "US", "symbol": "ABBV", "excd": "NYS"},
    {"region": "US", "symbol": "MRK", "excd": "NYS"},
    {"region": "US", "symbol": "LLY", "excd": "NYS"},
    {"region": "US", "symbol": "UNH", "excd": "NYS"},
    {"region": "US", "symbol": "TMO", "excd": "NYS"},
    {"region": "US", "symbol": "GILD", "excd": "NAS"},
    {"region": "US", "symbol": "AMGN", "excd": "NAS"},
    {"region": "US", "symbol": "BIIB", "excd": "NAS"},

    # ===== E. Energy / Materials (10) =====
    {"region": "US", "symbol": "XOM", "excd": "NYS"},
    {"region": "US", "symbol": "CVX", "excd": "NYS"},
    {"region": "US", "symbol": "OXY", "excd": "NYS"},
    {"region": "US", "symbol": "BP", "excd": "NYS"},
    {"region": "US", "symbol": "SHEL", "excd": "NYS"},
    {"region": "US", "symbol": "SLB", "excd": "NYS"},
    {"region": "US", "symbol": "HAL", "excd": "NYS"},
    {"region": "US", "symbol": "VLO", "excd": "NYS"},
    {"region": "US", "symbol": "MPC", "excd": "NYS"},
    {"region": "US", "symbol": "COP", "excd": "NYS"},

    # ===== F. Consumer / Retail (10) =====
    {"region": "US", "symbol": "WMT", "excd": "NYS"},
    {"region": "US", "symbol": "COST", "excd": "NAS"},
    {"region": "US", "symbol": "HD", "excd": "NYS"},
    {"region": "US", "symbol": "MCD", "excd": "NYS"},
    {"region": "US", "symbol": "SBUX", "excd": "NAS"},
    {"region": "US", "symbol": "TGT", "excd": "NYS"},
    {"region": "US", "symbol": "LOW", "excd": "NYS"},
    {"region": "US", "symbol": "NKE", "excd": "NYS"},
    {"region": "US", "symbol": "KO", "excd": "NYS"},
    {"region": "US", "symbol": "PEP", "excd": "NAS"},

    # ===== G. Industrial / Infra (8) =====
    {"region": "US", "symbol": "CAT", "excd": "NYS"},
    {"region": "US", "symbol": "DE", "excd": "NYS"},
    {"region": "US", "symbol": "GE", "excd": "NYS"},
    {"region": "US", "symbol": "RTX", "excd": "NYS"},
    {"region": "US", "symbol": "BA", "excd": "NYS"},
    {"region": "US", "symbol": "LMT", "excd": "NYS"},
    {"region": "US", "symbol": "HON", "excd": "NAS"},
    {"region": "US", "symbol": "UPS", "excd": "NYS"},

    # ===== H. Cloud / Software / Communication (6) =====
    {"region": "US", "symbol": "ORCL", "excd": "NYS"},
    {"region": "US", "symbol": "IBM", "excd": "NYS"},
    {"region": "US", "symbol": "CSCO", "excd": "NAS"},
    {"region": "US", "symbol": "VZ", "excd": "NYS"},
    {"region": "US", "symbol": "T", "excd": "NYS"},
    {"region": "US", "symbol": "SNOW", "excd": "NYS"},

    # ===== I. Platform / Entertainment (4) =====
    {"region": "US", "symbol": "DIS", "excd": "NYS"},
    {"region": "US", "symbol": "RBLX", "excd": "NYS"},
    {"region": "US", "symbol": "UBER", "excd": "NYS"},
    {"region": "US", "symbol": "LYFT", "excd": "NAS"},

    # ===== J. ETF (10) =====
    {"region": "US", "symbol": "SPY", "excd": "NYS"},
    {"region": "US", "symbol": "QQQ", "excd": "NAS"},
    {"region": "US", "symbol": "DIA", "excd": "NYS"},
    {"region": "US", "symbol": "IWM", "excd": "NYS"},
    {"region": "US", "symbol": "SMH", "excd": "NAS"},
    {"region": "US", "symbol": "XLE", "excd": "NYS"},
    {"region": "US", "symbol": "XLK", "excd": "NYS"},
    {"region": "US", "symbol": "XLV", "excd": "NYS"},
    {"region": "US", "symbol": "ARKK", "excd": "BAT"},
    {"region": "US", "symbol": "BITO", "excd": "NYS"},

    # ===== K. Growth / Theme (10) =====
    {"region": "US", "symbol": "PLTR", "excd": "NYS"},
    {"region": "US", "symbol": "ROKU", "excd": "NAS"},
    {"region": "US", "symbol": "SQ", "excd": "NYS"},
    {"region": "US", "symbol": "SHOP", "excd": "NYS"},
    {"region": "US", "symbol": "AFRM", "excd": "NAS"},
    {"region": "US", "symbol": "CRWD", "excd": "NAS"},
    {"region": "US", "symbol": "NET", "excd": "NYS"},
    {"region": "US", "symbol": "DDOG", "excd": "NAS"},
    {"region": "US", "symbol": "IONQ", "excd": "NYS"},
    {"region": "US", "symbol": "PATH", "excd": "NYS"},

]

AI_PARAMS = {
    "lookback": 80,
    "band_pct": 0.02
}
