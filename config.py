# config.py
import os
from dotenv import load_dotenv

load_dotenv()

APP_KEY = os.getenv("KIS_APP_KEY")
APP_SECRET = os.getenv("KIS_APP_SECRET")
ACCOUNT_NO = os.getenv("KIS_ACCOUNT_NO")
MODE = os.getenv("KIS_MODE", "virtual")


TARGET_STOCKS = [
    {"region": "KR", "symbol": "0093G0"},
    {"region": "KR", "symbol": "010170"},
    {"region": "KR", "symbol": "424870"},
    {"region": "KR", "symbol": "125490"},
    {"region": "KR", "symbol": "099440"},
    {"region": "KR", "symbol": "0015N0"},
    {"region": "KR", "symbol": "005930"},
    {"region": "KR", "symbol": "308080"},
    {"region": "KR", "symbol": "359090"},
    {"region": "KR", "symbol": "225190"},
    {"region": "KR", "symbol": "006340"},
    {"region": "KR", "symbol": "459550"},
    {"region": "KR", "symbol": "437730"},
    {"region": "KR", "symbol": "049630"},
    {"region": "KR", "symbol": "092200"},
    {"region": "KR", "symbol": "067000"},
    {"region": "KR", "symbol": "476060"},
    {"region": "KR", "symbol": "486990"},
    {"region": "KR", "symbol": "450140"},
    {"region": "KR", "symbol": "090710"},
    {"region": "KR", "symbol": "001520"},
    {"region": "KR", "symbol": "215790"},
    {"region": "KR", "symbol": "469610"},
    {"region": "KR", "symbol": "102280"},
    {"region": "KR", "symbol": "457370"},
    {"region": "KR", "symbol": "051980"},
    {"region": "KR", "symbol": "318060"},
    {"region": "KR", "symbol": "033340"},
    {"region": "KR", "symbol": "195990"},
    {"region": "KR", "symbol": "314130"},
    {"region": "KR", "symbol": "007460"},
    {"region": "KR", "symbol": "030530"},
    {"region": "KR", "symbol": "078590"},
    {"region": "KR", "symbol": "098460"},
    {"region": "KR", "symbol": "004060"},
    {"region": "KR", "symbol": "078130"},
    {"region": "KR", "symbol": "403850"},
    {"region": "KR", "symbol": "460940"},
    {"region": "KR", "symbol": "084670"},
    {"region": "KR", "symbol": "413630"},
    {"region": "KR", "symbol": "200470"},
    {"region": "KR", "symbol": "0120G0"},
    {"region": "KR", "symbol": "000660"},
    {"region": "KR", "symbol": "038500"},
    {"region": "KR", "symbol": "114450"},
    {"region": "KR", "symbol": "032820"},
    {"region": "KR", "symbol": "015760"},
    {"region": "KR", "symbol": "128820"},
    {"region": "KR", "symbol": "034020"},
    {"region": "KR", "symbol": "166480"},

]

UNIVERSE_STOCKS = [
    {"region": "US", "symbol": "AAPL", "excd": "NAS"},
    {"region": "US", "symbol": "TSLA", "excd": "NAS"},
    {"region": "US", "symbol": "MSFT", "excd": "NAS"},
    {"region": "US", "symbol": "AMZN", "excd": "NAS"},
    {"region": "US", "symbol": "GOOGL", "excd": "NAS"},
    {"region": "US", "symbol": "NVDA", "excd": "NAS"},
    {"region": "US", "symbol": "META", "excd": "NAS"},
]

AI_PARAMS = {
    "lookback": 80,
    "band_pct": 0.02
}
