# config.py
import os
from dotenv import load_dotenv

# 종목 리스트는 분리된 파일에서 가져옴
from config_us import US_TARGET_STOCKS, US_UNIVERSE_STOCKS
from config_kr import KR_TARGET_STOCKS, KR_UNIVERSE_STOCKS
from config_coin import CR_TARGET_STOCKS, CR_UNIVERSE_STOCKS

load_dotenv()

# -----------------------------
# 공통 KIS 설정
# -----------------------------
APP_KEY = os.getenv("KIS_APP_KEY")
APP_SECRET = os.getenv("KIS_APP_SECRET")
ACCOUNT_NO = os.getenv("KIS_ACCOUNT_NO")
MODE = os.getenv("KIS_MODE", "virtual")  # "real" / "virtual"

# -----------------------------
# AI / 전략 관련 공통 파라미터
# -----------------------------
AI_PARAMS = {
    "lookback": 80,
    "band_pct": 0.02,
}

AI_PARAMS_COIN = {
    "lookback": 80,
    "band_pct": 0.02,
}

# -----------------------------
# 종목 리스트 집계
# -----------------------------
# 기존 코드 호환용: TARGET_STOCKS / UNIVERSE_STOCKS 유지
TARGET_STOCKS = US_TARGET_STOCKS + KR_TARGET_STOCKS
UNIVERSE_STOCKS = US_UNIVERSE_STOCKS + KR_UNIVERSE_STOCKS

# 필요하면 개별 접근도 가능
US_TARGET_STOCKS = US_TARGET_STOCKS
US_UNIVERSE_STOCKS = US_UNIVERSE_STOCKS

KR_TARGET_STOCKS = KR_TARGET_STOCKS
KR_UNIVERSE_STOCKS = KR_UNIVERSE_STOCKS

CR_TARGET_STOCKS = CR_TARGET_STOCKS
CR_UNIVERSE_STOCKS = CR_UNIVERSE_STOCKS  # 코인은 별도 흐름에서 사용
