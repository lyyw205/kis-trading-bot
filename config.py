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
    "min_len": 120,
    "atr_period": 14,

    "atr_max_ratio": 0.06,
    "hl_max_ratio": 0.08,

    "use_rsi_filter": False,
    "use_trend_filter": False,
    "rsi_min": 35.0,
    "rsi_max": 75.0,

    "ml_min_r3": 0.0,
    "ml_min_r6": 0.0,
    "ml_min_r12": 0.0,
    "ml_min_score": 0.0,
    "ml_max_worst": -0.03,
    "ml_min_pos_ratio": 0.34,
    "ml_horizon_weights": [0.4, 0.35, 0.25],

    "ml_strong_score": 0.0020,
    "ml_weak_score": 0.0005,
    "atr_for_strong": 0.03,

    "lookback": 20,
    "band_pct": 0.005,
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
