# 프로젝트 공용 설정/파라미터 허브 파일

# - .env에서 KIS API 키/계좌번호/운영모드(MODE)를 불러오는 공통 환경설정
# - AI/전략 공통 파라미터(AI_PARAMS, AI_PARAMS_COIN)를 정의
# - 각 시장별 종목 리스트(config_us/kr/cr/bi)를 import 해서
#   기존 코드에서 TARGET_STOCKS/UNIVERSE_STOCKS로 한 번에 접근 가능하도록 집계

# 주요 내용:
# 1) APP_KEY, APP_SECRET, ACCOUNT_NO, MODE
#    : KIS(한국투자증권) API 공통 환경 변수 로딩
# 2) AI_PARAMS
#    : 주식용 기본 AI/전략 파라미터 (lookback, band_pct 등)
# 3) AI_PARAMS_COIN
#    : 코인용 엔트리/필터/ML 스코어 관련 파라미터 세트
# 4) US/KR/CR/BI_TARGET_STOCKS, _UNIVERSE_STOCKS
#    : 각 config_* 파일에서 가져온 종목 리스트를 재노출
# 5) TARGET_STOCKS, UNIVERSE_STOCKS
#    : 기존 코드 호환용으로 KR+US를 합친 전체 주식 타겟/유니버스 리스트


import os
from dotenv import load_dotenv

# 종목 리스트는 분리된 파일에서 가져옴
from f_config_us import US_TARGET_STOCKS, US_UNIVERSE_STOCKS
from f_config_kr import KR_TARGET_STOCKS, KR_UNIVERSE_STOCKS
from e_bith_config import CR_TARGET_STOCKS, CR_UNIVERSE_STOCKS
from bi_config import BI_TARGET_STOCKS, BI_UNIVERSE_STOCKS, BI_SPOT_UNIVERSE_STOCKS, BI_FUTURES_UNIVERSE_STOCKS

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

BI_TARGET_STOCKS = BI_TARGET_STOCKS
BI_UNIVERSE_STOCKS = BI_UNIVERSE_STOCKS

