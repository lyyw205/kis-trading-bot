# core_trade.py
import time
from typing import Optional

from db_manager import BotDatabase
from brk_kis_client import KisDataFetcher
from brk_bithumb_client import BithumbDataFetcher
from core_trade_brain import GlobalRealTimeTrader
from config import (
    APP_KEY,
    APP_SECRET,
    ACCOUNT_NO,
    MODE,
    TARGET_STOCKS,
    AI_PARAMS,
    CR_TARGET_STOCKS,
    AI_PARAMS_COIN
)
from ml_model import load_model  # 네가 main.py에서 쓰던 것
from core_trade_brain_cr import CoinRealTimeTrader
from brk_bithumb_client import BithumbDataFetcher
from config import CR_UNIVERSE_STOCKS


DB_PATH = "trading.db"


def load_active_model(db: BotDatabase, for_region: Optional[str] = None):
    """
    settings에서 region별 active_model_path를 읽어서 모델 로드.

    - for_region=None  → "active_model_path"
    - for_region="KR"  → "active_model_path_kr"
    - for_region="US"  → "active_model_path_us"
    - for_region="CR"  → "active_model_path_coin"  ← 코인용 여기!
    """
    if for_region == "KR":
        key = "active_model_path_kr"
    elif for_region == "US":
        key = "active_model_path_us"
    elif for_region == "CR":
        key = "active_model_path_coin"
    else:
        key = "active_model_path"

    model_path = db.get_setting(key, "")

    if not model_path:
        db.log(f"🤖 ML 모델 없음 → 룰 기반으로만 동작합니다. (settings.{key} 비어 있음)")
        return None

    model = load_model(model_path, db)
    return model


def load_ml_threshold(
    db: BotDatabase,
    default: float = 0.55,
    for_region: str | None = None,
) -> float:
    """
    settings에서 ML threshold 값을 읽어서 float로 반환.

    - for_region=None  → "ml_threshold" (공통)
    - for_region="KR"  → "ml_threshold_kr"
    - for_region="US"  → "ml_threshold_us"
    - for_region="CR"  → "ml_threshold_coin" (코인 전용)
    """
    if for_region == "KR":
        key = "ml_threshold_kr"
    elif for_region == "US":
        key = "ml_threshold_us"
    elif for_region == "CR":
        key = "ml_threshold_coin"
    else:
        key = "ml_threshold"

    val = db.get_setting(key, str(default))
    try:
        return float(val)
    except Exception:
        return default


def select_targets_by_region(region: Optional[str]):
    """
    region 값에 따라 TARGET_STOCKS 필터링.
    - region is None: 전체
    - "US" / "KR": 해당 region만
    - "CR": 코인 유니버스 별도 사용
    """
    if region is None:
        return TARGET_STOCKS

    if region == "CR":
        # 코인 유니버스는 별도 리스트 사용
        return [t for t in CR_UNIVERSE_STOCKS if t.get("region") == "CR"]

    return [t for t in TARGET_STOCKS if t.get("region") == region]


def select_ai_params(region: Optional[str]):
    """
    지금은 공통 AI_PARAMS 하나지만,
    나중에 region별로 다르게 쓰고 싶으면 여기서 분기하면 됨.
    """
    # 예: if region == "CR": return AI_PARAMS_CR
    return AI_PARAMS


# ---------------------------
# 코인 전용 런너
# ---------------------------
def run_realtime_coin_bot():
    db = BotDatabase(DB_PATH)
    db.log(f"🤖 실시간 트레이딩 시작 (region=CR, mode={MODE})")
    db.log("🔍 [DEBUG] COIN 모드 → BithumbDataFetcher 사용")

    # 코인용 브로커
    fetcher = BithumbDataFetcher(mode=MODE, logger=db.log)

    # 코인 유니버스 / 파라미터
    targets = [t for t in CR_TARGET_STOCKS if t.get("region") == "CR"]
    params = select_ai_params("CR")

    model = load_active_model(db, for_region="CR")
    ml_threshold = load_ml_threshold(db, default=0.55, for_region="CR")

    ml_threshold = 0.35  # 일단 고정값으로 사용 (원하면 settings에서 빼도 됨)
    db.log(f"🔧 [설정] COIN ML Threshold = {ml_threshold}")
    db.log(f"🎯 COIN 대상 종목 개수: {len(targets)}")

    bot = CoinRealTimeTrader(
        fetcher=fetcher,
        targets=targets,
        params=params,
        db=db,
        model=model,            # 코인용 ML모델 아직 없으면 None
        ml_threshold=ml_threshold,
        dry_run=False,          # 필요시 True로 돌려도 됨
    )

    try:
        while True:
            try:
                bot.run_check()
                time.sleep(60)  # 코인은 1분마다 스캔
            except Exception as e:
                db.log(f"⚠️ [COIN] 메인 루프 에러: {e}")
                time.sleep(10)
    except KeyboardInterrupt:
        print("\n🛑 COIN 봇 종료")
        db.log("🛑 COIN 봇 수동 종료")


# ---------------------------
# 기존 주식(KIS) 런너
# ---------------------------
def run_realtime_kis_bot(region: Optional[str] = None):
    """
    - region=None  : 글로벌(모든 TARGET_STOCKS)
    - region="US"  : US만
    - region="KR"  : KR만
    """
    db = BotDatabase(DB_PATH)
    db.log(f"🤖 실시간 트레이딩 시작 (region={region or 'ALL'}, mode={MODE})")

    # 2) KIS Fetcher
    fetcher = KisDataFetcher(APP_KEY, APP_SECRET, ACCOUNT_NO, mode=MODE, logger=db.log)
    db.log(f"🔍 [DEBUG] KIS 모드: {MODE}")

    # 3) ML 모델 로드
    model = load_active_model(db, for_region=region)

    # 4) ML threshold 로드
    ml_threshold = load_ml_threshold(db, default=0.55)
    db.log(f"🔧 [설정] ML Threshold = {ml_threshold}")

    # 5) 대상 종목 / 파라미터 선택
    targets = select_targets_by_region(region)
    params = select_ai_params(region)

    db.log(f"🎯 대상 종목 개수: {len(targets)} (region={region or 'ALL'})")

    # 6) 트레이더 인스턴스 생성
    bot = GlobalRealTimeTrader(
        fetcher=fetcher,
        targets=targets,
        params=params,
        db=db,
        model=model,
        ml_threshold=ml_threshold,
    )

    # 7) 메인 루프
    try:
        while True:
            try:
                bot.run_check()
                time.sleep(150)  # 3분마다 스캔
            except Exception as e:
                db.log(f"⚠️ 메인 루프 에러: {e}")
                time.sleep(10)
    except KeyboardInterrupt:
        print("\n🛑 프로그램 종료")
        db.log("🛑 봇 수동 종료")


# ---------------------------
# 공통 엔트리
# ---------------------------
def run_realtime_bot(region: Optional[str] = None):
    """
    실시간 자동매매 공통 엔트리.
    - region=None  : 글로벌(모든 TARGET_STOCKS, KIS)
    - region="US"  : US만 (KIS)
    - region="KR"  : KR만 (KIS)
    - region="CR"  : 코인만 (Bithumb)
    """
    if region == "CR":
        return run_realtime_coin_bot()
    else:
        return run_realtime_kis_bot(region=region)