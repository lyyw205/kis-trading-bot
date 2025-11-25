# main.py
import time
import joblib

from db import BotDatabase
from kis_api import KisDataFetcher
from trader import GlobalRealTimeTrader
from config import APP_KEY, APP_SECRET, ACCOUNT_NO, MODE, TARGET_STOCKS, AI_PARAMS


def load_active_model(db: BotDatabase):
    """settings.active_model_path를 읽어 모델을 로딩."""
    model_path = db.get_setting("active_model_path", "")

    if not model_path:
        db.log("🤖 ML 모델 없음 → 룰 기반으로만 동작합니다.")
        return None

    try:
        model = joblib.load(model_path)
        db.log(f"🤖 활성 모델 로드 완료: {model_path}")
        return model
    except Exception as e:
        db.log(f"⚠️ 모델 로드 실패: {e} → 룰 기반으로 동작")
        return None


def load_ml_threshold(db: BotDatabase):
    """settings.ml_threshold 값을 float로 반환."""
    val = db.get_setting("ml_threshold", "0.55")
    try:
        return float(val)
    except:
        return 0.55


if __name__ == "__main__":
    # -----------------------------
    # 0) DB 연결
    # -----------------------------
    db = BotDatabase("trading.db")
    db.log("🤖 트레이딩 시스템 시작 (DB + ML 연동)")

    # -----------------------------
    # 1) KIS API 연결
    # -----------------------------
    fetcher = KisDataFetcher(APP_KEY, APP_SECRET, ACCOUNT_NO, mode=MODE)

    # -----------------------------
    # 2) ML 모델 로드
    # -----------------------------
    model = load_active_model(db)

    # -----------------------------
    # 3) ML Threshold 로드
    # -----------------------------
    ml_threshold = load_ml_threshold(db)

    # -----------------------------
    # 4) 트레이더 인스턴스 생성
    # -----------------------------
    bot = GlobalRealTimeTrader(
        fetcher=fetcher,
        targets=TARGET_STOCKS,
        params=AI_PARAMS,
        db=db,
        model=model,
        ml_threshold=ml_threshold,
    )

    # -----------------------------
    # 5) 메인 루프 실행
    # -----------------------------
    try:
        while True:
            try:
                bot.run_check()
                time.sleep(180)  # 3분마다 스캔
            except Exception as e:
                db.log(f"⚠️ 메인 루프 에러: {e}")
                time.sleep(10)

    except KeyboardInterrupt:
        print("\n🛑 프로그램 종료")
        db.log("🛑 봇 수동 종료")
