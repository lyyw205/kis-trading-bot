# main.py
import time
import joblib
from db import BotDatabase
from kis_api import KisDataFetcher
from trader import GlobalRealTimeTrader
from config import APP_KEY, APP_SECRET, ACCOUNT_NO, MODE, TARGET_STOCKS, AI_PARAMS
from ml_model import load_model



def load_active_model(db: BotDatabase):
    """settings.active_model_path를 읽어 ML 모델 로딩 (ml_model.load_model 사용)."""
    model_path = db.get_setting("active_model_path", "")

    if not model_path:
        db.log("🤖 ML 모델 없음 → 룰 기반으로만 동작합니다.")
        return None

    # ⭐ 중앙화된 모델 로딩 함수 사용
    model = load_model(model_path, db)
    return model   # load_model()이 None 또는 모델 객체를 반환함


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
    fetcher = KisDataFetcher(APP_KEY, APP_SECRET, ACCOUNT_NO, mode=MODE, logger=db.log)

    # -----------------------------
    # 2) ML 모델 로드
    # -----------------------------
    model = load_active_model(db)

    # -----------------------------
    # 3) ML Threshold 로드
    # -----------------------------
    # ml_threshold = load_ml_threshold(db)
    ml_threshold = 0.30
    db.log(f"🔧 [설정] ML Threshold 강제 적용: {ml_threshold} (기존 DB설정 무시)")
    
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
                time.sleep(150)  # 3분마다 스캔
            except Exception as e:
                db.log(f"⚠️ 메인 루프 에러: {e}")
                time.sleep(10)

    except KeyboardInterrupt:
        print("\n🛑 프로그램 종료")
        db.log("🛑 봇 수동 종료")
