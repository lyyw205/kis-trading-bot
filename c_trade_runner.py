# "실시간 자동매매 공통 런너 (KIS 주식 / Bithumb 코인 / Binance 코인 통합 엔트리)

#  - 세 개의 실시간 트레이딩 봇을 하나의 진입점에서 관리하는 런처 스크립트
#    (KIS 주식용, 빗썸 코인용, 바이낸스 코인용)

# 주요 기능:
# 1) 공통 설정/헬퍼
#    - load_active_model(db, for_region):
#      · settings 테이블의 active_model_path_* 값을 읽어서 ML 모델 로드
#      · KR / US / CR 별로 각각 다른 모델 경로 키 사용
#    - load_ml_threshold(db, default, for_region):
#      · ml_threshold_* 설정값을 region별로 읽어와 float로 반환
#    - select_targets_by_region(region):
#      · region에 따라 사용할 타겟 리스트 선택
#        · None → 공통 TARGET_STOCKS
#        · "CR" → CR_UNIVERSE_STOCKS 중 CR
#        · "BI" → BI_TARGET_STOCKS
#        · 그 외(KR/US) → TARGET_STOCKS 중 해당 region
#    - select_ai_params(region):
#      · 코인 계열("CR", "BI")이면 AI_PARAMS_COIN, 그 외는 AI_PARAMS 사용

# 2) 코인 전용 빗썸 런너 (run_realtime_coin_bot)
#    - BotDatabase 생성 후 BithumbDataFetcher로 브로커 준비
#    - CR_TARGET_STOCKS 중 region == "CR" 대상
#    - AI_PARAMS_COIN 사용, 모델/ML threshold는 멀티스케일 엔트리 내부 사용 전제로 None/0.0
#    - CoinRealTimeTrader(fetcher, targets, params, db, ...) 인스턴스 생성
#    - 무한 루프에서 bot.run_check() + 60초 sleep
#    - 예외 발생 시 로그 찍고 10초 대기 후 재시도, KeyboardInterrupt 시 정상 종료 로그

# 3) 바이낸스 전용 런너 (run_realtime_binance_bot)
#    - BotDatabase 생성 후 Binance API 키/시크릿을 환경변수에서 읽어 BinanceDataFetcher 생성
#    - select_targets_by_region("BI")로 BI_TARGET_STOCKS 전체 불러온 뒤
#      · market=="spot" / "futures" 기준으로 타겟 분리
#    - AI_PARAMS_COIN 사용
#    - Spot:
#      · BinanceCoinRealTimeTrader(fetcher, spot_targets, params, db, dry_run=False, market_type="spot")
#    - Futures:
#      · BinanceCoinRealTimeTrader(fetcher, futures_targets, params, db, dry_run=False, market_type="futures", leverage=3)
#    - 트레이더가 하나도 없으면 경고 로그 후 종료
#    - 하나 이상의 트레이더가 있으면 traders 리스트에 담아
#      · 무한 루프에서 각 trader.run_check()를 순차 실행(Spot → Futures)
#      · 60초 sleep (Binance API rate limit 고려)
#      · 예외 시 로그 + 10초 대기, KeyboardInterrupt 시 종료 로그

# 4) KIS 주식 런너 (run_realtime_kis_bot)
#    - GlobalRealTimeTrader (stocks.trader.core_trade_brain.GlobalRealTimeTrader) 사용
#    - BotDatabase(DB_PATH) 생성
#    - KisDataFetcher(APP_KEY, APP_SECRET, ACCOUNT_NO, mode, logger) 생성
#    - load_active_model(db, for_region=region) 으로 해당 region용 ML 모델 로드
#    - load_ml_threshold(...) 로 ML threshold 로드
#    - select_targets_by_region(region)으로 KR/US/ALL 타겟 목록 구성
#    - select_ai_params(region)으로 AI 파라미터 선택
#    - GlobalRealTimeTrader(fetcher, targets, params, db, model, ml_threshold) 인스턴스 생성
#    - 무한 루프에서 bot.run_check() + 150초 sleep
#    - 예외 시 로그 + 10초 대기, KeyboardInterrupt 시 종료 로그

# 5) 공통 엔트리 (run_realtime_bot)
#    - 호출 시 region 인자를 기준으로 어떤 런너를 쓸지 결정:
#      · region == "CR" → run_realtime_coin_bot()
#      · region == "BI" → run_realtime_binance_bot()
#      · 그 외(KR/US/None) → run_realtime_kis_bot(region=region)
#    - CLI나 상위 스크립트에서 단일 함수만 호출하면 거래소/자산군별 런너가 분기되도록 하는 진입점 역할"

import time
import os
from typing import Optional

from c_db_manager import BotDatabase
from f_kis_client import KisDataFetcher
from e_bithumb_client import BithumbDataFetcher
from c_config import (
    APP_KEY,
    APP_SECRET,
    ACCOUNT_NO,
    MODE,
    TARGET_STOCKS,
    AI_PARAMS,
    CR_TARGET_STOCKS,
    AI_PARAMS_COIN,
    BI_SPOT_UNIVERSE_STOCKS,    
    BI_FUTURES_UNIVERSE_STOCKS, 
)
from c_ml_model import load_model
from c_config import CR_UNIVERSE_STOCKS

# Binance 관련
from bi_client import BinanceDataFetcher
from bi_trade_brain import BinanceCoinRealTimeTrader
import threading  # <--- 추가
import traceback  # <--- 추가

DB_PATH = "trading.db"


def load_active_model(db: BotDatabase, for_region: Optional[str] = None):
    # (기존 코드 동일)
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
    # (기존 코드 동일)
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
    # (기존 코드 동일)
    if region is None:
        return TARGET_STOCKS

    if region == "CR":
        return [t for t in CR_UNIVERSE_STOCKS if t.get("region") == "CR"]
    
    if region == "BI":
        # config.py의 BI_TARGET_STOCKS를 그대로 반환
        return BI_SPOT_UNIVERSE_STOCKS + BI_FUTURES_UNIVERSE_STOCKS
    return [t for t in TARGET_STOCKS if t.get("region") == region]


def select_ai_params(region: Optional[str]):
    # (기존 코드 동일)
    if region in ("CR", "BI"):
        return AI_PARAMS_COIN
    return AI_PARAMS


# ---------------------------
# 코인 전용 런너 (빗썸)
# ---------------------------
def run_realtime_coin_bot():
    # (기존 코드 동일)
    db = BotDatabase()
    db.log(f"🤖 실시간 트레이딩 시작 (region=CR, mode={MODE})")
    db.log("🔍 [DEBUG] COIN 모드 → BithumbDataFetcher 사용")

    from e_bith_trade_brain import CoinRealTimeTrader

    fetcher = BithumbDataFetcher(mode=MODE, logger=db.log)
    targets = [t for t in CR_TARGET_STOCKS if t.get("region") == "CR"]
    params = select_ai_params("CR")
    
    model = None
    ml_threshold = 0.0
    db.log("🔧 [설정] COIN ML Threshold = (미사용, Multi-Scale 모델 내장)")
    db.log(f"🎯 COIN 대상 종목 개수: {len(targets)}")

    bot = CoinRealTimeTrader(
        fetcher=fetcher,
        targets=targets,
        params=params,
        db=db,
        model=model,
        ml_threshold=ml_threshold,
        dry_run=False,
    )

    try:
        while True:
            try:
                bot.run_check()
                time.sleep(60) 
            except Exception as e:
                db.log(f"⚠️ [COIN] 메인 루프 에러: {e}")
                time.sleep(10)
    except KeyboardInterrupt:
        print("\n🛑 COIN 봇 종료")
        db.log("🛑 COIN 봇 수동 종료")


# ---------------------------
# 🔹 바이낸스 전용 런너 (region=BI) - Threading 적용
# ---------------------------

def _run_bi_thread_loop(trader, name, interval=60):
    """
    개별 트레이더를 독립된 스레드에서 무한 반복하는 헬퍼 함수
    """
    trader.db.log(f"🚀 [{name}] 스레드 루프 시작... (간격: {interval}초)")
    
    while True:
        try:
            # 1. 트레이더 로직 실행
            trader.run_check()
            
            # 2. 대기 (API Rate Limit 및 과부하 방지)
            time.sleep(interval)
            
        except Exception as e:
            # 루프 전체가 죽지 않도록 방어 + 에러 로그
            error_msg = traceback.format_exc()
            trader.db.log(f"❌ [{name}] 치명적 에러 발생 (10초 대기 후 재시작): {e}\n{error_msg}")
            time.sleep(10)

def run_realtime_binance_bot():
    """
    Binance Spot/Futures 통합 런너 (멀티 스레드 버전)
    - Spot과 Futures 봇을 별도의 스레드로 동시에 실행
    - 현물 쪽에서 에러가 나도 선물 쪽은 멈추지 않음
    """
    db = BotDatabase()
    db.log(f"🤖 실시간 트레이딩 시작 (region=BI, mode={MODE})")
    
    # API 키 로드
    BINANCE_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_SECRET = os.getenv("BINANCE_SECRET_KEY", "")

    # 브로커(Fetcher) 생성
    # (스레드 안전성을 위해 필요하다면 fetcher를 각각 생성할 수도 있지만, 
    #  단순 REST API 호출이라면 공유해도 보통 무방합니다. 여기선 공유합니다.)
    fetcher = BinanceDataFetcher(
        api_key=BINANCE_KEY, 
        secret_key=BINANCE_SECRET, 
        mode=MODE, 
        logger=db.log
    )
    db.log("🔍 [DEBUG] BI 모드 → BinanceDataFetcher 생성 완료")

    # 1. 타겟 및 파라미터 로드
    spot_targets = BI_SPOT_UNIVERSE_STOCKS
    futures_targets = BI_FUTURES_UNIVERSE_STOCKS
    params = select_ai_params("BI")

    threads = []

    # 2. Spot Trader 스레드 준비
    if spot_targets:
        db.log(f"🎯 [Spot] 대상 종목: {len(spot_targets)}개 -> 스레드 생성")
        trader_spot = BinanceCoinRealTimeTrader(
            fetcher=fetcher,
            targets=spot_targets,
            params=params,
            db=db,
            dry_run=False,
            market_type="spot"
        )

        # 스레드 생성
        t_spot = threading.Thread(
            target=_run_bi_thread_loop, 
            args=(trader_spot, "BI_SPOT_BOT", 60)
        )
        t_spot.daemon = True # 메인 프로세스 종료 시 같이 종료
        threads.append(t_spot)

    # 3. Futures Trader 스레드 준비
    ENABLE_BI_FUTURES = False  # ✅ 임시로 비활성화

    # 3. Futures Trader 스레드 준비
    if ENABLE_BI_FUTURES and futures_targets:
        db.log(f"🎯 [Futures] 대상 종목: {len(futures_targets)}개 -> 스레드 생성")
        trader_fut = BinanceCoinRealTimeTrader(
            fetcher=fetcher,
            targets=futures_targets,
            params=params,
            db=db,
            dry_run=False,
            market_type="futures",
            leverage=3
        )

        trader_fut.sync_positions_from_binance()

        t_fut = threading.Thread(
            target=_run_bi_thread_loop, 
            args=(trader_fut, "BI_FUTURES_BOT", 60)
        )
        t_fut.daemon = True
        threads.append(t_fut)
    else:
        if futures_targets:
            db.log("⏸️ [Futures] 잔액 이슈로 임시 비활성화됨 (ENABLE_BI_FUTURES=False)")

    # 4. 스레드 시작
    for t in threads:
        t.start()

    # 5. 메인 스레드 생존 유지 (스레드들이 백그라운드에서 돌게 둠)
    try:
        while True:
            time.sleep(60) # 1분마다 메인 프로세스 생존 확인
            # 필요하다면 여기서 주기적으로 상태 로그를 찍거나 DB 연결을 체크할 수 있음
    except KeyboardInterrupt:
        print("\n🛑 BI 봇 종료 (KeyboardInterrupt)")
        db.log("🛑 BI 봇 수동 종료")


# ---------------------------
# 기존 주식(KIS) 런너
# ---------------------------
def run_realtime_kis_bot(region: Optional[str] = None):
    # (기존 코드 동일)
    from f_trade_brain import GlobalRealTimeTrader
    
    db = BotDatabase(DB_PATH)
    db.log(f"🤖 실시간 트레이딩 시작 (region={region or 'ALL'}, mode={MODE})")

    fetcher = KisDataFetcher(APP_KEY, APP_SECRET, ACCOUNT_NO, mode=MODE, logger=db.log)
    db.log(f"🔍 [DEBUG] KIS 모드: {MODE}")

    model = load_active_model(db, for_region=region)
    ml_threshold = load_ml_threshold(db, default=0.55)
    db.log(f"🔧 [설정] ML Threshold = {ml_threshold}")

    targets = select_targets_by_region(region)
    params = select_ai_params(region)

    db.log(f"🎯 대상 종목 개수: {len(targets)} (region={region or 'ALL'})")

    bot = GlobalRealTimeTrader(
        fetcher=fetcher,
        targets=targets,
        params=params,
        db=db,
        model=model,
        ml_threshold=ml_threshold,
    )

    try:
        while True:
            try:
                bot.run_check()
                time.sleep(150)
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
    """
    if region == "CR":
        return run_realtime_coin_bot()
    elif region == "BI":
        return run_realtime_binance_bot()
    else:
        return run_realtime_kis_bot(region=region)