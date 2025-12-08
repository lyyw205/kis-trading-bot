# run_daily_pipeline.py
"""
장 마감 후 하루 루틴 전체를 한 번에 실행하는 스크립트.

[전체 흐름 요약]

(실시간 운용)
- main.py : 한국투자증권 API + ML 모델을 사용해서
            장중(프리장~정규장~애프터) 자동 매매/신호 감지 수행

(장 마감 후 이 스크립트 실행: python run_daily_pipeline.py)
1) (선택) OHLCV 백필: build_ohlcv_history.py
   - 오늘 장 중에 수집이 비었을 수 있는 캔들을 다시 채워 넣음
   - UNIVERSE_STOCKS 전체에 대해 5분봉 과거 데이터 보정/추가
   - ohlcv_data 테이블을 최신 상태로 맞춰줌

2) ML 샘플 생성: build_ml_seq_samples.py
   - ohlcv_data에서 Reversal + Momentum 조건에 맞는 "진입 시점"을 찾음
   - 각 진입 시점 이후 FUTURE_WINDOW 구간을 보면서
     TP/SL 기준으로 label(0/1)을 붙임
   - ml_seq_samples 테이블에 학습용 샘플(row: 하나의 진입 포인트)을 저장

3) ML 학습 + 백테스트 + 모델 조언: daily_ml_cycle.py
   - train_seq_model.py 실행
       · ml_seq_samples + ohlcv_data를 기반으로 시퀀스 피처 생성
       · RandomForestClassifier 학습
       · models 테이블에 버전 기록
       · settings.active_model_path 갱신
   - db_backtest.py 실행
       · 새로 학습한 모델로 과거 ohlcv_data 백테스트 수행 (네가 구현)
       · model_versions / backtests 테이블에 결과 기록
   - model_versions + backtests + 최근 실매매(trades) + settings를 모아서
     AI에 넘김 (make_model_update_advice)
       · active vs candidate 모델 비교
       · ml_threshold / max_positions 같은 설정 조정 제안
       · 모델 교체/유지/부분 적용 여부에 대한 조언 생성
   - 결과를 reports/YYYY-MM-DD_model_advice.txt 로 저장
   - logs/DB 에도 "daily_ml_cycle 완료" 로그 남김

4) 일일 트레이드 리포트 + 전략 아이디어: daily_ai_reports.py
   - trades 테이블에서 오늘 날짜의 트레이드들만 가져옴
   - 통계 요약:
       · 총 트레이드 수, 승률, 총/평균 손익, max_profit, max_loss
       · 심볼별 성과 요약
       · 베스트/워스트 트레이드 Top3
   - make_daily_trade_report() 호출
       · 위 통계를 기반으로 "일일 매매 리포트" 텍스트 생성
       · 오늘 장의 특징 / 문제점 / 내일부터 행동 가이드 정리
   - brainstorm_strategy_ideas() 호출
       · 종목/시간대/패턴별 성과를 보고
         새로운 전략 아이디어/필터링/ML 개선 방향을 제안
   - 결과를
       · reports/YYYY-MM-DD_daily_report.txt
       · reports/YYYY-MM-DD_strategy_ideas.txt
     두 개의 텍스트 파일로 저장
   - ai_reports 테이블에도 저장 → 대시보드에서 조회 가능
"""

import os
import subprocess
from datetime import date

from c_db_manager import BotDatabase

DB_PATH = "trading.db"

# ----------------------------------------------------------
# 🔧 옵션: OHLCV 백필 실행 여부
#   - True  : 매일 장 마감 후 build_ohlcv_history.py 실행
#   - False : 이미 DB가 충분히 채워져 있고, 자주 돌 필요 없을 때 끔
# ----------------------------------------------------------
RUN_BUILD_OHLCV = False


def run_step(name: str, cmd: list[str]) -> bool:
    """
    공통 실행 헬퍼.

    - name: 로그/콘솔에 찍을 단계 이름 (예: "ML 샘플 생성")
    - cmd : 실제로 실행할 커맨드 (예: ["python", "build_ml_seq_samples.py"])

    동작:
      1) 시작/종료 구분선 출력
      2) subprocess.run() 으로 외부 스크립트 실행
      3) stdout 모두 출력
      4) 에러(returncode != 0) 시 stderr 출력 + False 반환
         정상 종료(returncode == 0) 시 True 반환
    """
    print(f"\n==============================")
    print(f"▶ {name} 실행: {' '.join(cmd)}")
    print(f"==============================\n")

    result = subprocess.run(cmd, capture_output=True, text=True)

    # 스크립트 내부에서 print 한 내용
    if result.stdout:
        print(result.stdout)

    # 에러 처리
    if result.returncode != 0:
        print(f"❌ {name} 실행 실패")
        if result.stderr:
            print("---- stderr ----")
            print(result.stderr)
        return False

    print(f"✅ {name} 실행 완료")
    return True


if __name__ == "__main__":
    # DB 초기화 + 로그용 객체
    db = BotDatabase(DB_PATH)
    today_str = date.today().strftime("%Y-%m-%d")

    # 오늘 날짜 기준으로 전체 파이프라인 시작 로그
    db.log(f"🚀 run_daily_pipeline 시작 ({today_str})")

    # --------------------------------------------------
    # 1) OHLCV 백필 (옵션)
    #    - 장 중에 빠졌을 수 있는 캔들/심볼 데이터를 채워 넣어서
    #      이후 샘플 생성/학습/백테스트가 최대한 "완전한 데이터"로 돌아가도록 함.
    # --------------------------------------------------
    if RUN_BUILD_OHLCV:
        ok = run_step(
            "OHLCV 백필 (data_ohlcv_service.py)",
            ["python", "data_ohlcv_service.py"],
        )
        if not ok:
            db.log("❌ run_daily_pipeline: data_ohlcv_service.py 실패, 이후 스텝 중단")
            raise SystemExit(1)
    else:
        print("⏭ RUN_BUILD_OHLCV=False 설정이라 OHLCV 백필 스킵")

    # --------------------------------------------------
    # 2) ML 샘플 생성 (ml_build_seq_samples.py)
    #    - ohlcv_data → entry_signal 발생 구간 찾기
    #    - 각 진입시점 이후 TP/SL 도달 여부로 label(0/1) 생성
    #    - ml_seq_samples 테이블을 깨끗하게 다시 채움(DELETE 후 INSERT)
    # --------------------------------------------------
    ok = run_step(
        "ML 샘플 생성 (ml_build_seq_samples.py)",
        ["python", "ml_build_seq_samples.py"],
    )
    if not ok:
        db.log("❌ run_daily_pipeline: ml_build_seq_samples.py 실패, 이후 스텝 중단")
        raise SystemExit(1)

    # --------------------------------------------------
    # 3) 학습 + 백테스트 + 모델 조언 (run_daily_ml_cycle.py)
    #    - 내부에서 하는 일:
    #        (1) train_seq_model.py 실행
    #             · ml_seq_samples 기반으로 RandomForest 학습
    #             · models 테이블에 버전/정확도 기록
    #             · settings.active_model_path 갱신
    #        (2) db_backtest.py 실행
    #             · 새 모델로 과거 데이터 백테스트
    #             · model_versions/backtests 테이블에 결과 기록
    #        (3) make_model_update_advice() 호출
    #             · active vs candidate vs live 성능 비교
    #             · ml_threshold, max_positions 등 설정 조정 제안
    #             · 결과를 reports/YYYY-MM-DD_model_advice.txt 에 저장
    # --------------------------------------------------
    ok = run_step(
        "ML 학습/백테스트/모델 조언 (run_daily_ml_cycle.py)",
        ["python", "run_daily_ml_cycle.py"],
    )
    if not ok:
        db.log("❌ run_daily_pipeline: run_daily_ml_train.py 실패, 이후 스텝 중단")
        raise SystemExit(1)

    # --------------------------------------------------
    # 4) 일일 트레이드 리포트 + 전략 아이디어 (run_daily_ai_reports.py)
    #    - trades 테이블에서 오늘 날짜 트레이드만 가져옴
    #    - make_daily_trade_report():
    #         · 오늘 성과 정리 + 오늘의 문제점 + 내일부터의 행동 가이드 생성
    #    - brainstorm_strategy_ideas():
    #         · 종목/시간대/패턴별 성과를 기반으로
    #           실제 구현 가능한 전략/필터/ML 개선 아이디어 제안
    #    - 결과:
    #         · reports/YYYY-MM-DD_daily_report.txt
    #         · reports/YYYY-MM-DD_strategy_ideas.txt
    #         · ai_reports 테이블에도 저장 → 대시보드에서 조회 가능
    # --------------------------------------------------
    ok = run_step(
        "AI 일일 리포트/전략 아이디어 (run_daily_ai_reports.py)",
        ["python", "run_daily_ai_reports.py"],
    )
    if not ok:
        db.log("❌ run_daily_pipeline: run_daily_ai_reports.py 실패")
        raise SystemExit(1)

    # 전체 파이프라인 성공 로그
    db.log("🎉 run_daily_pipeline 전체 완료")
    print("\n🎉 모든 스텝 정상 완료!")
