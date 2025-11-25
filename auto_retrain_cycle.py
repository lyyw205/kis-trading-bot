# auto_retrain_cycle.py
"""
1) ohlcv_data 기반 ML 시퀀스 샘플 생성
   - build_ml_seq_samples.py

2) 새 시퀀스 모델 학습 + 모델 버전/설정 기록
   - train_seq_model.py  (model_versions, settings.active_model_path 등)

3) 최신 모델 포함 백테스트 실행
   - backtest_seq_model.py  (rule_only vs ML 성능을 backtests 테이블에 기록)

이 스크립트만 실행하면
- DB에 학습용 샘플 추가
- 모델 학습 및 버전 관리
- 백테스트 결과까지 누적 저장
모두 한 번에 처리됨.
"""

import os
import sys
import subprocess
from datetime import datetime

from db import BotDatabase

DB_PATH = "trading.db"


def run_step(db: BotDatabase, cmd, step_name: str):
    """하나의 서브 스크립트를 실행하고 로그를 남기는 헬퍼."""
    db.log(f"[AUTO] {step_name} 시작: {' '.join(cmd)}")
    try:
        # stdout/stderr 를 그대로 터미널에 보여주고, 실패 시 예외 발생
        result = subprocess.run(cmd, check=True)
        db.log(f"[AUTO] {step_name} 완료 (returncode={result.returncode})")
    except subprocess.CalledProcessError as e:
        db.log(f"[AUTO] {step_name} 실패: {e}")
        raise


if __name__ == "__main__":
    db = BotDatabase(DB_PATH)
    db.log("🌀 [AUTO] 시퀀스 모델 자동 재학습 + 백테스트 사이클 시작")

    python_exe = sys.executable  # 현재 파이썬 경로 그대로 사용

    # 1) ML 학습 샘플 생성
    run_step(
        db,
        [python_exe, "build_ml_seq_samples.py"],
        "ML 시퀀스 샘플 생성(build_ml_seq_samples.py)",
    )

    # 2) 새 모델 학습 (train_seq_model.py 안에서 model_versions + settings 업데이트)
    run_step(
        db,
        [python_exe, "train_seq_model.py"],
        "시퀀스 모델 학습(train_seq_model.py)",
    )

    # 3) 최신 모델 포함 백테스트
    run_step(
        db,
        [python_exe, "backtest_seq_model.py"],
        "백테스트 실행(backtest_seq_model.py)",
    )

    db.log("✅ [AUTO] 자동 재학습 + 백테스트 사이클 완료")
