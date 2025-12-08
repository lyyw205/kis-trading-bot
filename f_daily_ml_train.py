# run_daily_ml_cycle.py
"""
매일 장 마감 후 실행하는 ML 사이클:

1) 새 모델 학습 (train_seq_model.py)
2) 새 모델 백테스트 (db_backtest.py)
3) DB(model / backtests / settings / trades)를 바탕으로
   active vs candidate 모델 + live 성과 context 생성
4) AI에게 모델 교체/튜닝 조언 요청 (make_model_update_advice)
5) 결과를 파일(reports/*_model_advice.txt) + logs에 저장
"""

import os
import subprocess
from datetime import datetime, date

from c_db_manager import BotDatabase
from ai_helpers import make_model_update_advice
from ai_report_context import load_model_context_for_ai   # 🔹 새 파일에서 import


# -----------------------------
# 1. 학습/백테스트 스텝
# -----------------------------
def run_training_script() -> bool:
    """
    실제 트레이닝 스크립트를 호출하는 자리.
    예시: 'python ml_train_seq_model.py'
    """
    print("새 모델 학습 시작...")
    result = subprocess.run(
        ["python", "ml_train_cr.py"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("[ML] 학습 스크립트 오류:", result.stderr)
    return result.returncode == 0


def run_backtest_script() -> bool:
    """
    새로 학습한 모델에 대해 백테스트를 돌리는 자리.
    예시: 'python db_backtest.py'
    """
    print("새 모델 백테스트 시작...")
    result = subprocess.run(
        ["python", "db_backtest_cr.py"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("[ML] 백테스트 스크립트 오류:", result.stderr)
    return result.returncode == 0


# -----------------------------
# 2. 메인 실행부
# -----------------------------
if __name__ == "__main__":
    from datetime import date

    target_date = date.today()
    db = BotDatabase("trading.db")

    db.log("run_daily_ml_train 시작")

    # 1) 새 모델 학습
    ok_train = run_training_script()
    if not ok_train:
        db.log("run_daily_ml_train: 학습 실패, 이후 스텝 중단")
        exit(1)

    # 2) 새 모델 백테스트
    ok_bt = run_backtest_script()
    if not ok_bt:
        db.log("db_backtest.cr.py: 백테스트 실패, 이후 스텝 중단")
        exit(1)

    # 3) active vs candidate & live 성과 context 생성
    model_ctx = load_model_context_for_ai(db, target_date)

    # 4) AI에게 모델 교체/튜닝 조언 요청
    advice = make_model_update_advice(model_ctx)

    print("\n========================")
    print("[ML] 모델 업데이트/튜닝 조언")
    print("========================\n")
    print(advice)

    # 5) 파일이나 로그에 저장
    os.makedirs("reports", exist_ok=True)
    fname = f"reports/{target_date.strftime('%Y-%m-%d')}_model_advice.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(advice)

    db.log(f"run_daily_ml_train 완료, 모델 조언 저장: {fname}")
