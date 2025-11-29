"""
pipeline_cr_tcn.py

📌 CR(코인) 멀티스케일 TCN 전용 하루 파이프라인

실행 순서:
1) tcn_train_multiscale.py
   - CR 전용 멀티스케일 TCN 모델 학습
   - models/multiscale_cr_model.pth 저장

2) tcn_backtest_model.py
   - 새 엔트리 로직(make_entry_signal_coin_ms) 기반 CR 백테스트
   - backtests_cr/*.csv + 콘솔 요약 출력

3) 오늘 날짜 기준 CR 트레이드만 AI 리포트 생성
   - 일일 트레이드 리포트(v2)
   - 전략 아이디어 브레인스토밍
   - reports/ 폴더에 저장 + ai_reports 테이블에 region='CR_TCN'으로 저장
"""

import os
import subprocess
from datetime import date, datetime

from db_manager import BotDatabase
from ai_helpers import (
    make_daily_trade_report_v2,
    brainstorm_strategy_ideas,
)
from ai_report_context import (
    DB_PATH,
    load_trades_for_date,
    build_daily_context_v2,
    build_brainstorm_context,
)


# -------------------------------------------------------
# 공통 로그/실행 헬퍼
# -------------------------------------------------------
def log(db: BotDatabase, message: str):
    """DB와 콘솔에 동시에 로그 남기기."""
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{ts}] {message}"
    print(msg)
    if hasattr(db, "log"):
        try:
            db.log(msg)
        except Exception:
            # DB 로그 실패해도 파이프라인은 계속 진행
            pass


def run_script(script_name: str, description: str, db: BotDatabase) -> bool:
    """
    개별 파이썬 스크립트를 서브프로세스로 실행하는 헬퍼.
    - script_name: 예) 'tcn_train_multiscale.py'
    - description: 사람 눈에 보이는 설명용 텍스트
    반환값: 성공(True) / 실패(False)
    """
    if not os.path.exists(script_name):
        log(db, f"❌ [{description}] 실패 - 스크립트 파일 없음: {script_name}")
        return False

    log(db, f"🚀 [{description}] 시작 ({script_name})")

    result = subprocess.run(
        ["python", script_name],
        capture_output=True,
        text=True,
    )

    if result.stdout:
        print("\n[STDOUT]", "-" * 60)
        print(result.stdout)
    if result.stderr:
        print("\n[STDERR]", "-" * 60)
        print(result.stderr)

    if result.returncode != 0:
        log(db, f"❌ [{description}] 실패 (returncode={result.returncode})")
        return False

    log(db, f"✅ [{description}] 완료")
    return True


# -------------------------------------------------------
# 메인 파이프라인
# -------------------------------------------------------
if __name__ == "__main__":
    # 0) 공통 준비
    db = BotDatabase(DB_PATH)
    today = date.today()
    date_str = today.strftime("%Y-%m-%d")

    log(db, "=== 📌 CR TCN 일일 파이프라인 시작 ===")

    # 1) CR 멀티스케일 TCN 모델 학습
    ok_train = run_script(
        "tcn_train_models.py",
        "CR 멀티스케일 TCN 모델 학습",
        db,
    )

    # 2) 새 모델 기반 CR 백테스트
    ok_bt = run_script(
        "tcn_backtest.py",
        "CR 멀티스케일 TCN 엔트리 백테스트",
        db,
    )

    # 3) 오늘 날짜 기준 CR 트레이드 AI 리포트
    #    (실제 실전 운용에서 저장된 trades 테이블 기준)
    log(db, "📊 오늘자 CR 트레이드 로딩 중...")

    df_trades_cr = load_trades_for_date(today, region="CR")

    # 없더라도 build_daily_context_v2 / make_daily_trade_report_v2 가 안전하게 처리해줌
    daily_ctx_cr = build_daily_context_v2(
        df_trades_cr,
        today,
        region="CR",
    )
    daily_report_cr = make_daily_trade_report_v2(daily_ctx_cr, market="CR")

    brainstorm_ctx_cr = build_brainstorm_context(
        df_trades_cr,
        today,
        region="CR",
    )
    strategy_ideas_cr = brainstorm_strategy_ideas(brainstorm_ctx_cr, market="CR")

    # reports/ 저장
    os.makedirs("reports", exist_ok=True)

    daily_path = os.path.join("reports", f"{date_str}_daily_report_cr_tcn.txt")
    ideas_path = os.path.join("reports", f"{date_str}_strategy_ideas_cr_tcn.txt")

    with open(daily_path, "w", encoding="utf-8") as f:
        f.write(daily_report_cr)
    with open(ideas_path, "w", encoding="utf-8") as f:
        f.write(strategy_ideas_cr)

    log(db, f"✅ CR TCN 일일 리포트 저장: {daily_path}")
    log(db, f"✅ CR TCN 전략 아이디어 저장: {ideas_path}")

    # ai_reports 테이블에 저장 (region 태그를 'CR_TCN'으로 명시)
    try:
        db.save_ai_report(
            date_str=date_str,
            daily_report=daily_report_cr,
            strategy_ideas=strategy_ideas_cr,
            region="CR_TCN",
        )
        log(db, "✅ ai_reports 테이블에 CR_TCN 리포트 저장 완료")
    except Exception as e:
        log(db, f"⚠️ ai_reports 저장 중 오류: {e}")

    # 4) 전체 결과 요약
    if ok_train and ok_bt:
        log(db, "🎉 CR TCN 파이프라인 성공적으로 완료")
    else:
        log(db, "⚠️ 일부 단계에서 실패 발생 - 위 로그를 확인하세요.")

    log(db, "=== ✅ CR TCN 일일 파이프라인 종료 ===")
