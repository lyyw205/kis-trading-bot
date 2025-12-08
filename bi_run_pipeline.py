"""
bi_run_pipeline.py
"""
import os
import sys
import subprocess
from datetime import date, datetime
from c_db_manager import BotDatabase
from ai_helpers import make_daily_trade_report_v2, brainstorm_strategy_ideas
from ai_report_context import load_trades_for_date, build_daily_context_v2, build_brainstorm_context

# ==========================================
# 🎛️ 파이프라인 실행 스위치
# True: 실행함 / False: 건너뜀 (수동으로 했을 경우 False로)
# ==========================================
RUN_TRAIN = True       # 학습 단계 (이미 했으면 False)
RUN_BACKTEST = True    # 백테스트 단계 (이미 했으면 False)
RUN_REPORT = True       # 리포트 생성 단계
# ==========================================

# [중요] 윈도우 콘솔 출력 인코딩을 UTF-8로 강제 설정 (이모지 에러 방지)
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def log(db: BotDatabase, message: str):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # 화면 출력 시 인코딩 에러가 나면 안전하게 처리
    try:
        print(f"[{ts}] {message}")
    except UnicodeEncodeError:
        print(f"[{ts}] {message.encode('utf-8', 'ignore').decode('utf-8')}")
        
    if hasattr(db, "log"):
        try:
            db.log(f"[{ts}] {message}")
        except: pass

def run_script(script_name: str, description: str, db: BotDatabase) -> bool:
    if not os.path.exists(script_name):
        log(db, f"❌ [{description}] 실패 - 파일 없음: {script_name}")
        return False
    
    log(db, f"🚀 [{description}] 시작 ({script_name})")

    # [핵심] 서브 프로세스에 UTF-8 환경변수 주입
    my_env = os.environ.copy()
    my_env["PYTHONIOENCODING"] = "utf-8"

    try:
        proc = subprocess.Popen(
            ["python", script_name],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            encoding="utf-8",    # 읽을 때도 utf-8
            errors="replace",    # 깨진 글자는 대체
            env=my_env           # 환경변수 적용
        )
        
        if proc.stdout:
            for line in proc.stdout:
                try:
                    print(f"[{script_name}] {line}", end="", flush=True)
                except UnicodeEncodeError:
                    pass 

        proc.wait()
        if proc.returncode != 0:
            log(db, f"❌ [{description}] 실패 (code={proc.returncode})")
            return False
        
        log(db, f"✅ [{description}] 완료")
        return True
        
    except Exception as e:
        log(db, f"❌ [{description}] 예외: {e}")
        return False

if __name__ == "__main__":
    db = BotDatabase()
    today = date.today()
    date_str = today.strftime("%Y-%m-%d")

    log(db, "=== 📌 CR TCN 일일 파이프라인 시작 ===")

    # 기본값은 성공(True)으로 설정 (스킵 시 에러 방지)
    ok_train = True
    ok_bt = True

    # 1) 학습 단계
    if RUN_TRAIN:
        ok_train = run_script("tcn_train_models.py", "CR TCN 모델 학습", db)
    else:
        log(db, "⏭️ [SKIP] 학습 단계 건너뜀 (설정: False)")

    # 2) 백테스트 단계
    if RUN_BACKTEST:
        if ok_train:
            ok_bt = run_script("tcn_backtest.py", "CR TCN 백테스트", db)
        else:
            ok_bt = False
            log(db, "⚠️ 학습 실패로 백테스트 자동 스킵")
    else:
        log(db, "⏭️ [SKIP] 백테스트 단계 건너뜀 (설정: False)")

    # 3) 리포트 단계
    if RUN_REPORT:
        log(db, "📊 오늘자 CR 트레이드 리포트 생성 시작...")
        try:
            df_trades_cr = load_trades_for_date(today, region="CR")
            
            # 리포트 본문
            daily_ctx = build_daily_context_v2(df_trades_cr, today, region="CR")
            report_txt = make_daily_trade_report_v2(daily_ctx, market="CR")
            
            # 전략 아이디어
            idea_ctx = build_brainstorm_context(df_trades_cr, today, region="CR")
            ideas_txt = brainstorm_strategy_ideas(idea_ctx, market="CR")

            # 파일 저장
            os.makedirs("reports", exist_ok=True)
            r_path = os.path.join("reports", f"{date_str}_daily_report_cr_tcn.txt")
            i_path = os.path.join("reports", f"{date_str}_strategy_ideas_cr_tcn.txt")
            
            with open(r_path, "w", encoding="utf-8") as f: f.write(report_txt)
            with open(i_path, "w", encoding="utf-8") as f: f.write(ideas_txt)

            log(db, f"✅ 파일 저장 완료: {r_path}")
            
            # DB 저장
            try:
                db.save_ai_report(date_str, report_txt, ideas_txt, region="CR_TCN")
                log(db, "✅ DB 저장 완료 (ai_reports)")
            except Exception as e:
                log(db, f"⚠️ DB 저장 실패: {e}")
            
        except Exception as e:
            log(db, f"❌ 리포트 생성 중 에러: {e}")
    
    # 4) 최종 요약
    if ok_train and ok_bt:
        log(db, "🎉 CR TCN 파이프라인 최종 완료")
    else:
        log(db, "⚠️ 파이프라인 일부 단계 실패")

    log(db, "=== ✅ 종료 ===")