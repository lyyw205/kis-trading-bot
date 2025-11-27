# daily_ml_cycle.py
"""
매일 장 마감 후 실행하는 ML 사이클 뼈대:
1) 새 모델 학습 (train_seq_model.py 같은 것 호출)
2) 새 모델 백테스트
3) DB(model_versions, backtests, settings)에서 active vs candidate 성과 로드
4) AI에게 모델 교체/튜닝 조언 요청
5) 결과를 logs / 파일 / DB 등에 저장
"""

import os
import subprocess
import sqlite3
from datetime import datetime, date

import pandas as pd

from db import BotDatabase
from ai_helpers import make_model_update_advice

DB_PATH = "trading.db"


# -----------------------------
# 1. 학습/백테스트 스텝 (네 환경에 맞게 수정)
# -----------------------------
def run_training_script():
    """
    실제 트레이닝 스크립트를 호출하는 자리.
    예시는 'python train_seq_model.py' 로컬 호출.
    """
    print("새 모델 학습 시작...")
    result = subprocess.run(
        ["python", "train_seq_model.py"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("[ML] 학습 스크립트 오류:", result.stderr)
    return result.returncode == 0


def run_backtest_script():
    """
    새로 학습한 모델에 대해 백테스트를 돌리는 자리.
    """
    print("새 모델 백테스트 시작...")
    result = subprocess.run(
        ["python", "backtest_seq_model.py"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)
    if result.returncode != 0:
        print("[ML]백테스트 스크립트 오류:", result.stderr)
    return result.returncode == 0


# -----------------------------
# 2. DB에서 active/candidate 모델 성과 읽어오기
# -----------------------------
def load_model_context_for_ai(db: BotDatabase, target_date: date):
    """
    model_versions + backtests + settings를 조합해서
    make_model_update_advice()에 넘길 context 생성.
    """
    conn = sqlite3.connect(DB_PATH)

    # model_versions 전체
    df_models = pd.read_sql_query("SELECT * FROM model_versions ORDER BY id ASC", conn)
    # backtests 전체
    df_bt = pd.read_sql_query("SELECT * FROM backtests ORDER BY id ASC", conn)

    conn.close()

    # active 모델 ID / 이름은 settings에서 가져온다고 가정
    active_model_name = db.get_setting("active_model_name", None)
    active_model_id = db.get_setting("active_model_id", None)

    active = None
    if active_model_id:
        try:
            active_id_int = int(active_model_id)
            m = df_models[df_models["id"] == active_id_int]
            if not m.empty:
                m = m.iloc[0]
                bt = df_bt[df_bt["model_id"] == active_id_int]
                latest_bt = bt.iloc[-1] if not bt.empty else None

                active = {
                    "model_id": int(m["id"]),
                    "name": m.get("name"),
                    "version": m.get("version"),
                    "created_at": m.get("created_at"),
                    "backtest": None,
                }
                if latest_bt is not None:
                    active["backtest"] = {
                        "period": f'{latest_bt["start_date"]} ~ {latest_bt["end_date"]}',
                        "trades": int(latest_bt["trades"]),
                        "win_rate": float(latest_bt["win_rate"]),
                        "avg_profit": float(latest_bt["avg_profit"]),
                        "cum_return": float(latest_bt["cum_return"]),
                        "max_dd": float(latest_bt["max_dd"]),
                    }
        except Exception:
            pass

    # 가장 최근에 추가된 모델을 candidate로 가정 (active와 같을 수도 있음)
    candidate = None
    if not df_models.empty:
        m = df_models.iloc[-1]
        model_id = int(m["id"])
        bt = df_bt[df_bt["model_id"] == model_id]
        latest_bt = bt.iloc[-1] if not bt.empty else None

        candidate = {
            "model_id": int(m["id"]),
            "name": m.get("name"),
            "version": m.get("version"),
            "created_at": m.get("created_at"),
            "backtest": None,
        }
        if latest_bt is not None:
            candidate["backtest"] = {
                "period": f'{latest_bt["start_date"]} ~ {latest_bt["end_date"]}',
                "trades": int(latest_bt["trades"]),
                "win_rate": float(latest_bt["win_rate"]),
                "avg_profit": float(latest_bt["avg_profit"]),
                "cum_return": float(latest_bt["cum_return"]),
                "max_dd": float(latest_bt["max_dd"]),
            }

    # 최근 실 매매 성과 (예: 최근 3일치 trades 테이블 집계)
    conn = sqlite3.connect(DB_PATH)
    df_trades = pd.read_sql_query(
        """
        SELECT *
        FROM trades
        WHERE DATE(time) >= DATE(?, '-2 day')  -- 최근 3일
        """,
        conn,
        params=[target_date.strftime("%Y-%m-%d")],
    )
    conn.close()

    live_stats = {
        "recent_days": 3,
        "trades": 0,
        "win_rate": 0.0,
        "avg_profit": 0.0,
        "cum_profit": 0.0,
    }
    if not df_trades.empty:
        profit = df_trades["profit"].fillna(0)
        total = len(df_trades)
        wins = (profit > 0).sum()
        live_stats = {
            "recent_days": 3,
            "trades": int(total),
            "win_rate": float(wins / total * 100) if total > 0 else 0.0,
            "avg_profit": float(profit.mean()),
            "cum_profit": float(profit.sum()),
        }

    # settings
    settings = {
        "ml_threshold": float(db.get_setting("ml_threshold", 0.5)),
        "max_positions": int(db.get_setting("max_positions", 3) or 3),
    }

    ctx = {
        "date": target_date.strftime("%Y-%m-%d"),
        "active": active,
        "candidate": candidate,
        "live_stats": live_stats,
        "settings": settings,
    }
    return ctx


# -----------------------------
# 3. 메인 루프
# -----------------------------
if __name__ == "__main__":
    target_date = date.today()
    db = BotDatabase(DB_PATH)

    db.log("daily_ml_cycle 시작")

    # 1) 새 모델 학습
    ok_train = run_training_script()
    if not ok_train:
        db.log("daily_ml_cycle: 학습 실패, 이후 스텝 중단")
        exit(1)

    # 2) 새 모델 백테스트
    ok_bt = run_backtest_script()
    if not ok_bt:
        db.log("daily_ml_cycle: 백테스트 실패, 이후 스텝 중단")
        exit(1)

    # 3) active vs candidate & live 성과 context 생성
    model_ctx = load_model_context_for_ai(db, target_date)

    # 4) AI에게 모델 교체/튜닝 조언 요청
    advice = make_model_update_advice(model_ctx)

    print("\n========================")
    print("[ML] 모델 업데이트/튜닝 조언")
    print("========================\n")
    print(advice)

    # 5) 파일이나 로그에 저장 (원하면 별도 테이블로 빼도 됨)
    os.makedirs("reports", exist_ok=True)
    fname = f"reports/{target_date.strftime('%Y-%m-%d')}_model_advice.txt"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(advice)

    db.log(f"daily_ml_cycle 완료, 모델 조언 저장: {fname}")
