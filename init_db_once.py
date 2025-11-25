# init_db_once.py
import sqlite3
from db import BotDatabase

DB_PATH = "trading.db"

def create_additional_tables():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # ------------------------------------------------------
    # 1) 모델 버전 관리 테이블 (학습된 모델 기록)
    # ------------------------------------------------------
    c.execute("""
    CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        created_at TEXT,
        path TEXT,
        n_samples INTEGER,
        val_accuracy REAL
    )
    """)

    # ------------------------------------------------------
    # 2) 시스템 설정 테이블 (현재 활성 모델/threshold 저장)
    # ------------------------------------------------------
    c.execute("""
    CREATE TABLE IF NOT EXISTS settings (
        key TEXT PRIMARY KEY,
        value TEXT
    )
    """)

    # ------------------------------------------------------
    # 3) settings 기본 값 넣기 (없을 때만)
    # ------------------------------------------------------
    c.execute("SELECT COUNT(*) FROM settings")
    count_settings = c.fetchone()[0]

    if count_settings == 0:
        default_settings = [
            ("active_model_path", ""),   # 실전용 모델 없음
            ("ml_threshold", "0.60"),    # 기본 threshold
        ]
        c.executemany("INSERT INTO settings (key, value) VALUES (?, ?)", default_settings)
        print("🔧 settings 기본값 초기화 완료")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    # 1) BotDatabase 기본 테이블 생성
    db = BotDatabase(DB_PATH)
    print("✅ BotDatabase.init_db() 실행 완료 - 기본 테이블 생성됨")

    # 2) 우리가 추가한 확장 테이블 생성
    create_additional_tables()
    print("✅ 추가 테이블(models, settings) 생성 완료")

    print("🎉 DB 초기화 전체 완료")
