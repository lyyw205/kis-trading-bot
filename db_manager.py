#프로젝트 전체에서 사용하는 모든 핵심 테이블을 생성함:

# trades
# → 실시간 체결 기록 저장
# → 대시보드 성과 / EQ curve에 직접 연결

# logs
# → 모든 실행 로그 저장
# → 대시보드 로그 패널에 표시됨

# signals
# → 룰 기반 + ML 기반 진입 신호 저장
# → 개선안(suggest_improvements)에서 사용됨

# ohlcv_data
# → UNIVERSE 과거데이터 저장
# → build_ohlcv_history.py에서 입력
# → 백테스트 / ML 샘플 생성의 핵심 원천 데이터

# models / model_versions
# → ML 모델 버전 관리
# → active model, 실험 모델 비교 등에서 사용

# backtests
# → 자동·수동 백테스트 결과 저장
# → 대시보드 Backtest Summary에 표시

# settings
# → active_model_path, ml_threshold 등 저장
# → trader.py → 모델 실행 시 사용

import sqlite3
from datetime import datetime
import json

class BotDatabase:
    def __init__(self, db_name="trading.db"):
        self.db_name = db_name
        self.init_db()
        
    def init_db(self):
        conn = sqlite3.connect(self.db_name)
        c = conn.cursor()

        # 1) 매매 내역
        c.execute('''CREATE TABLE IF NOT EXISTS trades
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     time TEXT,
                     region TEXT,
                     symbol TEXT,
                     type TEXT,
                     price REAL,
                     qty INTEGER,
                     profit REAL,
                     signal_id INTEGER,
                     ml_proba REAL,
                     entry_allowed INTEGER,
                     order_no TEXT,
                     source TEXT,
                     entry_comment TEXT,
                     exit_comment TEXT)'''
                  )
        

        # 2) 로그
        c.execute('''CREATE TABLE IF NOT EXISTS logs
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      time TEXT,
                      message TEXT)''')

        # 3) 신호 / 상태 스냅샷
        c.execute('''CREATE TABLE IF NOT EXISTS signals
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      time TEXT,
                      region TEXT,
                      symbol TEXT,
                      price REAL,
                      at_support INTEGER,
                      is_bullish INTEGER,
                      price_up INTEGER,
                      lookback INTEGER,
                      band_pct REAL,
                      has_stock INTEGER,
                      entry_signal INTEGER,
                      ml_proba REAL,
                      entry_allowed INTEGER,
                      note TEXT)''')
        
        # 4) OHLCV 저장 테이블
        c.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region TEXT,
                symbol TEXT,
                interval TEXT,    
                dt TEXT,          
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                UNIQUE(region, symbol, interval, dt) 
            )
        """)

        # 5) 간단한 모델 버전 기록 (구버전 호환용)
        c.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT,
                path TEXT,
                n_samples INTEGER,
                val_accuracy REAL
            )
        """)

        # 6) 상세 모델 메타 정보
        c.execute('''CREATE TABLE IF NOT EXISTS model_versions
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      region TEXT, 
                      name TEXT,
                      version TEXT,
                      created_at TEXT,
                      params_json TEXT,
                      note TEXT)''')

        # 7) 백테스트 결과 요약
        c.execute('''CREATE TABLE IF NOT EXISTS backtests
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      region TEXT,
                      model_id INTEGER,
                      start_date TEXT,
                      end_date TEXT,
                      trades INTEGER,
                      win_rate REAL,
                      avg_profit REAL,
                      cum_return REAL,
                      max_dd REAL,
                      note TEXT,
                      FOREIGN KEY(model_id) REFERENCES model_versions(id))''')

        # 8) settings (active_model_path, ml_threshold 등)
        c.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        # 9) Universe 백필 실패 기록 테이블 (대시보드에서 사용)
        c.execute("""
            CREATE TABLE IF NOT EXISTS universe_backfill_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region TEXT,
                symbol TEXT,
                excd TEXT,
                interval TEXT,
                error_type TEXT,
                error_message TEXT,
                created_at TEXT
            )
        """)

        # 10) AI 일일 리포트 저장 (텍스트)
        c.execute("""
            CREATE TABLE IF NOT EXISTS ai_reports (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region TEXT,
                date TEXT,                 -- "YYYY-MM-DD"
                created_at TEXT,           -- 리포트 생성 시각
                daily_report TEXT,         -- 일일 매매 리포트 전문
                strategy_ideas TEXT        -- 전략 브레인스토밍 결과
            )
        """)

        conn.commit()
        conn.close()
        
    # -----------------------------
    # OHLCV 저장 관련
    # -----------------------------
    def save_ohlcv_batch(self, rows):
        """
        rows: [
            (region, symbol, interval, dt, open, high, low, close, volume),
            ...
        ]
        """
        if not rows:
            return
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            c.executemany("""
                INSERT OR IGNORE INTO ohlcv_data
                (region, symbol, interval, dt, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, rows)
            conn.commit()
            conn.close()
        except Exception as e:
            self.log(f"⚠️ save_ohlcv_batch 실패: {e}")

    def save_ohlcv_df(self, region, symbol, interval, df):
        """
        df: index = datetime, columns = ['open','high','low','close','volume']
        """
        if df is None or df.empty:
            return

        rows = []
        for idx, row in df.iterrows():
            dt_str = idx.strftime("%Y-%m-%d %H:%M:%S")
            rows.append((
                region,
                symbol,
                interval,
                dt_str,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
            ))

        self.save_ohlcv_batch(rows)

    # -----------------------------
    # 로그
    # -----------------------------
    def log(self, message):
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute("INSERT INTO logs (time, message) VALUES (?, ?)", (now, message))
            conn.commit()
            conn.close()
            print(f"[{now}] {message}")
        except Exception as e:
            # 로그조차 실패하면 어쩔 수 없이 print만 시도
            try:
                print(f"[LOG-FAIL] {message} (원인: {e})")
            except:
                pass

    # -----------------------------
    # 트레이드 / 시그널
    # -----------------------------
    def save_trade(
        self,
        region,
        symbol,
        trade_type,
        price,
        qty,
        profit,
        signal_id=None,
        ml_proba=None,
        entry_allowed=None,
        extra=None,
        trade_time=None,
    ):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()

        order_no = None
        source = None
        if extra:
            order_no = extra.get("order_no")
            source = extra.get("source")

        if trade_time is None:
            time_value = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        else:
            time_value = trade_time

        cur.execute(
            """
            INSERT INTO trades
                (time, region, symbol, type, price, qty,
                profit, signal_id, ml_proba, entry_allowed,
                order_no, source, entry_comment, exit_comment)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)
            """,
            (
                time_value,
                region,
                symbol,
                trade_type,
                float(price),
                int(qty),
                float(profit),
                signal_id,
                None if ml_proba is None else float(ml_proba),
                None if entry_allowed is None else int(entry_allowed),
                order_no,
                source,
            ),
        )

        trade_id = cur.lastrowid
        conn.commit()
        conn.close()
        return trade_id

    def save_signal(self, *, region, symbol, price,
                    at_support, is_bullish, price_up,
                    lookback, band_pct, has_stock,
                    entry_signal, ml_proba=None,
                    entry_allowed=None,
                    note=""):
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute(
                """INSERT INTO signals
                   (time, region, symbol, price,
                    at_support, is_bullish, price_up,
                    lookback, band_pct, has_stock,
                    entry_signal, ml_proba, entry_allowed, note)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    now, region, symbol, price,
                    int(at_support), int(is_bullish), int(price_up),
                    lookback, band_pct, int(has_stock),
                    int(entry_signal),
                    None if ml_proba is None else float(ml_proba),
                    None if entry_allowed is None else int(entry_allowed),
                    note
                )
            )
            signal_id = c.lastrowid
            conn.commit()
            conn.close()
            return signal_id
        except Exception as e:
            self.log(f"⚠️ save_signal 실패: {e}")
            return None

    # -----------------------------
    # Universe 백필 실패 기록 헬퍼
    # -----------------------------
    def log_universe_backfill_failure(self, region, symbol, excd, interval,
                                      error_type, error_message):
        """
        build_ohlcv_history.py 등에서 백필 실패 시 호출.
        대시보드 universe_failures 패널에서 보여줄 데이터.
        """
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute(
                """INSERT INTO universe_backfill_failures
                   (region, symbol, excd, interval, error_type, error_message, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (region, symbol, excd, interval, error_type, error_message, now)
            )
            conn.commit()
            conn.close()
            self.log(
                f"⚠️ [UNIVERSE-FAIL] {region} {symbol} ({interval}, {excd}) "
                f"{error_type}: {error_message}"
            )
        except Exception as e:
            self.log(f"⚠️ universe_backfill_failures 기록 실패: {e}")

    # -----------------------------
    # 모델 버전(상세 메타) 저장
    # -----------------------------
    def save_model_version(self, region, name, version, params: dict, note=""):
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            c.execute(
                """INSERT INTO model_versions
                (region, name, version, created_at, params_json, note)
                VALUES (?, ?, ?, ?, ?, ?)""",
                (region, name, version, now, json.dumps(params), note)
            )
            model_id = c.lastrowid
            conn.commit()
            conn.close()
            return model_id
        except Exception as e:
            self.log(f"⚠️ save_model_version 실패: {e}")
            return None

    # -----------------------------
    # settings (active_model_path, ml_threshold 등)
    # -----------------------------
    def get_setting(self, key, default=None):
        try:
            conn = sqlite3.connect(self.db_name)
            cur = conn.cursor()
            cur.execute("SELECT value FROM settings WHERE key = ?", (key,))
            row = cur.fetchone()
            conn.close()
            return row[0] if row else default
        except Exception as e:
            self.log(f"⚠️ get_setting 실패({key}): {e}")
            return default

    def set_setting(self, key, value):
        try:
            conn = sqlite3.connect(self.db_name)
            cur = conn.cursor()
            cur.execute("""
                INSERT INTO settings (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """, (key, str(value)))
            conn.commit()
            conn.close()
        except Exception as e:
            self.log(f"⚠️ set_setting 실패({key}): {e}")

    # -----------------------------
    # 백테스트 결과 저장
    # -----------------------------
    def save_backtest(self, region, model_id, start_date, end_date,
                      trades, win_rate, avg_profit,
                      cum_return, max_dd, note=""):
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            c.execute(
                """INSERT INTO backtests
                   (region, model_id, start_date, end_date,
                    trades, win_rate, avg_profit,
                    cum_return, max_dd, note)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (region, model_id, start_date, end_date,
                 trades, win_rate, avg_profit,
                 cum_return, max_dd, note)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.log(f"⚠️ save_backtest 실패: {e}")

    # -----------------------------
    # ai 코멘트
    # -----------------------------
    def update_trade_entry_comment(self, trade_id: int, comment: str):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE trades
            SET entry_comment = ?
            WHERE id = ?
            """,
            (comment, trade_id),
        )
        conn.commit()
        conn.close()  

    def update_trade_exit_comment(self, trade_id: int, comment: str):
        conn = sqlite3.connect(self.db_name)
        cur = conn.cursor()
        cur.execute(
            """
            UPDATE trades
            SET exit_comment = ?
            WHERE id = ?
            """,
            (comment, trade_id),
        )
        conn.commit()
        conn.close()

    # -----------------------------
    # AI 리포트 저장 / 조회
    # -----------------------------
    def save_ai_report(self, date_str: str, daily_report: str, strategy_ideas: str):
        """
        date_str: "YYYY-MM-DD"
        """
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            c.execute(
                """
                INSERT INTO ai_reports (date, created_at, daily_report, strategy_ideas)
                VALUES (?, ?, ?, ?)
                """,
                (date_str, now, daily_report, strategy_ideas),
            )
            conn.commit()
            conn.close()
            self.log(f"🧠 AI 리포트 저장 완료: {date_str}")
        except Exception as e:
            self.log(f"⚠️ save_ai_report 실패: {e}")

    def get_latest_ai_report(self):
        """
        가장 최근 생성된 AI 리포트 1개 반환.
        없으면 None 리턴.
        """
        try:
            conn = sqlite3.connect(self.db_name)
            c = conn.cursor()
            c.execute(
                """
                SELECT date, created_at, daily_report, strategy_ideas
                FROM ai_reports
                ORDER BY date DESC, id DESC
                LIMIT 1
                """
            )
            row = c.fetchone()
            conn.close()
            if not row:
                return None
            return {
                "date": row[0],
                "created_at": row[1],
                "daily_report": row[2],
                "strategy_ideas": row[3],
            }
        except Exception as e:
            self.log(f"⚠️ get_latest_ai_report 실패: {e}")
            return None
