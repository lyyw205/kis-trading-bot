# "PostgreSQL(Supabase) 기반 공용 DB 매니저 (BotDatabase)

#  - Supabase(Postgres)에 있는 trades / logs / signals / ohlcv_data / models / model_versions /
#    backtests / settings / universe_backfill_failures / ai_reports 를 생성·읽기·쓰기하는
#    트레이딩 시스템 공용 DB 레이어

# 주요 기능:
# 1) DB 초기화 및 연결
#    - DB_HOST/DB_NAME/DB_USER/DB_PASS/DB_PORT 상수로 Supabase 접속 정보 정의
#    - quote_plus 로 비밀번호 특수문자 인코딩 → DB_URL = postgresql://... 형태 생성
#    - BotDatabase(db_url=DB_URL)
#      · init_db()에서 필요한 테이블들을 전부 CREATE TABLE IF NOT EXISTS로 생성
#      · get_connection()은 psycopg2.connect(self.db_url) 로 커넥션 반환

#    - 생성 테이블:
#      1) trades: 매매 내역 (time, region, symbol, type, price, qty, profit, signal_id, ml_proba, entry_allowed, order_no, source, entry_comment, exit_comment)
#      2) logs: 로그 (time, message)
#      3) signals: 엔트리/ML 신호 스냅샷
#      4) ohlcv_data: OHLCV 캔들 (region, symbol, interval, dt, open/high/low/close/volume, UNIQUE(region, symbol, interval, dt))
#      5) models: (구버전) 모델 기록
#      6) model_versions: 모델 버전 메타데이터 (region, name, version, params_json, note)
#      7) backtests: 백테스트 결과 (region, model_id FK, 기간, trades, win_rate, avg_profit, cum_return, max_dd, note)
#      8) settings: key-value 환경 설정
#      9) universe_backfill_failures: 유니버스 OHLCV 백필 실패 이력
#      10) ai_reports: 일간 AI 리포트 (region, date, daily_report, strategy_ideas)

# 2) OHLCV 저장/조회
#    - save_ohlcv_batch(rows):
#      · ohlcv_data에 bulk insert, ON CONFLICT (region,symbol,interval,dt) DO NOTHING 으로 중복 방지
#    - save_ohlcv_df(region, symbol, interval, df):
#      · DataFrame을 rows 튜플 리스트로 변환해 save_ohlcv_batch 호출
#    - get_last_ohlcv_dt(region, symbol, interval):
#      · 해당 키 조합의 MAX(dt)를 조회해 마지막 캔들 시각을 문자열로 반환

# 3) 로그/트레이드/시그널
#    - log(message):
#      · logs 테이블에 time, message insert + 콘솔 출력 (에러 시 콘솔만 출력 시도)
#    - save_trade(...):
#      · trades에 1건 INSERT 후 RETURNING id 로 trade_id 반환
#      · region / symbol / type / price / qty / profit / signal_id / ml_proba / entry_allowed / order_no / source 저장
#    - save_signal(...):
#      · signals 테이블에 1건 INSERT 후 id 반환
#      · at_support / is_bullish / price_up / has_stock / entry_signal / ml_proba / entry_allowed / note 포함

# 4) 유니버스 백필 실패 기록
#    - log_universe_backfill_failure(region, symbol, excd, interval, error_type, error_message):
#      · universe_backfill_failures에 한 줄 INSERT
#      · 동시에 log()로 "[UNIVERSE-FAIL]" 로그 출력

# 5) 모델 버전 / 백테스트 / 설정
#    - save_model_version(region, name, version, params, note):
#      · model_versions에 INSERT + id 반환
#    - save_backtest(region, model_id, start_date, end_date, trades, win_rate, avg_profit, cum_return, max_dd, note):
#      · backtests에 INSERT
#    - get_setting(key, default):
#      · settings 테이블에서 value 조회, 없으면 default
#    - set_setting(key, value):
#      · INSERT ... ON CONFLICT(key) DO UPDATE SET value=excluded.value 로 upsert

# 6) 트레이드 코멘트 업데이트
#    - update_trade_entry_comment(trade_id, comment)
#    - update_trade_exit_comment(trade_id, comment)
#    → trades 테이블의 entry_comment / exit_comment 필드 업데이트

# 7) AI 리포트 저장/조회
#    - save_ai_report(date_str, daily_report, strategy_ideas, region="ALL"):
#      · ai_reports 테이블에 INSERT
#    - get_latest_ai_report():
#      · ai_reports에서 최신 1건을 가져와 dict(date, created_at, daily_report, strategy_ideas)로 반환

# → 요약: **전체 트레이딩 시스템이 공통으로 쓰는 Postgres DB 접근/스키마/저장 헬퍼**."

import psycopg2
from psycopg2.extras import DictCursor
from datetime import datetime
import json
import os
from urllib.parse import quote_plus  # [추가] 비밀번호 특수문자 처리를 위해 필요

# -----------------------------------------------------------
# [중요] 접속 정보를 분리해서 입력 (특수문자 에러 방지)
# -----------------------------------------------------------
DB_HOST = "aws-1-ap-northeast-2.pooler.supabase.com"
DB_NAME = "postgres"
DB_USER = "postgres.sxhtnkxulfrqykrtwxjx"  # [주의] 아이디가 이렇게 길어집니다
DB_PASS = "Shitdog205!@"                     # 기존 비밀번호 그대로
DB_PORT = "6543"                             # [주의] 포트가 6543입니다


# 비밀번호를 URL 인코딩하여 안전한 연결 주소 생성
encoded_pass = quote_plus(DB_PASS)
DB_URL = f"postgresql://{DB_USER}:{encoded_pass}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
# -----------------------------------------------------------

class BotDatabase:
    def __init__(self, db_url=DB_URL):
        self.db_url = db_url
        self.init_db()
        
    def get_connection(self):
        """DB 연결 객체 생성"""
        return psycopg2.connect(self.db_url)

    def init_db(self):
        conn = self.get_connection()
        c = conn.cursor()

        # PostgreSQL에서는 AUTOINCREMENT 대신 SERIAL을 사용합니다.
        
        # 1) 매매 내역
        c.execute('''CREATE TABLE IF NOT EXISTS trades
                    (id SERIAL PRIMARY KEY,
                     time TIMESTAMP,
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
                     exit_comment TEXT)''')

        # 2) 로그
        c.execute('''CREATE TABLE IF NOT EXISTS logs
                     (id SERIAL PRIMARY KEY,
                      time TIMESTAMP,
                      message TEXT)''')

        # 3) 신호 / 상태 스냅샷
        c.execute('''CREATE TABLE IF NOT EXISTS signals
                     (id SERIAL PRIMARY KEY,
                      time TIMESTAMP,
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
        
        # 4) OHLCV 저장 테이블 (UNIQUE 제약조건 명시 필요)
        c.execute("""
            CREATE TABLE IF NOT EXISTS ohlcv_data (
                id SERIAL PRIMARY KEY,
                region TEXT,
                symbol TEXT,
                interval TEXT,    
                dt TIMESTAMP,          
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                volume REAL,
                UNIQUE(region, symbol, interval, dt) 
            )
        """)

        # 5) 모델 (구버전)
        c.execute("""
            CREATE TABLE IF NOT EXISTS models (
                id SERIAL PRIMARY KEY,
                created_at TIMESTAMP,
                path TEXT,
                n_samples INTEGER,
                val_accuracy REAL
            )
        """)

        # 6) 모델 버전
        c.execute('''CREATE TABLE IF NOT EXISTS model_versions
                     (id SERIAL PRIMARY KEY,
                      region TEXT, 
                      name TEXT,
                      version TEXT,
                      created_at TIMESTAMP,
                      params_json TEXT,
                      note TEXT)''')

        # 7) 백테스트 결과
        c.execute('''CREATE TABLE IF NOT EXISTS backtests
                     (id SERIAL PRIMARY KEY,
                      region TEXT,
                      model_id INTEGER,
                      start_date TIMESTAMP,
                      end_date TIMESTAMP,
                      trades INTEGER,
                      win_rate REAL,
                      avg_profit REAL,
                      cum_return REAL,
                      max_dd REAL,
                      note TEXT,
                      FOREIGN KEY(model_id) REFERENCES model_versions(id))''')

        # 8) settings
        c.execute("""
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """)

        # 9) Universe 백필 실패
        c.execute("""
            CREATE TABLE IF NOT EXISTS universe_backfill_failures (
                id SERIAL PRIMARY KEY,
                region TEXT,
                symbol TEXT,
                excd TEXT,
                interval TEXT,
                error_type TEXT,
                error_message TEXT,
                created_at TIMESTAMP
            )
        """)

        # 10) AI 리포트
        c.execute("""
            CREATE TABLE IF NOT EXISTS ai_reports (
                id SERIAL PRIMARY KEY,
                region TEXT DEFAULT 'ALL',
                date TEXT,
                created_at TIMESTAMP DEFAULT NOW(),
                daily_report TEXT,
                strategy_ideas TEXT
            )
        """)

        conn.commit()
        conn.close()
        
    # -----------------------------
    # OHLCV 저장 관련
    # -----------------------------
    def save_ohlcv_batch(self, rows):
        if not rows:
            return
        try:
            conn = self.get_connection()
            c = conn.cursor()
            # PostgreSQL: INSERT OR IGNORE -> ON CONFLICT DO NOTHING
            # %s 사용 (sqlite의 ?)
            query = """
                INSERT INTO ohlcv_data
                (region, symbol, interval, dt, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (region, symbol, interval, dt) DO NOTHING
            """
            c.executemany(query, rows)
            conn.commit()
            conn.close()
        except Exception as e:
            self.log(f"⚠️ save_ohlcv_batch 실패: {e}")

    def save_ohlcv_df(self, region, symbol, interval, df):
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

    def get_last_ohlcv_dt(self, region: str, symbol: str, interval: str):
            conn = self.get_connection()
            cur = conn.cursor()
            try:
                cur.execute(
                    """
                    SELECT MAX(dt)
                    FROM ohlcv_data
                    WHERE region=%s AND symbol=%s AND interval=%s
                    """,
                    (region, symbol, interval),
                )
                row = cur.fetchone()
                if row is None or row[0] is None:
                    return None
                # Postgres timestamp -> string 변환
                return str(row[0]) 
            except Exception as e:
                self.log(f"⚠️ get_last_ohlcv_dt 읽기 실패: {e}")
                return None
            finally:
                conn.close()
    
    # -----------------------------
    # 로그
    # -----------------------------
    def log(self, message):
        try:
            conn = self.get_connection()
            c = conn.cursor()
            now = datetime.now() # datetime 객체 그대로 넣어도 됨
            c.execute("INSERT INTO logs (time, message) VALUES (%s, %s)", (now, message))
            conn.commit()
            conn.close()
            print(f"[{now}] {message}")
        except Exception as e:
            try:
                print(f"[LOG-FAIL] {message} (원인: {e})")
            except:
                pass

    # -----------------------------
    # 트레이드 / 시그널
    # -----------------------------
    def save_trade(self, region, symbol, trade_type, price, qty, profit, 
                   signal_id=None, ml_proba=None, entry_allowed=None, extra=None, trade_time=None):
        conn = self.get_connection()
        cur = conn.cursor()

        order_no = None
        source = None
        if extra:
            order_no = extra.get("order_no")
            source = extra.get("source")

        time_value = trade_time if trade_time else datetime.now()

        # Postgres에서 ID를 받으려면 RETURNING id 필요
        cur.execute(
            """
            INSERT INTO trades
                (time, region, symbol, type, price, qty,
                profit, signal_id, ml_proba, entry_allowed,
                order_no, source, entry_comment, exit_comment)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, NULL, NULL)
            RETURNING id
            """,
            (
                time_value, region, symbol, trade_type, float(price), int(qty),
                float(profit), signal_id,
                None if ml_proba is None else float(ml_proba),
                None if entry_allowed is None else int(entry_allowed),
                order_no, source,
            ),
        )

        trade_id = cur.fetchone()[0]
        conn.commit()
        conn.close()
        return trade_id

    def save_signal(self, *, region, symbol, price, at_support, is_bullish, price_up,
                    lookback, band_pct, has_stock, entry_signal, ml_proba=None,
                    entry_allowed=None, note=""):
        try:
            conn = self.get_connection()
            c = conn.cursor()
            now = datetime.now()
            c.execute(
                """INSERT INTO signals
                   (time, region, symbol, price,
                    at_support, is_bullish, price_up,
                    lookback, band_pct, has_stock,
                    entry_signal, ml_proba, entry_allowed, note)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING id""",
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
            signal_id = c.fetchone()[0]
            conn.commit()
            conn.close()
            return signal_id
        except Exception as e:
            self.log(f"⚠️ save_signal 실패: {e}")
            return None

    # -----------------------------
    # Universe 백필 실패 기록 헬퍼
    # -----------------------------
    def log_universe_backfill_failure(self, region, symbol, excd, interval, error_type, error_message):
        try:
            conn = self.get_connection()
            c = conn.cursor()
            now = datetime.now()
            c.execute(
                """INSERT INTO universe_backfill_failures
                   (region, symbol, excd, interval, error_type, error_message, created_at)
                   VALUES (%s, %s, %s, %s, %s, %s, %s)""",
                (region, symbol, excd, interval, error_type, error_message, now)
            )
            conn.commit()
            conn.close()
            self.log(f"⚠️ [UNIVERSE-FAIL] {region} {symbol} ({interval}, {excd}) {error_type}: {error_message}")
        except Exception as e:
            self.log(f"⚠️ universe_backfill_failures 기록 실패: {e}")

    # -----------------------------
    # 모델 버전 저장
    # -----------------------------
    def save_model_version(self, region, name, version, params: dict, note=""):
        try:
            conn = self.get_connection()
            c = conn.cursor()
            now = datetime.now()
            c.execute(
                """INSERT INTO model_versions
                (region, name, version, created_at, params_json, note)
                VALUES (%s, %s, %s, %s, %s, %s)
                RETURNING id""",
                (region, name, version, now, json.dumps(params), note)
            )
            model_id = c.fetchone()[0]
            conn.commit()
            conn.close()
            return model_id
        except Exception as e:
            self.log(f"⚠️ save_model_version 실패: {e}")
            return None

    # -----------------------------
    # settings
    # -----------------------------
    def get_setting(self, key, default=None):
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            cur.execute("SELECT value FROM settings WHERE key = %s", (key,))
            row = cur.fetchone()
            conn.close()
            return row[0] if row else default
        except Exception as e:
            self.log(f"⚠️ get_setting 실패({key}): {e}")
            return default

    def set_setting(self, key, value):
        try:
            conn = self.get_connection()
            cur = conn.cursor()
            # SQLite의 ON CONFLICT DO UPDATE 구문 -> Postgres도 동일하게 지원 (단, INSERT INTO ... VALUES ...)
            cur.execute("""
                INSERT INTO settings (key, value)
                VALUES (%s, %s)
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
            conn = self.get_connection()
            c = conn.cursor()
            c.execute(
                """INSERT INTO backtests
                   (region, model_id, start_date, end_date,
                    trades, win_rate, avg_profit,
                    cum_return, max_dd, note)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)""",
                (region, model_id, start_date, end_date,
                 trades, win_rate, avg_profit,
                 cum_return, max_dd, note)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            self.log(f"⚠️ save_backtest 실패: {e}")

    # -----------------------------
    # 코멘트 업데이트
    # -----------------------------
    def update_trade_entry_comment(self, trade_id: int, comment: str):
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute("UPDATE trades SET entry_comment = %s WHERE id = %s", (comment, trade_id))
        conn.commit()
        conn.close()  

    def update_trade_exit_comment(self, trade_id: int, comment: str):
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute("UPDATE trades SET exit_comment = %s WHERE id = %s", (comment, trade_id))
        conn.commit()
        conn.close()

    # -----------------------------
    # AI 리포트
    # -----------------------------
    def save_ai_report(self, date_str: str, daily_report: str, strategy_ideas: str, region: str = "ALL"):
        conn = self.get_connection()
        try:
            cur = conn.cursor()
            # Postgres는 컬럼 추가시 IF NOT EXISTS를 직접 지원하지 않으므로, 
            # 단순히 테이블이 없으면 생성하는 로직만 유지합니다. (이미 init_db에서 생성됨)
            
            cur.execute(
                """
                INSERT INTO ai_reports (date, region, daily_report, strategy_ideas)
                VALUES (%s, %s, %s, %s)
                """,
                (date_str, region, daily_report, strategy_ideas),
            )
            conn.commit()
        finally:
            conn.close()

    def get_latest_ai_report(self):
        try:
            conn = self.get_connection()
            # DictCursor를 사용하면 컬럼명으로 접근 가능하지만, 여기선 기본 커서 사용 후 매핑
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
                "date": str(row[0]),
                "created_at": str(row[1]),
                "daily_report": row[2],
                "strategy_ideas": row[3],
            }
        except Exception as e:
            self.log(f"⚠️ get_latest_ai_report 실패: {e}")
            return None