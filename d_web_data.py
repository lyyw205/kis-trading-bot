# d_web_data.py
import psycopg2
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# [DB 설정]
DB_HOST = "aws-1-ap-northeast-2.pooler.supabase.com"
DB_NAME = "postgres"
DB_USER = "postgres.sxhtnkxulfrqykrtwxjx"
DB_PASS = "Shitdog205!@"
DB_PORT = "6543"

def get_connection():
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT,
        sslmode='require'
    )

def load_trades() -> pd.DataFrame:
    """positions 테이블 로드 (DDL 반영)"""
    conn = get_connection()
    try:
        # [수정] DDL에 맞는 컬럼 조회
        query = """
        SELECT 
            id, region, symbol, trade_type, entry_qty, 
            entry_price, entry_time,
            exit_time, exit_price,
            status, pnl_pct
        FROM positions 
        ORDER BY entry_time DESC
        """
        df = pd.read_sql_query(query, conn)
        
        if not df.empty:
            df["entry_time"] = pd.to_datetime(df["entry_time"], errors='coerce')
            df["exit_time"] = pd.to_datetime(df["exit_time"], errors='coerce')
            df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors='coerce').fillna(0.0)
            
            # 분석 호환성을 위한 매핑
            df["time"] = df["entry_time"]
            df["profit"] = df["pnl_pct"]
            
        return df
    except Exception as e:
        print(f"!!! [ERROR] load_trades: {e}")
        return pd.DataFrame()
    finally:
        conn.close()

def load_logs() -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM logs ORDER BY time DESC", conn)
        if not df.empty and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

def load_signals(limit: int = 200) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(f"SELECT * FROM signals ORDER BY time DESC LIMIT {int(limit)}", conn)
        if not df.empty and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()
        
def load_model_versions(limit: int = 20) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(f"SELECT * FROM models ORDER BY created_at DESC LIMIT {int(limit)}", conn)
        if not df.empty and "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"], errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()
        
def load_backtests(limit: int = 50) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(f"SELECT * FROM backtests ORDER BY id DESC LIMIT {int(limit)}", conn)
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

def load_signals_by_date(target_date: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM signals WHERE time::date = %s ORDER BY time", conn, params=(target_date,))
        if not df.empty and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors='coerce')
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

def load_trades_by_date(target_date: str) -> pd.DataFrame:
    """positions 테이블에서 특정 날짜 조회 (DDL 반영)"""
    conn = get_connection()
    try:
        query = """
            SELECT 
                id, region, symbol, trade_type, entry_qty,
                entry_price, entry_time,
                exit_time, status, pnl_pct
            FROM positions
            WHERE entry_time::date = %s
            ORDER BY entry_time
        """
        df = pd.read_sql_query(query, conn, params=(target_date,))
        if not df.empty:
            df["entry_time"] = pd.to_datetime(df["entry_time"], errors='coerce')
            df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors='coerce').fillna(0.0)
            
            # 매핑
            df["time"] = df["entry_time"]
            df["profit"] = df["pnl_pct"]
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

def load_ml_signals(limit: int = 500) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql(f"SELECT * FROM signals WHERE ml_proba IS NOT NULL ORDER BY id DESC LIMIT {int(limit)}", conn)
        if not df.empty and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"], errors='coerce')
            return df.sort_values("time")
        return df
    except Exception:
        return pd.DataFrame()
    finally:
        conn.close()

def suggest_improvements(df_sig: pd.DataFrame, df_tr: pd.DataFrame, ml_threshold: float = 0.55):
    suggestions = []
    if df_sig.empty:
        suggestions.append("📉 오늘 저장된 신호가 없습니다.")
        return suggestions
        
    rule_signals = int(df_sig["entry_signal"].fillna(0).sum())
    if rule_signals == 0:
        suggestions.append("⚠️ 룰 기반 시그널이 0건입니다.")
        
    if "ml_proba" in df_sig.columns and df_sig["ml_proba"].notna().any():
        hi_ratio = (df_sig["ml_proba"] >= ml_threshold).mean()
        if hi_ratio < 0.05:
            suggestions.append(f"⚠️ ML 점수 {ml_threshold} 이상 비율이 낮습니다({hi_ratio:.1%}).")
            
    if not df_tr.empty and "profit" in df_tr.columns:
        realized = df_tr["profit"].dropna()
        realized = realized[realized != 0]
        if len(realized) > 0:
            wins = (realized > 0).sum()
            win_rate = wins / len(realized)
            suggestions.append(f"💰 오늘 트레이드: {len(realized)}건, 승률 {win_rate*100:.1f}%")

    if not suggestions:
        suggestions.append("✅ 특이사항 없습니다.")
        
    return suggestions

def build_round_trades(df_positions: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[dict]]]:
    """Positions 테이블 데이터를 웹 UI 형식으로 변환"""
    if df_positions is None or df_positions.empty:
        return pd.DataFrame(), {}
        
    rows = []
    details_map = {} 
    
    for _, row in df_positions.iterrows():
        # DDL 컬럼 사용
        entry_time = row.get("entry_time")
        exit_time = row.get("exit_time")
        status = row.get("status") # 'OPEN' / 'CLOSED'
        
        rows.append({
            "symbol": row.get("symbol"),
            "round_id": row.get("id"),
            "status": status,
            "entry_time": entry_time,
            "exit_time": exit_time,
            "entry_qty": float(row.get("entry_qty") or 0),
            "entry_price": float(row.get("entry_price") or 0),
            "realized_profit_pct": float(row.get("pnl_pct") or 0),
            "entry_comment": None, # 필요한 경우 쿼리에 추가 필요
            "exit_comment": None,
            "date": entry_time.strftime("%Y-%m-%d") if pd.notna(entry_time) else ""
        })
        # 세부 내용은 Positions 테이블 하나이므로 빈 리스트 (필요하면 가공 가능)
        details_map[f"{row.get('symbol')}__{row.get('id')}"] = []
        
    return pd.DataFrame(rows), details_map
    
def get_symbols_with_data(trades: pd.DataFrame) -> List[str]:
    if not trades.empty and "symbol" in trades.columns:
        return sorted(trades["symbol"].unique().tolist())
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT DISTINCT symbol FROM ohlcv_data ORDER BY symbol", conn)
        return df["symbol"].tolist()
    except Exception:
        return []
    finally:
        conn.close()