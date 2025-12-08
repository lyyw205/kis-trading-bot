"""
AI 리포트용 컨텍스트를 만드는 헬퍼 모듈.

- trades / signals / ohlcv_data 를 조합해서
  일일 리포트(v2) + 전략 브레인스토밍용 context 를 만든다.
- KR / US / COIN 등 시장별 리포트를 만들 수 있도록
  region / market 파라미터를 지원한다.
"""

import pandas as pd
from datetime import date, datetime
from c_db_manager import BotDatabase

# OHLCV 윈도우 설정 (엔트리 전/후 몇 개 캔들을 볼지)
INTERVAL = "5m"    # 기본 인터벌
PRE_BARS = 20      # 엔트리 이전 캔들 개수
POST_BARS = 20     # 엔트리 이후 캔들 개수


# -----------------------------
# 기본 로딩 함수들
# -----------------------------
def load_trades_for_date(target_date, region=None) -> pd.DataFrame:
    """
    trades 테이블에서 해당 날짜의 트레이드 로드.
    - time 컬럼 기준 (YYYY-MM-DD HH:MM:SS)
    - region 필터 (KR / US / COIN 등)
    """
    db = BotDatabase()
    conn = db.get_connection()
    date_str = target_date.strftime("%Y-%m-%d")

    try:
        if region is None:
            # Postgres 문법: time::date = %s
            query = """
                SELECT *
                FROM trades
                WHERE time::date = %s
            """
            params = [date_str]

        elif region in ("CR", "COIN"):
            # 과거 CR, 신규 COIN 둘 다 집계
            query = """
                SELECT *
                FROM trades
                WHERE time::date = %s
                  AND region IN ('CR', 'COIN')
            """
            params = [date_str]

        elif region in ("BI", "COIN"):
            # 과거 BI, 신규 COIN 둘 다 집계
            query = """
                SELECT *
                FROM trades
                WHERE time::date = %s
                  AND region IN ('BI', 'COIN')
            """
            params = [date_str]

        else:
            query = """
                SELECT *
                FROM trades
                WHERE time::date = %s
                  AND region = %s
            """
            params = [date_str, region]

        df = pd.read_sql_query(query, conn, params=params)
        
        if not df.empty and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        return df

    finally:
        conn.close()


def load_signal_for_trade(signal_id: int):
    """
    trades.signal_id -> signals.id 조인용 헬퍼.
    없으면 None 리턴.
    """
    if signal_id is None:
        return None
        
    db = BotDatabase()
    conn = db.get_connection()
    try:
        # SQLite ? -> Postgres %s
        row = pd.read_sql_query(
            "SELECT * FROM signals WHERE id = %s",
            conn,
            params=[signal_id],
        )
        if row.empty:
            return None
        return row.iloc[0].to_dict()
    finally:
        conn.close()


def load_ohlcv_window(region: str, symbol: str, entry_time: datetime):
    """
    엔트리 전/후로 PRE_BARS / POST_BARS 캔들을 가져오는 함수.
    """
    if not region or region == "UNKNOWN" or not symbol:
        return None, None

    db = BotDatabase()
    conn = db.get_connection()
    try:
        # SQLite ? -> Postgres %s
        df = pd.read_sql_query(
            """
            SELECT dt, open, high, low, close, volume
            FROM ohlcv_data
            WHERE region = %s
              AND symbol = %s
              AND interval = %s
            ORDER BY dt
            """,
            conn,
            params=[region, symbol, INTERVAL],
        )
        
        if df.empty:
            return None, None

        df["dt"] = pd.to_datetime(df["dt"])

        # entry_time과 가장 가까운 캔들을 기준으로 윈도우 슬라이스
        idx = (df["dt"] - entry_time).abs().idxmin()

        pre = df.iloc[max(0, idx - PRE_BARS): idx]
        post = df.iloc[idx: idx + POST_BARS]

        return pre, post

    finally:
        conn.close()


def summarize_pre_post_windows(entry_price: float, pre: pd.DataFrame, post: pd.DataFrame):
    """
    pre/post 캔들들로부터 간단한 지표들을 계산.
    (로직 변경 없음)
    """
    pre_summary = None
    post_summary = None

    if pre is not None and not pre.empty:
        first_close = pre["close"].iloc[0]
        last_close = pre["close"].iloc[-1]
        trend_pct = (last_close / first_close - 1.0) * 100.0 if first_close > 0 else 0.0

        vol_range = (pre["high"] - pre["low"]).abs()
        volatility_pct = (vol_range.mean() / entry_price * 100.0) if entry_price > 0 else 0.0

        pre_summary = {
            "bars": int(len(pre)),
            "trend_pct": float(trend_pct),
            "volatility_pct": float(volatility_pct),
        }

    if post is not None and not post.empty:
        max_high = float(post["high"].max())
        min_low = float(post["low"].min())
        last_close = float(post["close"].iloc[-1])

        mfe_pct = (max_high / entry_price - 1.0) * 100.0 if entry_price > 0 else 0.0
        mae_pct = (min_low / entry_price - 1.0) * 100.0 if entry_price > 0 else 0.0
        close_pct_after_n = (last_close / entry_price - 1.0) * 100.0 if entry_price > 0 else 0.0

        post_summary = {
            "bars": int(len(post)),
            "mfe_pct": float(mfe_pct),
            "mae_pct": float(mae_pct),
            "close_pct_after_n": float(close_pct_after_n),
        }

    return pre_summary, post_summary


# -----------------------------
# v2: 차트+시그널까지 포함한 context
# -----------------------------
def build_daily_context_v2(
    df_trades: pd.DataFrame,
    target_date: date,
    region: str | None = None,
) -> dict:
    """
    v2 일일 리포트용 context 생성.
    (로직 변경 없음)
    """
    date_str = target_date.strftime("%Y-%m-%d")
    market = region  # 컨텍스트 상의 시장 태그

    if df_trades.empty:
        return {
            "date": date_str,
            "region": market,
            "stats": {
                "total_trades": 0,
                "total_profit": 0.0,
                "win_rate": 0.0,
                "avg_profit": 0.0,
                "max_profit": 0.0,
                "max_loss": 0.0,
            },
            "trade_details": [],
        }

    profit = df_trades["profit"].fillna(0)
    total_trades = len(df_trades)
    wins = (profit > 0).sum()
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0.0

    stats = {
        "total_trades": int(total_trades),
        "total_profit": float(profit.sum()),
        "win_rate": round(float(win_rate), 2),
        "avg_profit": float(profit.mean()),
        "max_profit": float(profit.max()),
        "max_loss": float(profit.min()),
    }

    trade_details = []

    for _, row in df_trades.iterrows():
        trade_id = int(row["id"])
        symbol = row.get("symbol")
        trade_type = row.get("type")
        entry_time = pd.to_datetime(row["time"])
        entry_price = float(row.get("price", 0.0))
        qty = int(row.get("qty", 0) or 0)
        profit_val = float(row.get("profit", 0.0) or 0.0)

        trade_region = row.get("region") or market or "UNKNOWN"

        signal_data = None
        signal_id = row.get("signal_id")
        if signal_id is not None:
            try:
                sig_row = load_signal_for_trade(int(signal_id))
            except Exception:
                sig_row = None

            if sig_row:
                trade_region = sig_row.get("region") or trade_region
                signal_data = {
                    "region": trade_region,
                    "at_support": sig_row.get("at_support"),
                    "is_bullish": sig_row.get("is_bullish"),
                    "price_up": sig_row.get("price_up"),
                    "lookback": sig_row.get("lookback"),
                    "band_pct": sig_row.get("band_pct"),
                    "has_stock": sig_row.get("has_stock"),
                    "entry_signal": sig_row.get("entry_signal"),
                    "ml_proba": sig_row.get("ml_proba"),
                    "entry_allowed": sig_row.get("entry_allowed"),
                    "note": sig_row.get("note"),
                    "strategy_name": sig_row.get("strategy_name"),
                    "cr_swing_proba": sig_row.get("cr_swing_proba"),
                }

        pre, post = load_ohlcv_window(trade_region, symbol, entry_time)
        pre_summary, post_summary = summarize_pre_post_windows(entry_price, pre, post)

        trade_details.append(
            {
                "id": trade_id,
                "symbol": symbol,
                "type": trade_type,
                "entry_time": entry_time.isoformat(),
                "entry_price": entry_price,
                "qty": qty,
                "profit": profit_val,
                "source": row.get("source"),
                "region": trade_region,
                "entry_comment": row.get("entry_comment"),
                "exit_comment": row.get("exit_comment"),
                "signal": signal_data,
                "pre_window": pre_summary,
                "post_window": post_summary,
            }
        )

    context = {
        "date": date_str,
        "region": market,
        "stats": stats,
        "trade_details": trade_details,
    }
    return context


# -----------------------------
# 전략 아이디어 브레인스토밍용 context (profit 기반)
# -----------------------------
def build_brainstorm_context(df: pd.DataFrame, target_date, region=None) -> dict:
    """
    전략 아이디어 브레인스토밍용 context.
    (로직 변경 없음)
    """
    date_str = target_date.strftime("%Y-%m-%d")

    if region is not None and not df.empty and "region" in df.columns:
        if region in ("CR", "COIN"):
            df = df[df["region"].isin(["CR", "COIN"])].copy()
        if region in ("BI", "COIN"):
            df = df[df["region"].isin(["BI", "COIN"])].copy()
        else:
            df = df[df["region"] == region].copy()

    if df.empty:
        return {
            "date": date_str,
            "region": region,
            "overall": {
                "total_trades": 0,
                "win_rate": 0.0,
                "total_profit": 0.0,
            },
            "by_symbol": [],
            "by_time_block": [],
            "model_notes": "해당 일자 데이터가 없습니다.",
        }

    profit = df["profit"].fillna(0)
    total_trades = len(df)
    win_rate = (profit > 0).sum() / total_trades * 100 if total_trades > 0 else 0.0

    overall = {
        "total_trades": int(total_trades),
        "win_rate": round(float(win_rate), 2),
        "total_profit": float(profit.sum()),
    }

    # 1) 심볼별
    by_symbol = []
    for symbol, sub in df.groupby("symbol"):
        p = sub["profit"].fillna(0)
        tot = len(sub)
        wins = (p > 0).sum()
        win_rate_sym = wins / tot * 100 if tot > 0 else 0.0
        by_symbol.append(
            {
                "symbol": symbol,
                "trades": int(tot),
                "win_rate": round(float(win_rate_sym), 2),
                "avg_profit": float(p.mean()),
            }
        )

    # 2) 시간대 블럭
    by_time_block = []
    if "time" in df.columns:
        df2 = df.copy()
        df2["time"] = pd.to_datetime(df2["time"])
        df2["hour"] = df2["time"].dt.hour

        def block_name(h):
            if 6 <= h < 12:
                return "MORNING"
            elif 12 <= h < 18:
                return "AFTERNOON"
            else:
                return "NIGHT"

        df2["time_block"] = df2["hour"].apply(block_name)

        for block, sub in df2.groupby("time_block"):
            p = sub["profit"].fillna(0)
            tot = len(sub)
            wins = (p > 0).sum()
            win_rate_blk = wins / tot * 100 if tot > 0 else 0.0
            by_time_block.append(
                {
                    "block": block,
                    "trades": int(tot),
                    "win_rate": round(float(win_rate_blk), 2),
                    "avg_profit": float(p.mean()),
                }
            )

    context = {
        "date": date_str,
        "region": region,
        "overall": overall,
        "by_symbol": by_symbol,
        "by_time_block": by_time_block,
        "model_notes": "",
    }
    return context


# -----------------------------
# 모델 업데이트 조언용 context
# -----------------------------
def load_model_context_for_ai(
    db: BotDatabase,
    target_date: date,
    market: str | None = None,
) -> dict:
    """
    models + backtests + settings + 최근 trades 를 조합.
    """
    # 1) db 인스턴스를 받아서 연결 사용
    conn = db.get_connection()
    try:
        df_models = pd.read_sql_query("SELECT * FROM models ORDER BY id ASC", conn)
        df_bt = pd.read_sql_query("SELECT * FROM backtests ORDER BY id ASC", conn)
    finally:
        conn.close()

    # 2) active 모델
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
                    "name": m.get("name", active_model_name),
                    "version": m.get("version", None),
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
            active = None

    # 3) candidate 모델
    candidate = None
    if not df_models.empty:
        m = df_models.iloc[-1]
        model_id = int(m["id"])
        bt = df_bt[df_bt["model_id"] == model_id]
        latest_bt = bt.iloc[-1] if not bt.empty else None

        candidate = {
            "model_id": int(m["id"]),
            "name": m.get("name", None),
            "version": m.get("version", None),
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

    # 4) 최근 실매매 성과 (Postgres 문법 적용)
    conn = db.get_connection()
    base_date = target_date.strftime("%Y-%m-%d")

    try:
        if market is None or market == "ALL":
            # Postgres: time::date >= (DATE %s - INTERVAL '2 days')
            # 혹은 간단히 문자열 비교 사용
            # 여기서는 Postgres의 날짜 연산 문법 사용: current_date - interval '2 days'
            # 파라미터로 처리하기 위해 python 단에서 날짜 계산하는게 더 편함
            
            # 파이썬에서 날짜 계산
            start_dt = pd.to_datetime(base_date) - pd.Timedelta(days=2)
            start_str = start_dt.strftime("%Y-%m-%d")

            q = "SELECT * FROM trades WHERE time >= %s"
            params = [start_str]

        elif market in ("CR", "COIN"):
            start_dt = pd.to_datetime(base_date) - pd.Timedelta(days=2)
            start_str = start_dt.strftime("%Y-%m-%d")

            q = "SELECT * FROM trades WHERE time >= %s AND region IN ('CR', 'COIN')"
            params = [start_str]
        
        elif market in ("BI", "COIN"):
            start_dt = pd.to_datetime(base_date) - pd.Timedelta(days=2)
            start_str = start_dt.strftime("%Y-%m-%d")

            q = "SELECT * FROM trades WHERE time >= %s AND region IN ('BI', 'COIN')"
            params = [start_str]

        else:
            start_dt = pd.to_datetime(base_date) - pd.Timedelta(days=2)
            start_str = start_dt.strftime("%Y-%m-%d")

            q = "SELECT * FROM trades WHERE time >= %s AND region = %s"
            params = [start_str, market]

        df_trades = pd.read_sql_query(q, conn, params=params)
    finally:
        conn.close()

    live_stats = {
        "recent_days": 3,
        "trades": 0,
        "win_rate": 0.0,
        "avg_profit": 0.0,
        "cum_profit": 0.0,
    }
    if not df_trades.empty:
        df_trades["profit"] = df_trades["profit"].fillna(0)
        profit = df_trades["profit"]
        total = len(df_trades)
        wins = (profit > 0).sum()
        live_stats = {
            "recent_days": 3,
            "trades": int(total),
            "win_rate": float(wins / total * 100) if total > 0 else 0.0,
            "avg_profit": float(profit.mean()),
            "cum_profit": float(profit.sum()),
        }

    settings = {
        "ml_threshold": float(db.get_setting("ml_threshold", 0.5) or 0.5),
        "max_positions": int(db.get_setting("max_positions", 3) or 3),
    }

    ctx = {
        "date": base_date,
        "market": market,
        "active": active,
        "candidate": candidate,
        "live_stats": live_stats,
        "settings": settings,
    }
    return ctx