# daily_ai_reports.py
import os
import sqlite3
from datetime import date, datetime
import pandas as pd

from db import BotDatabase
from ai_helpers import make_daily_trade_report_v2, brainstorm_strategy_ideas

DB_PATH = "trading.db"

# OHLCV 윈도우 설정 (엔트리 전/후 몇 개 캔들을 볼지)
INTERVAL = "5m"    # 네가 주로 쓰는 기본 인터벌로 맞춰줘
PRE_BARS = 20      # 엔트리 이전 캔들 개수
POST_BARS = 20     # 엔트리 이후 캔들 개수


# -----------------------------
# 기본 로딩 함수들
# -----------------------------
def load_trades_for_date(target_date: date) -> pd.DataFrame:
    """
    trades 테이블에서 해당 날짜의 트레이드 모두 로드.
    - time 컬럼 기준 (YYYY-MM-DD HH:MM:SS)
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT *
        FROM trades
        WHERE DATE(time) = DATE(?)
    """
    df = pd.read_sql_query(query, conn, params=[target_date.strftime("%Y-%m-%d")])
    conn.close()

    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df


def load_signal_for_trade(signal_id: int):
    """
    trades.signal_id -> signals.id 조인용 헬퍼.
    없으면 None 리턴.
    """
    if signal_id is None:
        return None
    conn = sqlite3.connect(DB_PATH)
    row = pd.read_sql_query(
        "SELECT * FROM signals WHERE id = ?",
        conn,
        params=[signal_id],
    )
    conn.close()
    if row.empty:
        return None
    return row.iloc[0].to_dict()


def load_ohlcv_window(region: str, symbol: str, entry_time: datetime):
    """
    엔트리 전/후로 PRE_BARS / POST_BARS 캔들을 가져오는 함수.
    - interval은 고정(INTERVAL)으로 사용.
    - region 또는 symbol이 없으면 (예: UNKNOWN) None 리턴.
    """
    if not region or region == "UNKNOWN" or not symbol:
        return None, None

    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT dt, open, high, low, close, volume
        FROM ohlcv_data
        WHERE region = ?
          AND symbol = ?
          AND interval = ?
        ORDER BY dt
        """,
        conn,
        params=[region, symbol, INTERVAL],
    )
    conn.close()

    if df.empty:
        return None, None

    df["dt"] = pd.to_datetime(df["dt"])

    # entry_time과 가장 가까운 캔들을 기준으로 윈도우 슬라이스
    idx = (df["dt"] - entry_time).abs().idxmin()

    pre = df.iloc[max(0, idx - PRE_BARS) : idx]
    post = df.iloc[idx : idx + POST_BARS]

    return pre, post


def summarize_pre_post_windows(entry_price: float, pre: pd.DataFrame, post: pd.DataFrame):
    """
    pre/post 캔들들로부터 간단한 지표들을 계산.
    - pre_window: 최근 추세/변동성
    - post_window: MFE/MAE/엔트리 후 N캔들 수익률
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
def build_daily_context_v2(df_trades: pd.DataFrame, target_date: date) -> dict:
    """
    v2 일일 리포트용 context 생성.
    - stats: 하루 전체 성과
    - trade_details: 트레이드별로 signal + pre/post 차트 요약까지 포함
    """
    date_str = target_date.strftime("%Y-%m-%d")

    if df_trades.empty:
        return {
            "date": date_str,
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

        # 1) 시그널 로드 (region, at_support, ... 파악)
        signal_data = None
        region = "UNKNOWN"
        signal_id = row.get("signal_id")
        if signal_id is not None:
            try:
                sig_row = load_signal_for_trade(int(signal_id))
            except Exception:
                sig_row = None

            if sig_row:
                region = sig_row.get("region") or "UNKNOWN"
                signal_data = {
                    "region": region,
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
                }

        # 2) OHLCV window (엔트리 전/후 차트 요약)
        pre, post = load_ohlcv_window(region, symbol, entry_time)
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
                "entry_comment": row.get("entry_comment"),
                "exit_comment": row.get("exit_comment"),
                "signal": signal_data,
                "pre_window": pre_summary,
                "post_window": post_summary,
            }
        )

    context = {
        "date": date_str,
        "stats": stats,
        "trade_details": trade_details,
    }
    return context


# -----------------------------
# 전략 아이디어 브레인스토밍용 context (profit 기반)
# -----------------------------
def build_brainstorm_context(df: pd.DataFrame, target_date: date) -> dict:
    """
    전략 아이디어 브레인스토밍용 context.
    v2에서도 일단 profit 기준의 심플 버전으로 유지.
    """
    date_str = target_date.strftime("%Y-%m-%d")

    if df.empty:
        return {
            "date": date_str,
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

    # 2) 시간대 블럭 (time 기준)
    by_time_block = []
    if "time" in df.columns:
        df2 = df.copy()
        df2["hour"] = df2["time"].dt.hour

        def block_name(h):
            # 필요하면 네 운용 시간대에 맞게 조정
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
        "overall": overall,
        "by_symbol": by_symbol,
        "by_time_block": by_time_block,
        "model_notes": "",
    }
    return context


# -----------------------------
# 메인 실행부
# -----------------------------
if __name__ == "__main__":
    target_date = date.today()  # 필요하면 특정 날짜로 바꿔도 됨

    # reports 폴더 없으면 생성
    os.makedirs("reports", exist_ok=True)

    db = BotDatabase(DB_PATH)
    df = load_trades_for_date(target_date)

    # 1) v2 일일 리포트 (차트+시그널 포함)
    daily_ctx = build_daily_context_v2(df, target_date)
    daily_report = make_daily_trade_report_v2(daily_ctx)

    print("\n========================")
    print("📊 일일 트레이드 리포트 (v2)")
    print("========================\n")
    print(daily_report)

    # 2) 전략 브레인스토밍
    brainstorm_ctx = build_brainstorm_context(df, target_date)
    ideas = brainstorm_strategy_ideas(brainstorm_ctx)

    print("\n========================")
    print("🧠 전략 아이디어 브레인스토밍")
    print("========================\n")
    print(ideas)

    # 3) 파일로 저장
    date_str = target_date.strftime("%Y-%m-%d")
    with open(f"reports/{date_str}_daily_report.txt", "w", encoding="utf-8") as f:
        f.write(daily_report)

    with open(f"reports/{date_str}_strategy_ideas.txt", "w", encoding="utf-8") as f:
        f.write(ideas)

    # 4) DB에도 저장 (대시보드 탭에서 불러쓸 용도)
    db.save_ai_report(
        date_str=date_str,
        daily_report=daily_report,
        strategy_ideas=ideas,
    )
