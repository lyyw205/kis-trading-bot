# web_app.py
from flask import Flask, render_template
import sqlite3
import pandas as pd

DB_PATH = "trading.db"

app = Flask(__name__)

def load_trades():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM trades ORDER BY time", conn)
    conn.close()
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df

def load_logs():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM logs ORDER BY time DESC", conn)
    conn.close()
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df

def load_signals(limit=200):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT * FROM signals ORDER BY time DESC LIMIT {limit}", conn
    )
    conn.close()
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df

def load_model_versions(limit=20):
    """model_versions 테이블에서 최신 모델 버전 목록을 가져옴."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"""
        SELECT id, name, version, created_at, params_json, note
        FROM model_versions
        ORDER BY datetime(created_at) DESC
        LIMIT {int(limit)}
        """,
        conn,
    )
    conn.close()
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def load_backtests(limit=50):
    """backtests 테이블에서 최근 백테스트 결과를 가져옴."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"""
        SELECT
            b.id,
            b.model_id,
            b.start_date,
            b.end_date,
            b.trades,
            b.win_rate,
            b.avg_profit,
            b.cum_return,
            b.max_dd,
            b.note
        FROM backtests b
        ORDER BY b.id DESC
        LIMIT {int(limit)}
        """,
        conn,
    )
    conn.close()
    return df

@app.route("/")
def dashboard():
    trades = load_trades()
    logs = load_logs()
    signals = load_signals(limit=200)
    model_versions = load_model_versions()
    backtests = load_backtests(limit=50)


    # 기본 값들
    summary = {
        "total_trades": 0,
        "win_rate": 0.0,
        "avg_profit": 0.0,
        "cum_return_pct": 0.0,
    }
    equity_curve = []
    symbols_avg = []

    if not trades.empty:
        trades_sorted = trades.sort_values("time").copy()

        # 승률, 평균 수익률, 누적 수익률
        total_trades = len(trades_sorted)
        wins = trades_sorted[trades_sorted["profit"] > 0]
        win_rate = len(wins) / total_trades * 100
        avg_profit = trades_sorted["profit"].mean()
        cum_return = (1 + trades_sorted["profit"] / 100).cumprod() - 1

        summary = {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "avg_profit": round(avg_profit, 2),
            "cum_return_pct": round(cum_return.iloc[-1] * 100, 2),
        }

        # 시간별 누적 곡선 (JS용 데이터)
        trades_sorted["cum_return"] = cum_return
        equity_curve = [
            {
                "time": t.strftime("%Y-%m-%d %H:%M"),
                "value": float(v * 100),
            }
            for t, v in zip(trades_sorted["time"], trades_sorted["cum_return"])
        ]

        # 종목별 평균 수익률 (JS용 데이터)
        by_symbol = trades.groupby("symbol")["profit"].mean().sort_values(ascending=False)
        symbols_avg = [
            {"symbol": s, "avg_profit": float(p)}
            for s, p in by_symbol.items()
        ]

    # 로그는 최근 200개만
    logs_recent = logs.head(200) if not logs.empty else pd.DataFrame()

    return render_template(
        "dashboard.html",
        summary=summary,
        equity_curve=equity_curve,
        symbols_avg=symbols_avg,
        trades=trades.to_dict(orient="records") if not trades.empty else [],
        logs=logs_recent.to_dict(orient="records") if not logs_recent.empty else [],
        signals=signals.to_dict(orient="records") if not signals.empty else [],
        model_versions=model_versions.to_dict(orient="records") if not model_versions.empty else [],
        backtests=backtests.to_dict(orient="records") if not backtests.empty else [],

    )

if __name__ == "__main__":
    app.run(debug=True, port=8000)
