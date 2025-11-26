# web_app.py
from flask import Flask, render_template
import sqlite3
import pandas as pd
import numpy as np
from datetime import date
from db import BotDatabase
from build_ohlcv_history import (
    get_universe_coverage,
    get_last_universe_backfill_time,
    get_recent_backfill_failures,
)

DB_PATH = "trading.db"

app = Flask(__name__)

# -----------------------------
# DB 로딩 함수들
# -----------------------------
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
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT id, created_at, path, n_samples, val_accuracy
        FROM models
        ORDER BY datetime(created_at) DESC
        LIMIT ?
        """,
        conn,
        params=(int(limit),),
    )
    conn.close()
    if not df.empty:
        df["created_at"] = pd.to_datetime(df["created_at"])
    return df


def load_backtests(limit=50):
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


def load_signals_by_date(target_date: str):
    """YYYY-MM-DD 기준으로 해당 날짜 signals만 불러오기"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT *
        FROM signals
        WHERE date(time) = ?
        ORDER BY time
        """,
        conn,
        params=(target_date,),
    )
    conn.close()
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df


def load_trades_by_date(target_date: str):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT *
        FROM trades
        WHERE date(time) = ?
        ORDER BY time
        """,
        conn,
        params=(target_date,),
    )
    conn.close()
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df

def load_ml_signals(limit=500):
    """
    ML 점수가 찍힌 최근 신호들만 가져오는 헬퍼
    monitor_ml.py의 SELECT를 그대로 복붙한 버전
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(
        f"""
        SELECT time, symbol, ml_proba, entry_signal, entry_allowed
        FROM signals
        WHERE ml_proba IS NOT NULL
        ORDER BY id DESC
        LIMIT {int(limit)}
        """,
        conn,
    )
    conn.close()

    if df.empty:
        return df

    df["time"] = pd.to_datetime(df["time"])
    # time 오름차순으로 정렬 (과거 → 최근)
    return df.sort_values("time")


# -----------------------------
# ML 개선안 생성 함수
# -----------------------------
def suggest_improvements(
    df_sig: pd.DataFrame,
    df_tr: pd.DataFrame,
    ml_threshold: float = 0.55,
):
    suggestions = []

    # 0) 신호 자체가 없는 경우
    if df_sig.empty:
        suggestions.append(
            "📉 오늘 저장된 신호가 없습니다. 타겟 종목 수나 장시간이 너무 짧지 않은지 점검해 보세요."
        )
        return suggestions

    total_signals = len(df_sig)
    rule_signals = int(df_sig["entry_signal"].fillna(0).sum())
    allowed = int(df_sig["entry_allowed"].fillna(0).sum())

    # 1) 룰 신호 개수
    if rule_signals == 0:
        suggestions.append(
            "⚠️ 룰 기반 시그널(entry_signal)이 한 번도 발생하지 않았습니다. "
            "lookback / band_pct 값을 완화해서 지지선 조건을 조금 느슨하게 하는 걸 고려해 보세요."
        )
    elif rule_signals <= 5:
        suggestions.append(
            f"ℹ️ 룰 시그널이 {rule_signals}건으로 적은 편입니다. "
            "타겟 종목을 조금 늘리거나 band_pct를 살짝 키우는 것도 방법입니다."
        )

    # 2) ML threshold 적절성
    if "ml_proba" in df_sig.columns and df_sig["ml_proba"].notna().any():
        mean_proba = df_sig["ml_proba"].mean()
        hi_ratio = (df_sig["ml_proba"] >= ml_threshold).mean()

        if hi_ratio < 0.05:
            suggestions.append(
                f"⚠️ ML 확률이 threshold({ml_threshold:.2f}) 이상인 비율이 {hi_ratio*100:.1f}%로 매우 낮습니다. "
                "임계값을 0.05~0.10 정도 낮춰서 더 많은 후보를 통과시키는 것도 테스트해 볼 만 합니다."
            )
        elif hi_ratio > 0.5:
            suggestions.append(
                f"ℹ️ ML 확률이 threshold({ml_threshold:.2f}) 이상인 비율이 {hi_ratio*100:.1f}%입니다. "
                "필터링이 느슨할 수 있으니 threshold를 약간 올려도 될지 확인해 보세요."
            )

        suggestions.append(
            f"📈 오늘 ML 평균 확률은 {mean_proba:.3f} 입니다. "
            "0.5~0.7 사이에 고르게 분포한다면 모델은 정상적으로 작동 중입니다."
        )

    # 3) 오늘 체결된 트레이드 성과
    if not df_tr.empty:
        realized = df_tr["profit"].dropna()
        realized = realized[realized != 0]  # profit=0 (매수)은 제외
        num_trades = len(realized)
        if num_trades > 0:
            wins = (realized > 0).sum()
            win_rate = wins / num_trades
            avg_profit = realized.mean()

            suggestions.append(
                f"💰 오늘 체결된 트레이드는 {num_trades}건, 승률 {win_rate*100:.1f}%, "
                f"트레이드당 평균 수익률 {avg_profit:.2f}% 입니다."
            )

    # 4) 오늘 가장 많이 신호가 나온 종목
    sym_count = df_sig["symbol"].value_counts()
    if len(sym_count) > 0:
        top_sym = sym_count.index[0]
        top_cnt = sym_count.iloc[0]
        suggestions.append(
            f"🔍 오늘 가장 많이 신호가 나온 종목은 '{top_sym}' ({top_cnt}회) 입니다. "
            "차트를 직접 보면서 모델이 어떤 패턴을 포착했는지 눈으로 확인해 보세요."
        )

    if not suggestions:
        suggestions.append(
            "✅ 특이사항 없이 안정적으로 러닝이 돌아간 하루였습니다. "
            "현재 설정을 유지하면서 데이터만 더 쌓아도 좋습니다."
        )

    return suggestions


# -----------------------------
# 메인 대시보드 라우트
# -----------------------------
@app.route("/")
def dashboard():
    trades = load_trades()
    logs = load_logs()
    signals = load_signals(limit=200)
    model_versions = load_model_versions()
    backtests = load_backtests(limit=50)
    # Universe 데이터 대시보드용
    universe_cov = get_universe_coverage()
    last_universe_backfill = get_last_universe_backfill_time(db=BotDatabase(DB_PATH))
    universe_failures = get_recent_backfill_failures(limit=30)

    # ✅ 카드용 숫자들 미리 계산
    if not universe_cov.empty:
        num_universe_symbols = int(len(universe_cov))
        total_universe_candles = int(universe_cov["candles"].sum())
        max_days_covered = int(universe_cov["days_covered"].max())
    else:
        num_universe_symbols = 0
        total_universe_candles = 0
        max_days_covered = 0

    # 기본 요약
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

        trades_sorted["cum_return"] = cum_return
        equity_curve = [
            {
                "time": t.strftime("%Y-%m-%d %H:%M"),
                "value": float(v * 100),
            }
            for t, v in zip(trades_sorted["time"], trades_sorted["cum_return"])
        ]

        by_symbol = trades.groupby("symbol")["profit"].mean().sort_values(ascending=False)
        symbols_avg = [
            {"symbol": s, "avg_profit": float(p)}
            for s, p in by_symbol.items()
        ]

    logs_recent = logs.head(200) if not logs.empty else pd.DataFrame()

    # 오늘 기준 신호/트레이드 + 필요하면 최근 신호로 대체
    today_str = date.today().strftime("%Y-%m-%d")
    today_signals = load_signals_by_date(today_str)
    today_trades = load_trades_by_date(today_str)

    if today_signals.empty:
        today_signals = load_signals(limit=200)

    suggestions = suggest_improvements(
        df_sig=today_signals,
        df_tr=today_trades,
        ml_threshold=0.55,
    )

    ml_signals = load_ml_signals(limit=500)

    ml_hist_labels = []
    ml_hist_counts = []
    ml_time_series = []

    if not ml_signals.empty:
        # 1) 히스토그램용 (0.0~1.0을 10개 구간으로 자름)
        #    ml_proba 범위가 0~1 아니라면 bins 조절해도 됨
        bins = np.linspace(0, 1, 11)  # [0.0,0.1,...,1.0]
        ml_signals["bin"] = pd.cut(
            ml_signals["ml_proba"],
            bins=bins,
            include_lowest=True,
            right=False,
        )

        bin_counts = ml_signals["bin"].value_counts().sort_index()

        ml_hist_labels = [
            f"{interval.left:.1f}~{interval.right:.1f}"
            for interval in bin_counts.index
        ]
        ml_hist_counts = [int(c) for c in bin_counts.values]

        # 2) 시간 시계열용
        ml_time_series = [
            {
                "time": t.strftime("%Y-%m-%d %H:%M"),
                "proba": float(p),
                "entry_allowed": int(e) if pd.notna(e) else 0,
            }
            for t, p, e in zip(
                ml_signals["time"],
                ml_signals["ml_proba"],
                ml_signals["entry_allowed"].fillna(0),
            )
        ]

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
        suggestions=suggestions,
        universe_cov=universe_cov.to_dict(orient="records") if not universe_cov.empty else [],
        last_universe_backfill=last_universe_backfill,
        universe_failures=universe_failures.to_dict(orient="records") if not universe_failures.empty else [],
        num_universe_symbols=num_universe_symbols,
        total_universe_candles=total_universe_candles,
        max_days_covered=max_days_covered,
        ml_hist_labels=ml_hist_labels,
        ml_hist_counts=ml_hist_counts,
        ml_time_series=ml_time_series,
    )


if __name__ == "__main__":
    app.run(debug=True, port=8000)
