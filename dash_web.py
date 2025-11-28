# dash_web_app.py  (기존 web_app.py 내용을 교체하거나, 파일명을 이렇게 바꿔도 됨)
from flask import Flask, render_template, jsonify, request
import os
import sqlite3
import pandas as pd
import numpy as np
from glob import glob
from datetime import date
from zz_import_app_fills import sync_app_fills_main

from db_manager import BotDatabase
from dash_universe_ohlcv import (
    get_universe_coverage,
    get_last_universe_backfill_time,
    get_recent_backfill_failures,
)
from dash_data import (
    DB_PATH,
    load_trades,
    load_logs,
    load_signals,
    load_model_versions,
    load_backtests,
    load_signals_by_date,
    load_trades_by_date,
    build_round_trades,
    load_ml_signals,
    suggest_improvements,
    get_symbols_with_data,
)

app = Flask(__name__)


# -----------------------------
# 앱 체결 동기화 라우트들
# -----------------------------
def sync_app_trades():
    """
    TODO: 여기에서 앱(모바일/웹)에서 발생한 체결 내역을 브로커/DB 등에서 읽어와서
    trades 테이블에 INSERT 하는 로직 구현.
    반환값: 새로 추가된 row 개수 (int)
    """
    return 0  # 일단 더미


@app.route("/sync-app-trades", methods=["POST"])
def sync_app_trades_route():
    """
    대시보드에서 버튼 누를 때마다 한번만 호출되는 엔드포인트
    """
    try:
        inserted = sync_app_trades()
        return jsonify({"ok": True, "inserted": int(inserted)})
    except Exception as e:
        print("sync_app_trades 오류:", e)
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route("/sync_app_fills", methods=["POST"])
def sync_app_fills():
    try:
        inserted = sync_app_fills_main()
        return jsonify({
            "status": "ok",
            "message": f"동기화 완료: {inserted}개 체결 저장됨.",
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


# -----------------------------
# 시각화용 심볼 데이터 API
# -----------------------------
@app.route("/symbol_data")
def symbol_data():
    """
    특정 종목의 가격 시계열 + 매매 기록(BUY/SELL)을 리턴하는 API
    - ohlcv_data 에서 close 가격 시계열
    - trades 에서 해당 종목의 매매 내역
    """
    symbol = request.args.get("symbol")
    if not symbol:
        return jsonify({"error": "symbol parameter required"}), 400

    conn = sqlite3.connect(DB_PATH)

    # 1) 캔들 데이터 (예: 5분봉, 최근 500개만)
    candles = pd.read_sql_query(
        """
        SELECT dt, open, high, low, close
        FROM ohlcv_data
        WHERE symbol = ?
          AND interval = '5m'
        ORDER BY datetime(dt)
        LIMIT 500
        """,
        conn,
        params=(symbol,),
    )

    # 2) 해당 종목 매매 내역
    trades = pd.read_sql_query(
        """
        SELECT time, type, price, qty
        FROM trades
        WHERE symbol = ?
        ORDER BY datetime(time)
        """,
        conn,
        params=(symbol,),
    )

    conn.close()

    if candles.empty:
        return jsonify({"candles": [], "trades": []})

    candles["dt"] = pd.to_datetime(candles["dt"])
    if not trades.empty:
        trades["time"] = pd.to_datetime(trades["time"])

    candle_times = candles["dt"].values  # numpy array
    trade_rows = []
    if not trades.empty:
        for _, row in trades.iterrows():
            tt = row["time"].to_datetime64()
            pos = candle_times.searchsorted(tt, side="right") - 1
            if pos < 0 or pos >= len(candle_times):
                continue

            trade_rows.append(
                {
                    "x_index": int(pos),
                    "time": row["time"].strftime("%Y-%m-%d %H:%M:%S"),
                    "type": row["type"],
                    "price": float(row["price"]),
                    "qty": float(row["qty"]),
                }
            )

    return jsonify(
        {
            "candles": [
                {
                    "time": row["dt"].strftime("%Y-%m-%d %H:%M"),
                    "open": float(row["open"]),
                    "high": float(row["high"]),
                    "low": float(row["low"]),
                    "close": float(row["close"]),
                }
                for _, row in candles.iterrows()
            ],
            "trades": trade_rows,
        }
    )


# -----------------------------
# AI 리포트 API
# -----------------------------
@app.route("/api/ai-report/full")
def api_ai_report_full():
    """
    최신 AI 리포트(일일 트레이드 리포트 + 전략 아이디어) +
    최신 모델 조언 텍스트까지 한 번에 내려주는 엔드포인트.
    프론트에서 3단 레이아웃으로 쓰는 용도.
    """
    result = {
        "date": None,
        "created_at": None,
        "daily_report": "",
        "strategy_ideas": "",
        "model_advice_date": None,
        "model_advice": "",
    }

    # 1) ai_reports 테이블에서 최신 1건 가져오기
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            SELECT date, created_at, daily_report, strategy_ideas
            FROM ai_reports
            ORDER BY date DESC, id DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        conn.close()

        if row:
            result["date"] = row[0]
            result["created_at"] = row[1]
            result["daily_report"] = row[2] or ""
            result["strategy_ideas"] = row[3] or ""
    except Exception as e:
        print("ai_reports 조회 오류:", e)

    # 2) reports 폴더에서 *_model_advice.txt 중 가장 최신 파일 읽기
    try:
        os.makedirs("reports", exist_ok=True)
        files = glob(os.path.join("reports", "*_model_advice.txt"))
        if files:
            latest_file = max(files, key=os.path.getmtime)
            base = os.path.basename(latest_file)
            if base.endswith("_model_advice.txt"):
                date_part = base.replace("_model_advice.txt", "")
            else:
                date_part = None

            with open(latest_file, "r", encoding="utf-8") as f:
                text = f.read()

            result["model_advice_date"] = date_part
            result["model_advice"] = text
    except Exception as e:
        print("model_advice 파일 로드 오류:", e)

    return jsonify(result)


# -----------------------------
# 메인 대시보드 라우트
# -----------------------------
@app.route("/")
def dashboard():
        # ✅ URL 쿼리로 region 받기 : ALL, KR, US, CR
    region = request.args.get("region", "ALL").upper()
    if region not in ("ALL", "KR", "US", "CR"):
        region = "ALL"

    # 공통 필터 함수
    def filter_by_region(df):
        if df is None or len(getattr(df, "columns", [])) == 0:
            return df
        if "region" not in df.columns:
            # region 컬럼 없으면 건드리지 않음 (예: 글로벌 로그, 모델 히스토리 등)
            return df
        if region == "ALL":
            return df
        return df[df["region"] == region].copy()
    
    trades = load_trades()
    logs = load_logs()
    signals = load_signals(limit=200)
    model_versions = load_model_versions()
    backtests = load_backtests(limit=50)

    # Universe 데이터 대시보드용
    universe_cov = get_universe_coverage()
    last_universe_backfill = get_last_universe_backfill_time(db=BotDatabase(DB_PATH))
    universe_failures = get_recent_backfill_failures(limit=30)

    # ✅ region 필터 적용
    trades = filter_by_region(trades)
    logs = filter_by_region(logs)
    signals = filter_by_region(signals)
    model_versions = filter_by_region(model_versions)
    backtests = filter_by_region(backtests)
    universe_cov = filter_by_region(universe_cov)
    universe_failures = filter_by_region(universe_failures)

    if not universe_cov.empty:
        universe_cov = universe_cov.sort_values("candles", ascending=False)

    if not trades.empty:
        round_trades_df, round_details = build_round_trades(trades)
    else:
        round_trades_df, round_details = pd.DataFrame(), {}

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
    daily_summaries = []

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

        tmp = trades_sorted.copy()
        tmp["date"] = tmp["time"].dt.strftime("%Y-%m-%d")

        daily_summaries = []
        for d, df_day in tmp.groupby("date"):
            n = len(df_day)
            wins_day = (df_day["profit"] > 0).sum()
            win_rate_day = (wins_day / n * 100) if n > 0 else 0.0
            avg_profit_day = df_day["profit"].mean() if n > 0 else 0.0
            cum_ret_day = (1 + df_day["profit"] / 100).prod() - 1

            daily_summaries.append({
                "date": d,
                "total_trades": int(n),
                "win_rate": round(win_rate_day, 2),
                "avg_profit": round(avg_profit_day, 2),
                "cum_return_pct": round(cum_ret_day * 100, 2),
            })

        daily_summaries.sort(key=lambda x: x["date"], reverse=True)

    logs_recent = logs.head(200) if not logs.empty else pd.DataFrame()

    # 오늘 기준 신호/트레이드 + 필요하면 최근 신호로 대체
    today_str = date.today().strftime("%Y-%m-%d")
    today_signals = load_signals_by_date(today_str)
    today_trades = load_trades_by_date(today_str)

    # ✅ region 필터
    today_signals = filter_by_region(today_signals)
    today_trades = filter_by_region(today_trades)

    if today_signals.empty:
        today_signals = filter_by_region(load_signals(limit=200))

    suggestions = suggest_improvements(
        df_sig=today_signals,
        df_tr=today_trades,
        ml_threshold=0.55,
    )

    ml_signals = load_ml_signals(limit=500)
    ml_signals = filter_by_region(ml_signals)

    ml_hist_labels = []
    ml_hist_counts = []
    ml_time_series = []

    if not ml_signals.empty:
        # 1) 히스토그램용 (0.0~1.0을 10개 구간으로 자름)
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

    # 종목 드롭다운용 심볼 리스트
    symbols_with_data = get_symbols_with_data(trades)

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
        round_trades=round_trades_df.to_dict(orient="records") if not round_trades_df.empty else [],
        round_details=round_details,
        daily_summaries=daily_summaries,
        symbols_with_data=symbols_with_data,
        region=region,   # ✅ 추가
    )


if __name__ == "__main__":
    app.run(debug=True, port=8000)
