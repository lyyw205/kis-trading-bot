# d_web.py
from flask import Flask, render_template, jsonify, request, redirect
import pandas as pd
import numpy as np
from datetime import date
from urllib.parse import quote_plus
import warnings

# SQLAlchemy 경고 숨기기
warnings.filterwarnings("ignore", category=UserWarning)

from c_db_manager import BotDatabase
from d_web_universe_ohlcv import (
    get_universe_coverage,
    get_last_universe_backfill_time,
    get_recent_backfill_failures,
)
from d_web_data import (
    get_connection,
    load_logs,
    load_signals,
    load_model_versions,
    load_backtests,
    load_signals_by_date,
    load_ml_signals,
    suggest_improvements,
)

app = Flask(__name__)

# -----------------------------------------------------------
# [DB 설정] Supabase
# -----------------------------------------------------------
DB_HOST = "aws-1-ap-northeast-2.pooler.supabase.com"
DB_NAME = "postgres"
DB_USER = "postgres.sxhtnkxulfrqykrtwxjx"
DB_PASS = "Shitdog205!@"
DB_PORT = "6543"

encoded_pass = quote_plus(DB_PASS)
DB_URL = f"postgresql://{DB_USER}:{encoded_pass}@{DB_HOST}:{DB_PORT}/{DB_NAME}?sslmode=require"

# -----------------------------
# 공통 유틸
# -----------------------------
def normalize_region(region_raw: str) -> str:
    region = (region_raw or "ALL").upper()
    if region not in ("ALL", "KR", "US", "CR", "BI"):
        region = "ALL"
    return region

def filter_by_region(region: str, df: pd.DataFrame):
    if df is None or df.empty:
        return df
    if "region" not in df.columns:
        return df
    if region == "ALL":
        return df
    return df[df["region"] == region].copy()

def fetch_positions_as_trades(region: str):
    """
    [수정] positions 테이블 스키마(DDL)에 맞춘 데이터 로드
    """
    conn = get_connection()
    # 새로운 컬럼명 반영
    query = """
        SELECT
            id,
            region,
            symbol,
            trade_type,
            entry_qty,
            entry_price,
            entry_time,
            exit_time,
            exit_price,
            pnl_pct,
            status,
            entry_comment,
            exit_comment,
            ml_proba
        FROM positions
    """
    params = []
    
    if region != "ALL":
        query += " WHERE region = %s"
        params.append(region)
        
    query += " ORDER BY entry_time DESC"

    try:
        df = pd.read_sql_query(query, conn, params=tuple(params))
    except Exception as e:
        print(f"!!! [ERROR] Failed to fetch positions: {e}")
        df = pd.DataFrame()
    finally:
        conn.close()

    if df.empty:
        return pd.DataFrame()

    # 날짜 변환
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors='coerce')
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors='coerce')
    
    # 숫자 변환
    df["pnl_pct"] = pd.to_numeric(df["pnl_pct"], errors='coerce').fillna(0.0)
    df["entry_qty"] = pd.to_numeric(df["entry_qty"], errors='coerce').fillna(0.0)
    df["entry_price"] = pd.to_numeric(df["entry_price"], errors='coerce').fillna(0.0)

    # 내부 로직 호환성을 위한 컬럼 매핑
    # trade_type -> side, pnl_pct -> profit, entry_time -> time
    df["side"] = df["trade_type"]
    df["profit"] = df["pnl_pct"]
    df["time"] = df["entry_time"]
    df["is_open"] = df["status"] == "OPEN"
    
    return df

def build_round_trades_structure(df: pd.DataFrame):
    if df.empty:
        return pd.DataFrame(), {}

    rows = []
    round_details = {}

    for _, row in df.iterrows():
        entry_time = row.get("entry_time")
        
        # 날짜 문자열
        date_str = entry_time.strftime("%Y-%m-%d") if pd.notna(entry_time) else ""
        
        # 라운드 ID 키 생성 (상세 정보 매핑용)
        rid = row.get("id")
        symbol = row.get("symbol")
        key = f"{symbol}__{rid}"

        rows.append({
            "round_id": rid,
            "symbol": symbol,
            "region": row.get("region"),
            "date": date_str,
            "entry_time": entry_time,
            "exit_time": row.get("exit_time"), # exit_time 사용
            "status": row.get("status"),       # 'OPEN' or 'CLOSED'
            "entry_qty": float(row.get("entry_qty") or 0.0),
            "entry_price": float(row.get("entry_price") or 0.0),
            "realized_profit_pct": float(row.get("pnl_pct") or 0.0),
            "entry_comment": row.get("entry_comment"),
            "exit_comment": row.get("exit_comment"),
        })

        # 상세(Detail) 행에 보여줄 가상의 체결 내역 생성
        # (Positions 테이블 하나로 퉁치므로, 진입/청산 2개의 레코드로 쪼개서 보여줌)
        details = []
        
        # 1. 진입
        if pd.notna(entry_time):
            details.append({
                "time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": "ENTRY (" + str(row.get("trade_type")) + ")",
                "price": float(row.get("entry_price") or 0),
                "qty": float(row.get("entry_qty") or 0),
                "ml_proba": float(row.get("ml_proba")) if pd.notna(row.get("ml_proba")) else None
            })
        
        # 2. 청산 (있을 경우)
        exit_time = row.get("exit_time")
        if pd.notna(exit_time):
            details.append({
                "time": exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                "type": "EXIT",
                "price": float(row.get("exit_price") or 0),
                "qty": float(row.get("exit_qty") or 0),
                "ml_proba": None 
            })
            
        round_details[key] = details

    return pd.DataFrame(rows), round_details

def load_all_dashboard_data(region_raw: str):
    region = normalize_region(region_raw)

    trades_df = fetch_positions_as_trades(region)
    
    logs = filter_by_region(region, load_logs())
    signals = filter_by_region(region, load_signals(limit=200))
    model_versions = filter_by_region(region, load_model_versions())
    backtests = filter_by_region(region, load_backtests(limit=50))
    universe_cov = filter_by_region(region, get_universe_coverage())
    
    try:
        db_instance = BotDatabase(DB_URL)
        last_universe_backfill = get_last_universe_backfill_time(db=db_instance)
    except Exception as e:
        print(f"[Warning] Universe check failed: {e}")
        last_universe_backfill = None

    universe_failures = filter_by_region(region, get_recent_backfill_failures(limit=30))

    if not universe_cov.empty:
        universe_cov = universe_cov.sort_values("candles", ascending=False)

    round_trades_df, round_details = build_round_trades_structure(trades_df)

    # 통계
    if not universe_cov.empty:
        num_universe_symbols = int(len(universe_cov))
        total_universe_candles = int(universe_cov["candles"].sum())
        max_days_covered = int(universe_cov["days_covered"].max())
    else:
        num_universe_symbols = 0
        total_universe_candles = 0
        max_days_covered = 0

    summary = {
        "total_trades": 0,
        "win_rate": 0.0,
        "avg_profit": 0.0,
        "cum_return_pct": 0.0,
    }
    equity_curve = []
    symbols_avg = []
    daily_summaries = []

    if not trades_df.empty:
        trades_sorted = trades_df.sort_values("time").copy()
        
        # 청산 완료된 건만 통계에 포함할지, 전체 포함할지 결정. 여기선 PNL이 있는 것 기준.
        closed_trades = trades_sorted[trades_sorted["status"] == "CLOSED"]
        
        total_trades = len(closed_trades)
        if total_trades > 0:
            wins = closed_trades[closed_trades["profit"] > 0]
            win_rate = (len(wins) / total_trades * 100)
            avg_profit = closed_trades["profit"].mean()
            
            # 누적 수익률
            cum_return = (1 + closed_trades["profit"] / 100).cumprod() - 1
            last_cum = float(cum_return.iloc[-1]) if len(cum_return) > 0 else 0.0

            summary = {
                "total_trades": total_trades,
                "win_rate": round(win_rate, 2),
                "avg_profit": round(avg_profit, 2),
                "cum_return_pct": round(last_cum * 100, 2),
            }

            closed_trades["cum_return"] = cum_return
            equity_curve = [
                {
                    "time": t.strftime("%Y-%m-%d %H:%M"),
                    "value": float(v * 100),
                }
                for t, v in zip(closed_trades["time"], closed_trades["cum_return"])
            ]

        # 심볼별 평균 (전체 기준)
        by_symbol = trades_sorted.groupby("symbol")["profit"].mean().sort_values(ascending=False)
        symbols_avg = [{"symbol": s, "avg_profit": float(p)} for s, p in by_symbol.items()]

        # 일별 요약 (Closed 기준)
        if not closed_trades.empty:
            closed_trades["date_str"] = closed_trades["time"].dt.strftime("%Y-%m-%d")
            for d, df_day in closed_trades.groupby("date_str"):
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
    
    today_str = date.today().strftime("%Y-%m-%d")
    today_signals = filter_by_region(region, load_signals_by_date(today_str))
    
    today_trades = pd.DataFrame()
    if not trades_df.empty:
        today_trades = trades_df[trades_df['time'].dt.strftime("%Y-%m-%d") == today_str]

    if today_signals.empty:
        today_signals = filter_by_region(region, load_signals(limit=200))

    suggestions = suggest_improvements(
        df_sig=today_signals,
        df_tr=today_trades,
        ml_threshold=0.55,
    )

    ml_signals = filter_by_region(region, load_ml_signals(limit=500))
    ml_hist_labels, ml_hist_counts, ml_time_series = [], [], []

    if not ml_signals.empty:
        bins = np.linspace(0, 1, 11)
        ml_signals["bin"] = pd.cut(ml_signals["ml_proba"], bins=bins, include_lowest=True, right=False)
        bin_counts = ml_signals["bin"].value_counts().sort_index()

        ml_hist_labels = [f"{interval.left:.1f}~{interval.right:.1f}" for interval in bin_counts.index]
        ml_hist_counts = [int(c) for c in bin_counts.values]

        ml_time_series = [
            {
                "time": t.strftime("%Y-%m-%d %H:%M"),
                "proba": float(p),
                "entry_allowed": int(e) if pd.notna(e) else 0,
            }
            for t, p, e in zip(ml_signals["time"], ml_signals["ml_proba"], ml_signals["entry_allowed"].fillna(0))
        ]

    symbols_with_data = []
    if not trades_df.empty:
        symbols_with_data = sorted(trades_df['symbol'].unique().tolist())

    return {
        "region": region,
        "summary": summary,
        "equity_curve": equity_curve,
        "symbols_avg": symbols_avg,
        "trades": trades_df,
        "logs_recent": logs_recent,
        "signals": signals,
        "model_versions": model_versions,
        "backtests": backtests,
        "universe_cov": universe_cov,
        "last_universe_backfill": last_universe_backfill,
        "universe_failures": universe_failures,
        "num_universe_symbols": num_universe_symbols,
        "total_universe_candles": total_universe_candles,
        "max_days_covered": max_days_covered,
        "round_trades_df": round_trades_df,
        "round_details": round_details,
        "daily_summaries": daily_summaries,
        "suggestions": suggestions,
        "ml_hist_labels": ml_hist_labels,
        "ml_hist_counts": ml_hist_counts,
        "ml_time_series": ml_time_series,
        "symbols_with_data": symbols_with_data,
    }

@app.route("/symbol_data")
def symbol_data():
    symbol = request.args.get("symbol")
    region_raw = request.args.get("region", "ALL")
    
    if not symbol:
        return jsonify({"error": "symbol parameter required"}), 400

    region = (region_raw or "ALL").upper()
    conn = get_connection()
    
    try:
        # 캔들
        candles = pd.read_sql_query(
            """
            SELECT dt, open, high, low, close
            FROM (
                SELECT dt, open, high, low, close
                FROM ohlcv_data
                WHERE symbol = %s
                  AND interval = '5m'
                ORDER BY dt DESC
                LIMIT 500
            ) t
            ORDER BY dt ASC
            """,
            conn,
            params=(symbol,),
        )

        # positions (새 스키마)
        pos_query = """
            SELECT 
                region, trade_type, entry_qty, entry_time, entry_price, 
                exit_time, exit_price, status
            FROM positions
            WHERE symbol = %s
        """
        pos_params = [symbol]
        if region != "ALL":
            pos_query += " AND region = %s"
            pos_params.append(region)
        pos_query += " ORDER BY entry_time"
        
        positions = pd.read_sql_query(pos_query, conn, params=tuple(pos_params))

    except Exception as e:
        print(f"Error fetching symbol data: {e}")
        return jsonify({"error": "DB Error"}), 500
    finally:
        conn.close()

    if candles.empty:
        return jsonify({"candles": [], "trades": []})

    candles["dt"] = pd.to_datetime(candles["dt"])
    candles_json = [
        {
            "time": row["dt"].strftime("%Y-%m-%d %H:%M"),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "close": float(row["close"]),
        }
        for _, row in candles.iterrows()
    ]

    trade_rows = []
    if not positions.empty:
        positions["entry_time"] = pd.to_datetime(positions["entry_time"])
        positions["exit_time"] = pd.to_datetime(positions["exit_time"])
        candle_times = candles["dt"].values

        for _, row in positions.iterrows():
            # 1. 진입 마커
            et = row.get("entry_time")
            ep = row.get("entry_price")
            trade_type = str(row.get("trade_type") or "BUY").upper()
            
            if pd.notna(et) and ep is not None:
                # 롱(BUY)이면 BUY 마커, 숏(SELL)이면 SELL 마커
                m_type = "BUY"
                if "SHORT" in trade_type or "SELL" in trade_type:
                    m_type = "SELL"
                
                tt = et.to_datetime64()
                pos_idx = candle_times.searchsorted(tt, side="right") - 1
                if 0 <= pos_idx < len(candle_times):
                    trade_rows.append({
                        "x_index": int(pos_idx),
                        "time": et.strftime("%Y-%m-%d %H:%M:%S"),
                        "type": m_type,
                        "price": float(ep),
                        "qty": float(row.get("entry_qty") or 0)
                    })
            
            # 2. 청산 마커
            xt = row.get("exit_time")
            xp = row.get("exit_price")
            status = row.get("status")
            
            # CLOSED 상태이고 청산 정보가 있을 때만
            if status == "CLOSED" and pd.notna(xt) and xp is not None:
                # 청산은 진입의 반대 마커
                m_type_exit = "SELL" if m_type == "BUY" else "BUY"
                
                tt_exit = xt.to_datetime64()
                pos_idx_exit = candle_times.searchsorted(tt_exit, side="right") - 1
                if 0 <= pos_idx_exit < len(candle_times):
                    trade_rows.append({
                        "x_index": int(pos_idx_exit),
                        "time": xt.strftime("%Y-%m-%d %H:%M:%S"),
                        "type": m_type_exit,
                        "price": float(xp),
                        "qty": float(row.get("entry_qty") or 0) # entry_qty 기준으로 표시
                    })

    trade_rows.sort(key=lambda x: x["time"])
    return jsonify({"candles": candles_json, "trades": trade_rows})

# ... (나머지 라우트는 동일) ...
@app.route("/sync-app-trades", methods=["POST"])
def sync_app_trades_route():
    return jsonify({"ok": True, "inserted": 0})

@app.route("/api/ai-report/full")
def api_ai_report_full():
    return jsonify({}) 

@app.route("/")
def root():
    region = normalize_region(request.args.get("region", "ALL"))
    return redirect(f"/dash/overview?region={region}")

@app.route("/dash/overview")
def dash_overview():
    region = request.args.get("region", "ALL")
    data = load_all_dashboard_data(region)
    return render_template(
        "dash_overview.html",
        page_title="요약 / 성능",
        active_page="overview",
        region=data["region"],
        summary=data["summary"],
        equity_curve=data["equity_curve"],
        symbols_avg=data["symbols_avg"],
        suggestions=data["suggestions"],
    )

@app.route("/dash/ai-report")
def dash_ai_report():
    region = request.args.get("region", "ALL")
    data = load_all_dashboard_data(region)
    return render_template("dash_ai_report.html", page_title="AI 리포트", active_page="ai-report", region=data["region"])

@app.route("/dash/trades")
@app.route("/dash/logs")
def dash_redirects():
    return redirect(f"/dash/auto?region={normalize_region(request.args.get('region', 'ALL'))}")

@app.route("/dash/universe")
def dash_universe():
    region = request.args.get("region", "ALL")
    data = load_all_dashboard_data(region)
    return render_template(
        "dash_universe.html",
        page_title="Universe OHLCV",
        active_page="universe",
        region=data["region"],
        universe_cov=data["universe_cov"].to_dict(orient="records") if not data["universe_cov"].empty else [],
        last_universe_backfill=data["last_universe_backfill"],
        universe_failures=data["universe_failures"].to_dict(orient="records") if not data["universe_failures"].empty else [],
        num_universe_symbols=data["num_universe_symbols"],
        total_universe_candles=data["total_universe_candles"],
        max_days_covered=data["max_days_covered"],
    )

@app.route("/dash/ml")
def dash_ml():
    region = request.args.get("region", "ALL")
    data = load_all_dashboard_data(region)
    return render_template(
        "dash_ml.html",
        page_title="ML 모니터링",
        active_page="ml",
        region=data["region"],
        ml_hist_labels=data["ml_hist_labels"],
        ml_hist_counts=data["ml_hist_counts"],
        ml_time_series=data["ml_time_series"],
        model_versions=data["model_versions"].to_dict(orient="records") if not data["model_versions"].empty else [],
        backtests=data["backtests"].to_dict(orient="records") if not data["backtests"].empty else [],
    )

@app.route("/dash/symbols")
def dash_symbols():
    region = request.args.get("region", "ALL")
    data = load_all_dashboard_data(region)
    return render_template(
        "dash_symbols.html",
        page_title="종목별 차트",
        active_page="symbols",
        region=data["region"],
        symbols_with_data=data["symbols_with_data"],
    )

@app.route("/dash/auto")
def dash_auto():
    region = request.args.get("region", "ALL")
    data = load_all_dashboard_data(region)
    
    trades_list = []
    if not data["round_trades_df"].empty:
        sorted_round_trades = data["round_trades_df"].sort_values("entry_time", ascending=False)
        trades_list = sorted_round_trades.to_dict(orient="records")

    return render_template(
        "dash_auto.html",
        page_title="자동매매",
        active_page="auto",
        region=data["region"],
        universe_failures=data["universe_failures"].to_dict(orient="records") if not data["universe_failures"].empty else [],
        round_trades=trades_list,
        round_details=data["round_details"],
        daily_summaries=data["daily_summaries"],
        summary=data["summary"],
        logs=data["logs_recent"].to_dict(orient="records") if not data["logs_recent"].empty else [],
        signals=data["signals"].to_dict(orient="records") if not data["signals"].empty else [],
        symbols_with_data=data["symbols_with_data"],
    )

if __name__ == "__main__":
    app.run(debug=True, port=8000)