# web_app.py
from flask import Flask, render_template, jsonify, request
import subprocess
import os
import sqlite3
import pandas as pd
import numpy as np
from glob import glob 
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

def build_round_trades(df_trades: pd.DataFrame):
    """
    trades 테이블에서 '포지션 단위(라운드 트립)' 요약 + 라운드별 체결 리스트 생성

    반환:
      round_trades_df, round_details_dict

      - round_trades_df: 각 (symbol, round_id)별 요약 행
      - round_details_dict: { "SYMBOL__round_id": [ {time, type, price, qty, ml_proba}, ... ] }
    """
    if df_trades.empty:
        return pd.DataFrame(), {}

    df = df_trades.sort_values("time").copy()

    if "type" not in df.columns:
        return pd.DataFrame(), {}

    # 종목별 포지션 추적
    def assign_round_id(group):
        signed_qty = np.where(group["type"] == "BUY", group["qty"], -group["qty"])
        group["signed_qty"] = signed_qty
        group["cum_pos"] = group["signed_qty"].cumsum()

        start_flags = (group["cum_pos"].shift(fill_value=0) == 0) & (group["cum_pos"] != 0)
        group["round_id"] = start_flags.cumsum()
        return group

    df = df.groupby("symbol", group_keys=False).apply(assign_round_id)

    rows = []
    details_map = {}

    for (symbol, rid), g in df.groupby(["symbol", "round_id"]):
        if g.empty:
            continue

        status = "OPEN" if g["cum_pos"].iloc[-1] != 0 else "CLOSED"

        buys = g[g["type"] == "BUY"]
        if buys.empty:
            continue

        entry_time = buys["time"].iloc[0]
        exit_time = g["time"].iloc[-1]

        entry_qty = buys["qty"].sum()
        entry_price = (buys["price"] * buys["qty"]).sum() / entry_qty

        realized_profit_pct = g["profit"].fillna(0).sum()

        round_key = f"{symbol}__{int(rid)}"

        # ▼ 이 포지션에 속한 개별 체결들 리스트
        detail_rows = []
        for _, row in g.iterrows():
            ml_val = None
            if "ml_proba" in g.columns and pd.notna(row.get("ml_proba", None)):
                ml_val = float(row["ml_proba"])
            detail_rows.append(
                {
                    "time": row["time"],
                    "type": row["type"],
                    "price": float(row["price"]),
                    "qty": int(row["qty"]),
                    "ml_proba": ml_val,
                }
            )

        details_map[round_key] = detail_rows

        rows.append(
            {
                "symbol": symbol,
                "round_id": int(rid),
                "status": status,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "entry_qty": int(entry_qty),
                "entry_price": float(entry_price),
                "realized_profit_pct": float(realized_profit_pct),
                "entry_comment": None,
                "exit_comment": None,
                "date": entry_time.strftime("%Y-%m-%d"),
            }
        )

    return pd.DataFrame(rows), details_map

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
    """
    앱 체결내역을 trading.db로 동기화하는 라우트
    """
    try:
        # import_app_fills.py 실행
        result = subprocess.run(
            ["python", "import_app_fills.py"],
            capture_output=True,
            text=True
        )

        # 에러 체크
        if result.returncode != 0:
            return jsonify({
                "status": "error",
                "message": result.stderr
            }), 500

        return jsonify({
            "status": "ok",
            "message": result.stdout
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500
    
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

    # 시간형으로 변환
    candles["dt"] = pd.to_datetime(candles["dt"])
    if not trades.empty:
        trades["time"] = pd.to_datetime(trades["time"])

    # 각 트레이드를 어느 캔들 인덱스에 표시할지 계산
    candle_times = candles["dt"].values  # numpy array
    trade_rows = []
    if not trades.empty:
        for _, row in trades.iterrows():
            tt = row["time"].to_datetime64()
            # dt <= trade_time 인 마지막 캔들 위치
            pos = candle_times.searchsorted(tt, side="right") - 1
            if pos < 0 or pos >= len(candle_times):
                continue  # 범위 밖이면 스킵

            trade_rows.append(
                {
                    "x_index": int(pos),
                    "time": row["time"].strftime("%Y-%m-%d %H:%M:%S"),
                    "type": row["type"],   # "BUY" / "SELL"
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

@app.route("/api/ai-report/full")
def api_ai_report_full():
    """
    최신 AI 리포트(일일 트레이드 리포트 + 전략 아이디어) +
    최신 모델 조언 텍스트까지 한 번에 내려주는 엔드포인트.
    프론트에서 3단 레이아웃으로 쓰는 용도.
    """
    # 기본 응답 뼈대
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
            # 파일명 또는 수정시간 기준 최신 선택
            latest_file = max(files, key=os.path.getmtime)

            # 파일명에서 날짜 추출 시도 (예: 'reports/2025-11-27_model_advice.txt')
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

    # 종목 드롭다운용:
    #  - trades 테이블에서 실제 거래가 있었던 심볼만
    #  - 마지막(trades.time 기준) 거래 시각 순서대로 정렬
    if not trades.empty:
        # time은 load_trades()에서 이미 datetime으로 변환됨
        last_trade_by_symbol = (
            trades.groupby("symbol")["time"]
            .max()                              # 각 심볼의 마지막 거래 시각
            .sort_values(ascending=False)       # 최근 거래 순으로 정렬
        )
        symbols_with_data = last_trade_by_symbol.index.tolist()
    else:
        # 혹시 trades가 완전 비어 있을 때는 ohlcv 기준으로라도 보여주기 (백업)
        conn = sqlite3.connect(DB_PATH)
        df_sym_ohlcv = pd.read_sql_query(
            "SELECT DISTINCT symbol FROM ohlcv_data ORDER BY symbol", conn
        )
        conn.close()
        symbols_with_data = df_sym_ohlcv["symbol"].tolist()
    

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
    )


if __name__ == "__main__":
    app.run(debug=True, port=8000)
