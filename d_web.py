# "트레이딩 대시보드 웹 서버 (Flask / Supabase / 공용 리포트)

#  - Flask 기반 웹 앱으로, Supabase(Postgres)에 저장된 trades / logs / signals / models / backtests / ohlcv_data
#    를 읽어와 여러 페이지(요약, Universe, ML 모니터링, 심볼 차트, 자동매매 로그, AI 리포트)를 렌더링하는 대시보드 서버

# 주요 기능:
# 1) DB 설정 및 공용 데이터 로딩
#    - Supabase 접속 정보(DB_HOST/DB_NAME/DB_USER/DB_PASS/DB_PORT) 정의 후, 비밀번호를 URL에 안전하게 넣기 위해 quote_plus로 인코딩
#    - DB_URL을 만들어 BotDatabase(DB_URL)와 연동
#    - common.dash_data 모듈의 get_connection / load_trades / load_logs / load_signals / load_model_versions /
#      load_backtests / load_signals_by_date / load_trades_by_date / build_round_trades / load_ml_signals /
#      suggest_improvements / get_symbols_with_data 를 재사용
#    - dash_universe_ohlcv 의 get_universe_coverage / get_last_universe_backfill_time / get_recent_backfill_failures 를 사용해
#      Universe OHLCV 상태와 백필 실패 내역 로딩

# 2) 공통 유틸 + 대시보드용 집계 (load_all_dashboard_data)
#    - normalize_region(region_raw): ALL/KR/US/CR 중 하나로 정규화
#    - filter_by_region(region, df): region 컬럼 기준으로 DataFrame 필터링
#    - load_all_dashboard_data(region_raw):
#      · trades/logs/signals/model_versions/backtests/universe_cov/universe_failures 를 모두 로드 후 region별 필터 적용
#      · trades 기준 성과 요약(summary): 총 트레이드 수, 승률, 평균 수익률, 누적 수익률(cum_return_pct)
#      · 에쿼티 커브(equity_curve): 시간/누적 수익률(%) 시계열
#      · symbols_avg: 심볼별 평균 수익률 랭킹
#      · daily_summaries: 날짜별 트레이드 수, 승률, 평균 수익률, 일별 누적 수익률
#      · round_trades_df / round_details: build_round_trades로 포지션 단위 라운드 트레이드 구조 생성
#      · Universe 통계: 종목 수, 총 캔들 수, 최대 커버 일수
#      · 오늘(or 최근) 신호/트레이드를 기반으로 suggest_improvements 호출 → 설정/전략 개선에 대한 한국어 피드백 리스트(suggestions)
#      · ML 모니터링용: ml_signals 로부터
#        - histogram(ml_hist_labels / ml_hist_counts),
#        - 시계열(ml_time_series: time / proba / entry_allowed)
#      · 심볼 선택용: get_symbols_with_data 로 symbols_with_data 리스트 반환

# 3) 앱 체결 동기화 관련 API
#    - /sync-app-trades (POST): 현재는 stub(sync_app_trades → 0) 형태, 향후 앱 체결 데이터 연동용 확장 포인트
#    - /sync_app_fills (POST): zz_import_app_fills.sync_app_fills_main() 호출하여 앱 체결 내역을 DB에 동기화하는 엔드포인트

# 4) 심볼 차트용 데이터 API (/symbol_data)
#    - 쿼리스트링 symbol(필수), region(옵션)을 받고
#    - ohlcv_data에서 해당 심볼의 5분봉 캔들(dt, open/high/low/close)를 최대 500개 로드
#    - trades 테이블에서 같은 심볼의 체결(time, type, price, qty)를 로드
#    - 캔들 시계열(dt) 기준으로 각 트레이드가 어느 캔들 인덱스(x_index)에 해당하는지 매핑
#    - candles 리스트(시각/가격)와 trades 리스트(캔들 인덱스/시각/type/price/qty)를 JSON으로 반환 → 프론트 차트에서 매수/매도 마커 표시용

# 5) AI 리포트 API (/api/ai-report/full)
#    - ai_reports 테이블에서 최신 1건(date, created_at, daily_report, strategy_ideas) 조회
#    - 로컬 reports 폴더의 *_model_advice.txt 중 최신 파일을 찾아 model_advice_date + model_advice 텍스트 로드
#    - 이 둘을 합쳐 일간 AI 리포트 + 모델 조언을 한 번에 반환하는 JSON API

# 6) 페이지 라우트 (Flask 템플릿 렌더)
#    - 루트(/): region 쿼리 파라미터를 정규화 후 /dash/overview로 리다이렉트
#    - /dash/overview:
#      · 요약/성능 페이지: summary, equity_curve, symbols_avg, suggestions 렌더
#    - /dash/ai-report:
#      · AI 리포트 페이지: region만 넘기고 실제 리포트 내용은 프론트에서 API를 호출해 사용할 수 있게 설계 가능
#    - /dash/trades, /dash/logs:
#      · 각각 /dash/auto로 region만 바꿔 리다이렉트 (자동매매 탭에서 통합 관리)
#    - /dash/universe:
#      · Universe OHLCV 탭: universe_cov, last_universe_backfill, universe_failures, universe 통계 숫자 렌더
#    - /dash/ml:
#      · ML 모니터링 탭: ML 히스토그램/시계열, 모델 버전 리스트, 백테스트 결과 렌더
#    - /dash/symbols:
#      · 종목별 차트 탭: symbols_with_data 리스트를 템플릿에 전달 → 사용자 선택 후 /symbol_data API로 차트 그리기
#    - /dash/auto:
#      · 자동매매 탭:
#        - round_trades / round_details: 포지션 단위 내역
#        - daily_summaries / summary: 일별/전체 성과 요약
#        - logs_recent / signals: 최근 로그와 엔트리/ML 신호 목록
#        - universe_failures: 백필 실패 목록까지 통합 표시

# 7) 개발 서버 실행
#    - __main__ 블록에서 app.run(debug=True, port=8000)으로 로컬 개발용 실행

from flask import Flask, render_template, jsonify, request, redirect
import os
import psycopg2
import pandas as pd
import numpy as np
from glob import glob
from datetime import date
from urllib.parse import quote_plus  # [추가] 특수문자 비번 처리를 위해 필요

# 기존 파일들 import
from c_db_manager import BotDatabase
from d_web_universe_ohlcv import (
    get_universe_coverage,
    get_last_universe_backfill_time,
    get_recent_backfill_failures,
)
# dash_data에서 get_connection을 가져와서 재사용합니다
from d_web_data import (
    get_connection, 
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

# -----------------------------------------------------------
# [중요] dash_data.py와 똑같이 정보를 입력하세요
# -----------------------------------------------------------
DB_HOST = "aws-1-ap-northeast-2.pooler.supabase.com"
DB_NAME = "postgres"
DB_USER = "postgres.sxhtnkxulfrqykrtwxjx"  # [주의] 아이디가 이렇게 길어집니다
DB_PASS = "Shitdog205!@"                     # 기존 비밀번호 그대로
DB_PORT = "6543"                             # [주의] 포트가 6543입니다
# -----------------------------------------------------------

# [핵심] 특수문자가 포함된 비밀번호를 안전하게 변환하여 DB_URL 생성
# (BotDatabase 클래스가 이 URL을 필요로 함)
encoded_pass = quote_plus(DB_PASS)
DB_URL = f"postgresql://{DB_USER}:{encoded_pass}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# -----------------------------
# 공통 유틸
# -----------------------------
def normalize_region(region_raw: str) -> str:
    region = (region_raw or "ALL").upper()
    if region not in ("ALL", "KR", "US", "CR","BI"):
        region = "ALL"
    return region


def filter_by_region(region: str, df):
    if df is None or len(getattr(df, "columns", [])) == 0:
        return df
    if "region" not in df.columns:
        return df
    if region == "ALL":
        return df
    return df[df["region"] == region].copy()


def load_all_dashboard_data(region_raw: str):
    """
    모든 페이지에서 재사용할 공통 데이터 로딩/집계 함수
    """
    region = normalize_region(region_raw)

    trades = load_trades()
    logs = load_logs()
    signals = load_signals(limit=200)
    model_versions = load_model_versions()
    backtests = load_backtests(limit=50)

    universe_cov = get_universe_coverage()
    
    # [수정] 안전하게 만든 DB_URL을 사용하여 BotDatabase 연결
    last_universe_backfill = get_last_universe_backfill_time(db=BotDatabase(DB_URL))
    
    universe_failures = get_recent_backfill_failures(limit=30)

    # region 필터 적용
    trades = filter_by_region(region, trades)
    logs = filter_by_region(region, logs)
    signals = filter_by_region(region, signals)
    model_versions = filter_by_region(region, model_versions)
    backtests = filter_by_region(region, backtests)
    universe_cov = filter_by_region(region, universe_cov)
    universe_failures = filter_by_region(region, universe_failures)

    if not universe_cov.empty:
        universe_cov = universe_cov.sort_values("candles", ascending=False)

    # 라운드 트레이드
    if region == "BI":
        # 🔹 Binance는 positions 기준으로 리스트 생성
        round_trades_df = build_round_trades_from_positions(region)
        round_details = {}  # BI는 상세 구조 아직 안 쓰면 빈 dict
    else:
        # 기존 KR/US/CR은 trades 기반 라운드 로직 그대로 유지
        if not trades.empty:
            round_trades_df, round_details = build_round_trades(trades)
        else:
            round_trades_df, round_details = pd.DataFrame(), {}

    # Universe 숫자
    if not universe_cov.empty:
        num_universe_symbols = int(len(universe_cov))
        total_universe_candles = int(universe_cov["candles"].sum())
        max_days_covered = int(universe_cov["days_covered"].max())
    else:
        num_universe_symbols = 0
        total_universe_candles = 0
        max_days_covered = 0

    # 요약/에쿼티/심볼 평균/일별 요약
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
        last_cum = float(cum_return.iloc[-1]) if len(cum_return) > 0 else 0.0

        summary = {
            "total_trades": total_trades,
            "win_rate": round(win_rate, 2),
            "avg_profit": round(avg_profit, 2),
            "cum_return_pct": round(last_cum * 100, 2),
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

    # 오늘 기준 신호/트레이드
    today_str = date.today().strftime("%Y-%m-%d")
    today_signals = filter_by_region(region, load_signals_by_date(today_str))
    today_trades = filter_by_region(region, load_trades_by_date(today_str))

    if today_signals.empty:
        today_signals = filter_by_region(region, load_signals(limit=200))

    suggestions = suggest_improvements(
        df_sig=today_signals,
        df_tr=today_trades,
        ml_threshold=0.55,
    )

    # ML 모니터링용
    ml_signals = filter_by_region(region, load_ml_signals(limit=500))

    ml_hist_labels = []
    ml_hist_counts = []
    ml_time_series = []

    if not ml_signals.empty:
        bins = np.linspace(0, 1, 11)
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

    symbols_with_data = get_symbols_with_data(trades)

    return {
        "region": region,
        "summary": summary,
        "equity_curve": equity_curve,
        "symbols_avg": symbols_avg,
        "trades": trades,
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

def build_round_trades_from_positions(region: str):
    """
    positions 테이블 기준으로 dash_auto.html에서 쓰는 round_trades 형식으로 변환
    """
    conn = get_connection()
    try:
        if region == "ALL":
            df = pd.read_sql_query(
                """
                SELECT
                    id,
                    region,
                    symbol,
                    side,
                    qty,
                    entry_price,
                    entry_time,
                    closed_at,
                    is_open,
                    last_roi
                FROM positions
                ORDER BY entry_time DESC
                """,
                conn,
            )
        else:
            df = pd.read_sql_query(
                """
                SELECT
                    id,
                    region,
                    symbol,
                    side,
                    qty,
                    entry_price,
                    entry_time,
                    closed_at,
                    is_open,
                    last_roi
                FROM positions
                WHERE region = %s
                ORDER BY entry_time DESC
                """,
                conn,
                params=(region,),
            )
    finally:
        conn.close()

    if df.empty:
        # round_trades_df, round_details 형식을 맞춰주기 위해 튜플 리턴
        return pd.DataFrame(), {}

    df["entry_time"] = pd.to_datetime(df["entry_time"])
    if "closed_at" in df.columns:
        df["closed_at"] = pd.to_datetime(df["closed_at"])

    rows = []
    for _, row in df.iterrows():
        entry_time = row.get("entry_time")
        exit_time = row.get("closed_at")
        side_raw = (row.get("side") or "").upper()
        qty = float(row.get("qty") or 0.0)
        entry_price = float(row.get("entry_price") or 0.0)

        status = "OPEN" if row.get("is_open") else "CLOSED"

        # data-date 에 쓸 날짜 (진입일 기준)
        if pd.notna(entry_time):
            date_str = entry_time.strftime("%Y-%m-%d")
        else:
            date_str = ""

        # last_roi: 청산 시점 실현 수익률(%) 라고 가정
        roi = row.get("last_roi")
        try:
            realized_pct = float(roi) if roi is not None else 0.0
        except Exception:
            realized_pct = 0.0

        rows.append({
            "round_id": int(row.get("id")) if not pd.isna(row.get("id")) else None,
            "symbol": row.get("symbol"),
            "region": row.get("region"),

            "date": date_str,                            # ✅ data-date 용
            "entry_time": entry_time,                    # ✅ 그대로 찍어도 됨
            "exit_time": exit_time,                      # ✅ 없으면 None

            "status": status,                            # ✅ OPEN / CLOSED
            "entry_qty": qty,
            "entry_price": entry_price,
            "realized_profit_pct": realized_pct,         # ✅ "%.2f"|format() 대상

            # 디테일 행에서 쓰지만 없어도 자동으로 비어 나옴
            "entry_comment": None,
            "exit_comment": None,
        })

    round_trades_df = pd.DataFrame(rows)
    round_details = {}  # BI에선 아직 세부 체결 로그 안 쓰므로 빈 dict

    return round_trades_df, round_details

# -----------------------------
# 앱 체결 동기화 라우트
# -----------------------------
def sync_app_trades():
    return 0

@app.route("/sync-app-trades", methods=["POST"])
def sync_app_trades_route():
    try:
        inserted = sync_app_trades()
        return jsonify({"ok": True, "inserted": int(inserted)})
    except Exception as e:
        print("sync_app_trades 오류:", e)
        return jsonify({"ok": False, "error": str(e)}), 500




# -----------------------------
# 심볼 차트용 데이터 API (get_connection 재사용)
# -----------------------------
@app.route("/symbol_data")
def symbol_data():
    symbol = request.args.get("symbol")
    region_raw = request.args.get("region", "ALL")  # 예: KR, US, CR, BI ...
    if not symbol:
        return jsonify({"error": "symbol parameter required"}), 400

    region = (region_raw or "ALL").upper()

    conn = get_connection()
    try:
        # 1) 캔들 로드 (기존과 동일)
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

        # 2) 포지션 기반 마커 로드
        #    - entry: entry_time / entry_price
        #    - exit : closed_at / exit_price (닫힌 포지션만)
        if region == "ALL":
            # region 상관없이 해당 심볼의 모든 포지션
            positions = pd.read_sql_query(
                """
                SELECT
                    region,
                    side,
                    qty,
                    entry_time,
                    entry_price,
                    closed_at,
                    exit_price,
                    is_open
                FROM positions
                WHERE symbol = %s
                  AND exchange = 'BINANCE'
                  AND market_type IN ('spot', 'futures')
                ORDER BY entry_time
                """,
                conn,
                params=(symbol,),
            )
        else:
            positions = pd.read_sql_query(
                """
                SELECT
                    region,
                    side,
                    qty,
                    entry_time,
                    entry_price,
                    closed_at,
                    exit_price,
                    is_open
                FROM positions
                WHERE symbol = %s
                  AND exchange = 'BINANCE'
                  AND region = %s
                  AND market_type IN ('spot', 'futures')
                ORDER BY entry_time
                """,
                conn,
                params=(symbol, region),
            )

    finally:
        conn.close()

    # 캔들이 없으면 바로 빈 값 리턴
    if candles.empty:
        return jsonify({"candles": [], "trades": []})

    candles["dt"] = pd.to_datetime(candles["dt"])

    # 3) 캔들 리스트 구성 (기존과 동일)
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

    # 4) positions → 엔트리/청산 마커로 변환
    trade_rows = []
    if not positions.empty:
        positions["entry_time"] = pd.to_datetime(positions["entry_time"])
        if "closed_at" in positions.columns:
            positions["closed_at"] = pd.to_datetime(positions["closed_at"])

        candle_times = candles["dt"].values

        # (1) 엔트리 마커
        for _, row in positions.iterrows():
            et = row.get("entry_time")
            ep = row.get("entry_price")
            qty = row.get("qty", 0)
            side = (row.get("side") or "").upper()

            if pd.isna(et) or ep is None:
                continue

            # LONG/BUY → BUY 마커, SHORT → SELL 마커
            if side in ("LONG", "BUY"):
                m_type = "BUY"
            elif side in ("SHORT", "SELL"):
                m_type = "SELL"
            else:
                m_type = "BUY"

            tt = et.to_datetime64()
            pos = candle_times.searchsorted(tt, side="right") - 1
            if pos < 0 or pos >= len(candle_times):
                continue

            trade_rows.append(
                {
                    "x_index": int(pos),
                    "time": et.strftime("%Y-%m-%d %H:%M:%S"),
                    "type": m_type,
                    "price": float(ep),
                    "qty": float(abs(qty) if qty is not None else 0.0),
                }
            )

        # (2) 청산 마커 (닫힌 포지션 + exit_price 있는 것만)
        for _, row in positions.iterrows():
            ct = row.get("closed_at")
            xp = row.get("exit_price")
            qty = row.get("qty", 0)
            side = (row.get("side") or "").upper()
            is_open = row.get("is_open")

            if is_open:  # 아직 열려있는 포지션이면 청산 마커 없이 패스
                continue
            if pd.isna(ct) or xp is None:
                continue

            # 엔트리의 반대 방향으로 표시
            if side in ("LONG", "BUY"):
                m_type = "SELL"
            elif side in ("SHORT", "SELL"):
                m_type = "BUY"
            else:
                m_type = "SELL"

            tt = ct.to_datetime64()
            pos = candle_times.searchsorted(tt, side="right") - 1
            if pos < 0 or pos >= len(candle_times):
                continue

            trade_rows.append(
                {
                    "x_index": int(pos),
                    "time": ct.strftime("%Y-%m-%d %H:%M:%S"),
                    "type": m_type,
                    "price": float(xp),
                    "qty": float(abs(qty) if qty is not None else 0.0),
                }
            )

    # 시간 순으로 정렬 (엔트리/청산 섞여 있으므로)
    trade_rows.sort(key=lambda x: x["time"])

    return jsonify(
        {
            "candles": candles_json,
            "trades": trade_rows,
        }
    )


# -----------------------------
# AI 리포트 API (get_connection 재사용)
# -----------------------------
@app.route("/api/ai-report/full")
def api_ai_report_full():
    result = {
        "date": None,
        "created_at": None,
        "daily_report": "",
        "strategy_ideas": "",
        "model_advice_date": None,
        "model_advice": "",
    }

    # 1) ai_reports 테이블에서 최신 1건
    try:
        conn = get_connection()
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

    # 2) reports 폴더의 *_model_advice.txt
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
# 라운트: 페이지들
# -----------------------------
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
    return render_template(
        "dash_ai_report.html",
        page_title="AI 리포트",
        active_page="ai-report",
        region=data["region"],
    )


@app.route("/dash/trades")
def dash_trades():
    region = normalize_region(request.args.get("region", "ALL"))
    return redirect(f"/dash/auto?region={region}")


@app.route("/dash/logs")
def dash_logs():
    region = normalize_region(request.args.get("region", "ALL"))
    return redirect(f"/dash/auto?region={region}")


@app.route("/dash/universe")
def dash_universe():
    region = request.args.get("region", "ALL")
    data = load_all_dashboard_data(region)
    return render_template(
        "dash_universe.html",
        page_title="Universe OHLCV",
        active_page="universe",
        region=data["region"],
        universe_cov=data["universe_cov"].to_dict(orient="records"),
        last_universe_backfill=data["last_universe_backfill"],
        universe_failures=data["universe_failures"].to_dict(orient="records"),
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
        model_versions=data["model_versions"].to_dict(orient="records"),
        backtests=data["backtests"].to_dict(orient="records"),
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
    sorted_round_trades = data["round_trades_df"].sort_values("entry_time", ascending=False)
    return render_template(
        "dash_auto.html",
        page_title="자동매매",
        active_page="auto",
        region=data["region"],
        universe_failures=data["universe_failures"].to_dict(orient="records"),
        # 트레이드용
        round_trades=sorted_round_trades.to_dict(orient="records"),
        round_details=data["round_details"],
        daily_summaries=data["daily_summaries"],
        summary=data["summary"],
        # 로그/신호용
        logs=data["logs_recent"].to_dict(orient="records"),
        signals=data["signals"].to_dict(orient="records"),
        symbols_with_data=data["symbols_with_data"],
    )

if __name__ == "__main__":
    app.run(debug=True, port=8000)  