# "Supabase 기반 트레이딩 대시보드/리포트용 DB 헬퍼 (v1)

#  - Supabase(PostgreSQL)에 직접 접속해서 trades / logs / signals / models / backtests 테이블을
#    DataFrame으로 불러오고, 라운드트립 집계·ML 시그널 분석·심볼 리스트 추출까지 담당하는 분석 전용 모듈

# 주요 기능:
# 1) DB 연결 설정
#    - Supabase 호스트/DB명/유저/비밀번호/포트 상수 정의
#    - get_connection(): psycopg2로 PostgreSQL 커넥션 생성 (각 쿼리 함수에서 try/finally로 닫기)

# 2) 기본 조회 함수 (DataFrame 반환)
#    - load_trades(): trades 전부를 time 기준 오름차순으로 로드, time 컬럼을 datetime으로 변환
#    - load_logs(): logs를 time DESC로 로드, time 컬럼을 datetime으로 변환
#    - load_signals(limit): signals를 최신 순으로 지정 개수만 로드
#    - load_model_versions(limit): models 테이블에서 id/created_at/path/n_samples/val_accuracy 조회
#    - load_backtests(limit): backtests 테이블에서 주요 지표(id, model_id, 기간, 트레이드 수, 승률, 수익률, MDD, note 등)를 조회
#    - load_signals_by_date(date), load_trades_by_date(date):
#      · time::date = target_date 조건으로 일자별 신호/체결 내역만 필터링해서 로드

# 3) 라운드 트립(포지션 단위) 집계 (build_round_trades)
#    - trades DataFrame을 받아서 심볼별 포지션 단위(한 번 진입~완전 청산까지)를 그룹핑
#    - 내장 로직:
#      · type=="BUY" → +qty, 그 외 → -qty 로 signed_qty 생성
#      · signed_qty 누적합(cum_pos)으로 포지션 시작/종료를 판단해 round_id 부여
#    - 각 (symbol, round_id) 그룹에 대해:
#      · status: 마지막 cum_pos가 0이면 "CLOSED", 아니면 "OPEN"
#      · entry_time: 첫 BUY 체결 시간
#      · exit_time: 해당 라운드의 마지막 체결 시간
#      · entry_qty / entry_price: 매수 수량 총합과 가중 평균 매수가
#      · realized_profit_pct: profit 컬럼 합(%)으로 실현 수익률 집계
#      · entry_comment: 첫 BUY 행의 entry_comment를 진입 코멘트로 사용
#      · exit_comment: 마지막 SELL/청산 행의 exit_comment를 청산 코멘트로 사용
#    - 반환값:
#      · 요약 DataFrame (포지션별 한 행)
#      · details_map: "SYMBOL__round_id" 키로, 해당 라운드의 개별 체결(시간/type/price/qty/ml_proba) 리스트

# 4) ML 관련 헬퍼
#    - load_ml_signals(limit):
#      · signals 테이블에서 time, symbol, ml_proba, entry_signal, entry_allowed만 가져와
#        시간 순으로 정렬해 ML 시그널 분석에 사용
#    - suggest_improvements(df_sig, df_tr, ml_threshold):
#      · 시그널/트레이드 데이터를 기반으로 설정 튜닝 가이드를 생성하는 추천 엔진
#      · 주요 분석 포인트:
#        · 총 신호 수, 룰 기반 entry_signal 횟수, ML 통과(entry_allowed) 횟수
#        · ml_proba 분포: threshold 이상 비율(hi_ratio), 평균 확률(mean_proba)
#        · 지나치게 엄격/느슨한 threshold에 대한 조언
#        · 실제 트레이드 성과: 승률, 평균 수익률, 트레이드 수
#        · 가장 신호가 많이 나온 심볼과 횟수
#      · 그 결과를 한국어 설명이 담긴 문자열 리스트(suggestions)로 반환

# 5) 심볼 리스트 헬퍼
#    - get_symbols_with_data(trades):
#      · 인자로 받은 trades DataFrame이 비어있지 않으면:
#        · 심볼별 마지막 트레이드 시간을 기준으로 정렬해 심볼 리스트 반환
#      · trades가 비어 있으면:
#        · DB에서 ohlcv_data 테이블의 DISTINCT symbol 목록을 로딩 후 리스트로 반환
#    - 대시보드/리포트 UI에서 ‘데이터가 있는 심볼들’ 셀렉박스 등에 쓰기 좋은 헬퍼"


import psycopg2
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# -----------------------------------------------------------
# [중요] 여기에 Supabase 접속 정보를 나눠서 입력하세요.
# (URL에서 @ 뒤에 있는 주소가 HOST입니다)
# -----------------------------------------------------------
DB_HOST = "aws-1-ap-northeast-2.pooler.supabase.com"
DB_NAME = "postgres"
DB_USER = "postgres.sxhtnkxulfrqykrtwxjx"  # [주의] 아이디가 이렇게 길어집니다
DB_PASS = "Shitdog205!@"                     # 기존 비밀번호 그대로
DB_PORT = "6543"                             # [주의] 포트가 6543입니다
# -----------------------------------------------------------

# 호환성용 변수 (무시하셔도 됩니다)
DB_PATH = "trading.db"
DB_URL = "" 

def get_connection():
    """Supabase DB 연결 (안전한 방식)"""
    return psycopg2.connect(
        host=DB_HOST,
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        port=DB_PORT
    )

# -----------------------------
# 기본 로딩 함수들
# -----------------------------
def load_trades() -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM trades ORDER BY time", conn)
        if not df.empty and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        return df
    finally:
        conn.close()


def load_logs() -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query("SELECT * FROM logs ORDER BY time DESC", conn)
        if not df.empty and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        return df
    finally:
        conn.close()


def load_signals(limit: int = 200) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            f"SELECT * FROM signals ORDER BY time DESC LIMIT {int(limit)}",
            conn,
        )
        if not df.empty and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        return df
    finally:
        conn.close()


def load_model_versions(limit: int = 20) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT id, created_at, path, n_samples, val_accuracy
            FROM models
            ORDER BY created_at DESC
            LIMIT %s
            """,
            conn,
            params=(int(limit),),
        )
        if not df.empty and "created_at" in df.columns:
            df["created_at"] = pd.to_datetime(df["created_at"])
        return df
    finally:
        conn.close()


def load_backtests(limit: int = 50) -> pd.DataFrame:
    conn = get_connection()
    try:
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
        return df
    finally:
        conn.close()


def load_signals_by_date(target_date: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM signals
            WHERE time::date = %s
            ORDER BY time
            """,
            conn,
            params=(target_date,),
        )
        if not df.empty and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        return df
    finally:
        conn.close()


def load_trades_by_date(target_date: str) -> pd.DataFrame:
    conn = get_connection()
    try:
        df = pd.read_sql_query(
            """
            SELECT *
            FROM trades
            WHERE time::date = %s
            ORDER BY time
            """,
            conn,
            params=(target_date,),
        )
        if not df.empty and "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        return df
    finally:
        conn.close()


# -----------------------------
# 라운드 트립(포지션 단위) 집계 [수정됨]
# -----------------------------
def build_round_trades(df_trades: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[dict]]]:
    if df_trades.empty:
        return pd.DataFrame(), {}

    df = df_trades.sort_values("time").copy()

    if "type" not in df.columns:
        return pd.DataFrame(), {}

    def assign_round_id(group: pd.DataFrame) -> pd.DataFrame:
        signed_qty = np.where(group["type"] == "BUY", group["qty"], -group["qty"])
        group["signed_qty"] = signed_qty
        group["cum_pos"] = group["signed_qty"].cumsum()

        start_flags = (group["cum_pos"].shift(fill_value=0) == 0) & (group["cum_pos"] != 0)
        group["round_id"] = start_flags.cumsum()
        return group

    df = df.groupby("symbol", group_keys=False).apply(assign_round_id)

    rows = []
    details_map: Dict[str, List[dict]] = {}

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

        detail_rows: List[dict] = []
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

        # ✅ [수정] 코멘트 가져오기 로직 추가
        # g: 해당 라운드(포지션 시작~끝)에 속한 모든 트레이드 행들
        
        # 1) 진입 코멘트: 가장 첫 매수(BUY)의 코멘트를 가져옴
        entry_comment = None
        if "entry_comment" in buys.columns:
             # 첫 번째 매수 행의 코멘트
             val = buys.iloc[0]["entry_comment"]
             if pd.notna(val) and val:
                 entry_comment = str(val)

        # 2) 청산 코멘트: 가장 마지막 매도/청산 행의 코멘트
        exit_comment = None
        # 매도/청산 타입이 있는 행들만 필터링 (BUY가 아닌 것)
        sells = g[g["type"] != "BUY"]
        if not sells.empty and "exit_comment" in sells.columns:
            val = sells.iloc[-1]["exit_comment"]
            if pd.notna(val) and val:
                exit_comment = str(val)

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
                "entry_comment": entry_comment,  # ✅ 수정됨
                "exit_comment": exit_comment,    # ✅ 수정됨
                "date": entry_time.strftime("%Y-%m-%d"),
            }
        )

    return pd.DataFrame(rows), details_map


# -----------------------------
# ML 관련 헬퍼
# -----------------------------
def load_ml_signals(limit: int = 500) -> pd.DataFrame:
    conn = get_connection()
    try:
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
        if df.empty:
            return df

        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        return df.sort_values("time")
    finally:
        conn.close()


def suggest_improvements(
    df_sig: pd.DataFrame,
    df_tr: pd.DataFrame,
    ml_threshold: float = 0.55,
):
    suggestions = []

    if df_sig.empty:
        suggestions.append(
            "📉 오늘 저장된 신호가 없습니다. 타겟 종목 수나 장시간이 너무 짧지 않은지 점검해 보세요."
        )
        return suggestions

    total_signals = len(df_sig)
    rule_signals = int(df_sig["entry_signal"].fillna(0).sum())
    
    if "entry_allowed" in df_sig.columns:
        allowed = int(df_sig["entry_allowed"].fillna(0).sum())
    else:
        allowed = 0

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

    if not df_tr.empty:
        realized = df_tr["profit"].dropna()
        realized = realized[realized != 0]
        num_trades = len(realized)
        if num_trades > 0:
            wins = (realized > 0).sum()
            win_rate = wins / num_trades
            avg_profit = realized.mean()

            suggestions.append(
                f"💰 오늘 체결된 트레이드는 {num_trades}건, 승률 {win_rate*100:.1f}%, "
                f"트레이드당 평균 수익률 {avg_profit:.2f}% 입니다."
            )

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


def get_symbols_with_data(trades: pd.DataFrame) -> List[str]:
    if not trades.empty:
        last_trade_by_symbol = (
            trades.groupby("symbol")["time"]
            .max()
            .sort_values(ascending=False)
        )
        return last_trade_by_symbol.index.tolist()

    conn = get_connection()
    try:
        df_sym_ohlcv = pd.read_sql_query(
            "SELECT DISTINCT symbol FROM ohlcv_data ORDER BY symbol", conn
        )
        return df_sym_ohlcv["symbol"].tolist()
    finally:
        conn.close()