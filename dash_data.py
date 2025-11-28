# dash_data.py
import sqlite3
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

DB_PATH = "trading.db"


# -----------------------------
# 기본 로딩 함수들
# -----------------------------
def load_trades() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM trades ORDER BY time", conn)
    conn.close()
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df


def load_logs() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM logs ORDER BY time DESC", conn)
    conn.close()
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df


def load_signals(limit: int = 200) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        f"SELECT * FROM signals ORDER BY time DESC LIMIT {int(limit)}",
        conn,
    )
    conn.close()
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df


def load_model_versions(limit: int = 20) -> pd.DataFrame:
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


def load_backtests(limit: int = 50) -> pd.DataFrame:
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


def load_signals_by_date(target_date: str) -> pd.DataFrame:
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


def load_trades_by_date(target_date: str) -> pd.DataFrame:
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


# -----------------------------
# 라운드 트립(포지션 단위) 집계
# -----------------------------
def build_round_trades(df_trades: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[dict]]]:
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

        # ▼ 이 포지션에 속한 개별 체결들 리스트
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


# -----------------------------
# ML 관련 헬퍼
# -----------------------------
def load_ml_signals(limit: int = 500) -> pd.DataFrame:
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


def suggest_improvements(
    df_sig: pd.DataFrame,
    df_tr: pd.DataFrame,
    ml_threshold: float = 0.55,
):
    """
    ML / 룰 / 트레이드 결과를 기반으로 한 텍스트 코멘트 생성
    (기존 web_app.py 의 suggest_improvements 그대로 이동)
    """
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
# 드롭다운용 심볼 리스트 헬퍼
# -----------------------------
def get_symbols_with_data(trades: pd.DataFrame) -> List[str]:
    """
    대시보드 종목 드롭다운용 심볼 리스트 생성
    - trades가 있으면: 마지막 트레이드 시각 기준 정렬
    - trades가 비어 있으면: ohlcv_data 기준으로 심볼 목록 생성
    """
    if not trades.empty:
        last_trade_by_symbol = (
            trades.groupby("symbol")["time"]
            .max()
            .sort_values(ascending=False)
        )
        return last_trade_by_symbol.index.tolist()

    conn = sqlite3.connect(DB_PATH)
    df_sym_ohlcv = pd.read_sql_query(
        "SELECT DISTINCT symbol FROM ohlcv_data ORDER BY symbol", conn
    )
    conn.close()
    return df_sym_ohlcv["symbol"].tolist()
