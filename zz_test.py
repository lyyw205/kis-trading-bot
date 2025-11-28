# backtest_entry_coin_rule.py
"""
코인(CR) 전용 엔트리 룰 (make_entry_signal_coin) 승률 테스트 스크립트

- ohlcv_data 테이블의 CR/5m 캔들을 사용
- SEQ_LEN 윈도우를 굴리면서 make_entry_signal_coin 적용
- 진입 신호가 나온 지점에서 STEP_AHEAD 뒤 종가 기준 수익률 계산
"""

import sqlite3
from collections import defaultdict

import pandas as pd
import numpy as np

from ml_features import SEQ_LEN
from st_entry_cr import make_entry_signal_coin

# ✅ DB 경로 - 네 환경에 맞게 수정
from dash_data import DB_PATH  # 이미 쓰고 있으면 이대로, 아니면 문자열로 바꿔도 됨
# DB_PATH = "trading.db"


# ▷ CR 전용 파라미터 (원하면 튜닝 가능)
CR_PARAMS = {
    "lookback": 20,
    "band_pct": 0.01,
    "atr_max_ratio": 0.025,   # ATR/close 2.5% 이상이면 너무 미쳐돌아가는 구간 → 진입X
    "hl_max_ratio": 0.035,    # (high-low)/close 3.5% 이상인 캔들도 컷
}

# ▷ 몇 캔들 뒤 수익률을 볼지
STEP_AHEAD = 3


def load_cr_symbols(conn):
    """
    ohlcv_data 에서 CR 심볼 목록 가져오기
    region 컬럼이 없다면, DISTINCT symbol 만 써도 됨.
    """
    try:
        df = pd.read_sql_query(
            """
            SELECT DISTINCT symbol
            FROM ohlcv_data
            WHERE interval = '5m'
              AND region = 'CR'
            """,
            conn,
        )
    except Exception:
        # region 컬럼이 없다면 interval 만으로 필터링
        df = pd.read_sql_query(
            """
            SELECT DISTINCT symbol
            FROM ohlcv_data
            WHERE interval = '5m'
            """,
            conn,
        )
    return df["symbol"].tolist()


def load_ohlcv_for_symbol(conn, symbol: str) -> pd.DataFrame:
    """
    특정 심볼의 5분봉 전체 로드
    """
    df = pd.read_sql_query(
        """
        SELECT dt, open, high, low, close, volume
        FROM ohlcv_data
        WHERE symbol = ?
          AND interval = '5m'
        ORDER BY datetime(dt)
        """,
        conn,
        params=(symbol,),
    )
    if df.empty:
        return df

    df["dt"] = pd.to_datetime(df["dt"])
    df = df.rename(columns={"dt": "time"})
    return df


def backtest_symbol(df: pd.DataFrame, symbol: str):
    """
    단일 심볼에 대해:
      - SEQ_LEN 창을 굴려가며 make_entry_signal_coin 적용
      - entry_signal True인 지점의 STEP_AHEAD 수익률 계산

    return:
      - list of dicts: 개별 신호 결과
    """
    results = []

    if len(df) < SEQ_LEN + STEP_AHEAD + 1:
        return results

    # 인덱스를 깔끔하게 0..N-1 로
    df = df.reset_index(drop=True)

    # 반복: i = 시그널 판단 시점 index
    # i번째 캔들까지 포함된 구간: df[i-SEQ_LEN+1 : i+1] 길이 = SEQ_LEN
    for i in range(SEQ_LEN, len(df) - STEP_AHEAD):
        window = df.iloc[i - SEQ_LEN : i].copy()

        # st_entry_coin은 "최근 SEQ_LEN 개 전체 df"를 기대하므로
        # window를 그대로 넘김
        signal = make_entry_signal_coin(
            window[["open", "high", "low", "close", "volume"]],
            CR_PARAMS,
        )

        if not signal.get("entry_signal", False):
            continue

        # 진입/청산 시점
        entry_idx = i - 1  # window 마지막 캔들 = 실질 시점
        exit_idx = entry_idx + STEP_AHEAD
        if exit_idx >= len(df):
            continue  # 미래 캔들이 없으면 스킵

        entry_row = df.iloc[entry_idx]
        exit_row = df.iloc[exit_idx]

        entry_price = float(entry_row["close"])
        exit_price = float(exit_row["close"])
        if entry_price <= 0:
            continue

        ret_pct = (exit_price / entry_price - 1.0) * 100.0

        results.append(
            {
                "symbol": symbol,
                "entry_time": entry_row["time"],
                "exit_time": exit_row["time"],
                "entry_price": entry_price,
                "exit_price": exit_price,
                "ret_pct": ret_pct,
                "strategy_name": signal.get("strategy_name", "UNKNOWN"),
                "note": signal.get("note", ""),
            }
        )

    return results


def main():
    conn = sqlite3.connect(DB_PATH)

    symbols = load_cr_symbols(conn)
    print(f"🔍 CR 테스트 대상 심볼 수: {len(symbols)}")

    all_results = []

    for sym in symbols:
        df = load_ohlcv_for_symbol(conn, sym)
        if df.empty:
            print(f"  - {sym}: 캔들 없음, 스킵")
            continue

        res = backtest_symbol(df, sym)
        all_results.extend(res)
        print(f"  - {sym}: 신호 {len(res)}개")

    conn.close()

    if not all_results:
        print("❌ 생성된 신호가 없습니다. 조건이 너무 빡센지 확인해보세요.")
        return

    df_res = pd.DataFrame(all_results)

    # 전체 통계
    total = len(df_res)
    wins = (df_res["ret_pct"] > 0).sum()
    win_rate = wins / total * 100.0
    avg_ret = df_res["ret_pct"].mean()
    med_ret = df_res["ret_pct"].median()

    print("\n==============================")
    print("📊 전체 CR 엔트리 룰 성능")
    print("==============================")
    print(f"총 신호 수          : {total}")
    print(f"승률(>0%)           : {win_rate:.2f}%  ({wins}/{total})")
    print(f"평균 수익률(%)      : {avg_ret:.3f}%")
    print(f"중앙값 수익률(%)    : {med_ret:.3f}%")

    # 전략별 통계
    print("\n------------------------------")
    print("전략 유형별 성능 (strategy_name)")
    print("------------------------------")
    for name, grp in df_res.groupby("strategy_name"):
        n = len(grp)
        w = (grp["ret_pct"] > 0).sum()
        wr = w / n * 100.0
        avg = grp["ret_pct"].mean()
        print(f"[{name:20}] 신호 {n:4d}개 | 승률 {wr:6.2f}% | 평균 {avg:7.3f}%")

    # 원하면 CSV로 저장해서 엑셀로도 볼 수 있음
    out_path = "cr_entry_rule_results.csv"
    df_res.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 개별 신호 결과 CSV 저장: {out_path}")


if __name__ == "__main__":
    main()
