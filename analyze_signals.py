# analyze_signals.py
import sqlite3
import pandas as pd

DB_PATH = "trading.db"

STEP_AHEAD = 3          # 몇 번째 신호 뒤 가격을 볼지 (같은 symbol 내에서)
THRESH_PCT = 0.5        # 미래 수익률이 몇 % 이상이면 좋은 신호로 볼지

def load_signals():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM signals ORDER BY symbol, time", conn)
    conn.close()
    if df.empty:
        print("signals 테이블이 비어있습니다.")
        return df
    df["time"] = pd.to_datetime(df["time"])
    return df

if __name__ == "__main__":
    df = load_signals()
    if df.empty:
        exit()

    # symbol별로 시간 순 정렬
    df = df.sort_values(["symbol", "time"]).reset_index(drop=True)

    # 같은 symbol 안에서 STEP_AHEAD 뒤의 price를 future_price로
    df["future_price"] = (
        df.groupby("symbol")["price"]
          .shift(-STEP_AHEAD)
    )

    # 마지막 쪽 몇 개는 future_price가 없음 → 학습 데이터에서 제외
    df = df.dropna(subset=["future_price"]).copy()

    # 미래 수익률 (%)
    df["future_return_pct"] = (df["future_price"] - df["price"]) / df["price"] * 100

    # 라벨: 미래 수익률이 기준 THRESH_PCT 이상이면 1, 아니면 0
    df["label_good"] = (df["future_return_pct"] >= THRESH_PCT).astype(int)

    # 우리가 쓸 feature들만 추리기
    feature_cols = [
        "at_support",
        "is_bullish",
        "price_up",
        "has_stock",
        "lookback",
        "band_pct",
    ]

    dataset = df[["symbol", "time", "price", "future_return_pct", "label_good"] + feature_cols]

    dataset.to_csv("dataset_signals.csv", index=False, encoding="utf-8-sig")
    print("✅ dataset_signals.csv 저장 완료")
    print("행 개수:", len(dataset))
    print(dataset.head())
