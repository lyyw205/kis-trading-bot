# monitor_ml.py
import sqlite3
import pandas as pd
import matplotlib.pyplot as plt

DB_PATH = "trading.db"

def load_signals(limit=500):
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql(f"""
        SELECT time, symbol, ml_proba, entry_signal, entry_allowed
        FROM signals
        WHERE ml_proba IS NOT NULL
        ORDER BY id DESC
        LIMIT {limit}
    """, conn)
    conn.close()
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time")

def main():
    df = load_signals(500)

    if df.empty:
        print("signals 데이터가 없습니다.")
        return

    # 1) ML 확률 히스토그램
    plt.figure()
    df["ml_proba"].hist(bins=20)
    plt.title("ML 확률 분포 (최근 500 신호)")
    plt.xlabel("ml_proba")
    plt.ylabel("count")
    plt.tight_layout()

    # 2) 시간에 따른 ML 확률 + entry_allowed 표시
    plt.figure()
    plt.plot(df["time"], df["ml_proba"], marker="o", linestyle="-", alpha=0.5)
    # entry_allowed 된 것만 빨간 점으로 표시
    allowed = df[df["entry_allowed"] == 1]
    plt.scatter(allowed["time"], allowed["ml_proba"], marker="o", edgecolor="red", facecolor="none", s=80, label="entry_allowed")

    plt.title("시간에 따른 ML 확률 (최근 500 신호)")
    plt.xlabel("time")
    plt.ylabel("ml_proba")
    plt.legend()
    plt.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()
