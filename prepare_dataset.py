# prepare_dataset.py
import sqlite3
import pandas as pd

DB_PATH = "trading.db"

def load_signals():
    conn = sqlite3.connect(DB_PATH)
    df_sig = pd.read_sql_query("SELECT * FROM signals", conn)
    conn.close()
    df_sig["time"] = pd.to_datetime(df_sig["time"])
    return df_sig

def load_trades():
    conn = sqlite3.connect(DB_PATH)
    df_tr = pd.read_sql_query("SELECT * FROM trades", conn)
    conn.close()
    df_tr["time"] = pd.to_datetime(df_tr["time"])
    return df_tr

if __name__ == "__main__":
    sig = load_signals()
    tr = load_trades()

    print("signals:", sig.shape)
    print("trades:", tr.shape)

    # 예시: 신호 이후 3개 트레이드 안에 해당 심볼 매도 발생했는지 라벨 만들기…
    # 여기는 나중에 전략/라벨링 방식 정해지면 구체적으로 채우면 됨.
