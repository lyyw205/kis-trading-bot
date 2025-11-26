# prepare_dataset.py
# 완전 심플한 데이터 로딩 유틸
# signals 테이블 전체를 pandas DataFrame으로 읽어오는 함수
# ML 학습용 feature 만들어내기 등의 작업을 Jupyter / Colab에서 하기에 딱 좋은 형태

# !! ML 모델을 만들기 전에 반드시 필요한 preprocessing 단계를 담당하는 파일 !!

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
