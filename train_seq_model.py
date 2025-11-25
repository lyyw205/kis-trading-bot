# train_seq_model.py
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

from db import BotDatabase

DB_PATH = "trading.db"
MODEL_DIR = "models"

# 시퀀스 길이 (build_ml_seq_samples.py / trader.py 와 동일해야 함)
SEQ_LEN = 40


def load_ml_seq_samples():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM ml_seq_samples", conn)
    conn.close()
    return df


def load_all_ohlcv():
    """
    ohlcv_data를 region+symbol+interval 별로 DataFrame으로 묶어서 dict로 반환
    key: (region, symbol, interval)
    val: df (index=dt, columns=[open, high, low, close, volume])
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT region, symbol, interval, dt, open, high, low, close, volume
        FROM ohlcv_data
        ORDER BY region, symbol, interval, dt
        """,
        conn,
    )
    conn.close()

    if df.empty:
        return {}

    df["dt"] = pd.to_datetime(df["dt"])

    groups = {}
    for (region, symbol, interval), g in df.groupby(
        ["region", "symbol", "interval"], sort=False
    ):
        g = g.copy()
        g = g.sort_values("dt")
        g.set_index("dt", inplace=True)
        g = g[["open", "high", "low", "close", "volume"]].astype(float)
        groups[(region, symbol, interval)] = g

    return groups


def build_feature_from_seq(df_seq):
    """
    df_seq: index=dt, columns=['open','high','low','close','volume']
    길이 = SEQ_LEN

    간단한 패턴 인식용 feature:
      - 가격들을 첫 캔들의 종가 대비 수익률로 정규화
      - 거래량을 평균 대비 비율로 정규화
      -> close_rel(SEQ_LEN) + high_rel(SEQ_LEN) + low_rel(SEQ_LEN) + vol_norm(SEQ_LEN)
         = 4 * SEQ_LEN 차원
    """
    if len(df_seq) != SEQ_LEN:
        return None

    close = df_seq["close"].values
    high = df_seq["high"].values
    low = df_seq["low"].values
    vol = df_seq["volume"].values

    base = close[0]
    if base <= 0:
        return None

    close_rel = close / base - 1.0
    high_rel = high / base - 1.0
    low_rel = low / base - 1.0

    vol_mean = np.mean(vol) if np.mean(vol) > 0 else 1.0
    vol_norm = vol / vol_mean

    feat = np.concatenate([close_rel, high_rel, low_rel, vol_norm])
    return feat


if __name__ == "__main__":
    os.makedirs(MODEL_DIR, exist_ok=True)
    db = BotDatabase(DB_PATH)
    db.log("🧠 시퀀스 기반 ML 모델 학습 시작")

    # 1) 샘플 로드
    df_samples = load_ml_seq_samples()
    if df_samples.empty:
        print("❌ ml_seq_samples 테이블이 비어 있습니다. 먼저 build_ml_seq_samples.py 를 실행하세요.")
        raise SystemExit

    # label 이 0/1인 것만 사용
    df_samples = df_samples[df_samples["label"].isin([0, 1])].copy()
    if df_samples.empty:
        print("❌ 사용 가능한 라벨(0/1)이 없습니다.")
        raise SystemExit

    # 2) OHLCV 전체 로드
    ohlcv_dict = load_all_ohlcv()
    if not ohlcv_dict:
        print("❌ ohlcv_data 테이블이 비어 있거나 데이터가 없습니다.")
        raise SystemExit

    X_list = []
    y_list = []

    skip_count = 0

    # 3) 각 샘플에 대해 시퀀스 피처 구성
    for _, row in df_samples.iterrows():
        region = row["region"]
        symbol = row["symbol"]
        interval = row["interval"]
        dt_entry_str = row["dt_entry"]
        label = int(row["label"])

        key = (region, symbol, interval)
        if key not in ohlcv_dict:
            skip_count += 1
            continue

        df_ohlcv = ohlcv_dict[key]

        dt_entry = pd.to_datetime(dt_entry_str)
        if dt_entry not in df_ohlcv.index:
            # 타임존 또는 분 단위 차이가 있는 경우 여기서 nearest 매칭 로직을 넣을 수도 있음
            skip_count += 1
            continue

        pos = df_ohlcv.index.get_loc(dt_entry)
        if isinstance(pos, slice):
            pos = pos.stop - 1

        if pos < SEQ_LEN - 1:
            skip_count += 1
            continue

        df_seq = df_ohlcv.iloc[pos - SEQ_LEN + 1 : pos + 1]

        feat = build_feature_from_seq(df_seq)
        if feat is None:
            skip_count += 1
            continue

        X_list.append(feat)
        y_list.append(label)

    if not X_list:
        print("❌ 유효한 피처를 가진 샘플이 없습니다.")
        raise SystemExit

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)

    print(f"✅ 학습에 사용되는 샘플 수: {len(X)}")
    print(f"   스킵된 샘플 수: {skip_count}")
    print(f"   피처 차원: {X.shape[1]}")

    # 4) Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5) 모델 학습
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    # 6) 평가
    y_pred = model.predict(X_test)
    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    print("=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    accuracy = float((y_pred == y_test).mean())
    print(f"✅ Validation Accuracy: {accuracy:.4f}")

    # 7) 모델 버전 이름/경로 생성
    now = datetime.now()
    version_str = now.strftime("%Y%m%d_%H%M%S")
    model_filename = f"seq_model_{version_str}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)

    # 8) 모델 저장
    joblib.dump(model, model_path)
    print(f"💾 모델 저장 완료: {model_path}")

    # 9) models 테이블에 버전 기록
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO models (created_at, path, n_samples, val_accuracy)
            VALUES (?, ?, ?, ?)
            """,
            (now.strftime("%Y-%m-%d %H:%M:%S"), model_path, int(len(X)), accuracy),
        )
        conn.commit()
        conn.close()
        db.log(f"✅ models 테이블에 버전 기록 완료: {model_path}")
    except Exception as e:
        db.log(f"⚠️ models 테이블 기록 실패: {e}")

    # 10) settings 에 active_model_path 갱신
    try:
        db.set_setting("active_model_path", model_path)
        db.log(f"🎯 active_model_path 갱신: {model_path}")
    except Exception as e:
        db.log(f"⚠️ active_model_path 갱신 실패: {e}")

    db.log(f"🎉 시퀀스 기반 ML 모델 학습/저장 완료 (accuracy={accuracy:.4f})")
