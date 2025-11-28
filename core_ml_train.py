# core_ml_train_seq_model.py
"""
시퀀스 기반 ML 모델 '공통 학습 코어'

- 입력:
  - ml_seq_samples 테이블
  - ohlcv_data 테이블
  - universe (KR / US / CR 유니버스 리스트)
  - region_filter ("KR" / "US" / "CR" or None)
  - model_setting_key (settings에 저장할 키 이름)

- 출력:
  - 모델 파일 저장
  - models 테이블 버전 기록
  - settings.<model_setting_key> 갱신
"""

import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

from db_manager import BotDatabase
from ml_features import SEQ_LEN, build_feature_from_seq  # 공통 모듈

DB_PATH = "trading.db"


# -----------------------------------------------------------
# 1) 학습용 샘플 / OHLCV 로딩
# -----------------------------------------------------------
def load_ml_seq_samples() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM ml_seq_samples", conn)
    conn.close()
    return df


def load_all_ohlcv() -> dict:
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
        g = g.copy().sort_values("dt")
        g.set_index("dt", inplace=True)

        g = g[["open", "high", "low", "close", "volume"]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()

        if g.empty:
            continue

        groups[(region, symbol, interval)] = g

    return groups


# -----------------------------------------------------------
# 2) 공통 학습 함수
# -----------------------------------------------------------
def train_seq_model_for_universe(
    universe: list[dict],
    *,
    region_filter: str | None,
    model_setting_key: str,
    note_prefix: str = "",
    model_dir: str = "models",
):
    """
    하나의 유니버스(KR / US / CR)에 대해 모델을 학습하고 저장하는 공통 함수.

    - universe: [{"region": "...", "symbol": "...", "excd": "..."}, ...]
    - region_filter: "KR", "US", "CR" 등 (None이면 필터링 안 함)
    - model_setting_key: settings에 저장할 키 (예: active_model_path_kr)
    - note_prefix: 로그/파일명에 붙일 접두어 (예: "[KR_STOCK] ")
    - model_dir: 모델 저장 디렉토리
    """
    os.makedirs(model_dir, exist_ok=True)
    db = BotDatabase(DB_PATH)
    db.log(f"{note_prefix}🧠 시퀀스 기반 ML 모델 학습 시작 (setting_key={model_setting_key})")

    # 1) 샘플 로드
    df_samples = load_ml_seq_samples()
    if df_samples.empty:
        print("ml_seq_samples 테이블이 비어 있습니다. 먼저 ml_build_seq_samples.py 를 실행하세요.")
        return

    # label 0/1인 것만
    df_samples = df_samples[df_samples["label"].isin([0, 1])].copy()
    if df_samples.empty:
        print("사용 가능한 라벨(0/1)이 없습니다.")
        return

    # 1-1) region 필터 (KR / US / CR 등)
    if region_filter is not None:
        before = len(df_samples)
        df_samples = df_samples[df_samples["region"] == region_filter].copy()
        after = len(df_samples)
        print(f"[{note_prefix}] region={region_filter} 필터: {before} → {after}")
        if df_samples.empty:
            print(f"region={region_filter} 에 해당하는 샘플이 없습니다.")
            return

    # 1-2) universe에 포함된 종목만 남기기
    universe_pairs = {(s["region"], s["symbol"]) for s in universe}
    before_cnt = len(df_samples)
    df_samples = df_samples[
        df_samples[["region", "symbol"]]
        .apply(lambda r: (r["region"], r["symbol"]) in universe_pairs, axis=1)
    ].copy()
    after_cnt = len(df_samples)

    print(f"[{note_prefix}] UNIVERSE 필터 전 샘플 수: {before_cnt}")
    print(f"[{note_prefix}] UNIVERSE 필터 후 샘플 수: {after_cnt}")

    if df_samples.empty:
        print(f"{note_prefix} UNIVERSE에 해당하는 샘플이 없습니다.")
        return

    # 2) OHLCV 전체 로드
    ohlcv_dict = load_all_ohlcv()
    if not ohlcv_dict:
        print("ohlcv_data 테이블이 비어 있거나 데이터가 없습니다.")
        return

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
        print(f"{note_prefix} 유효한 피처를 가진 샘플이 없습니다.")
        return

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)

    print(f"{note_prefix} 학습에 사용되는 샘플 수: {len(X)}")
    print(f"{note_prefix} 스킵된 샘플 수: {skip_count}")
    print(f"{note_prefix} 피처 차원: {X.shape[1]}")

    # 4) Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 5) 모델 학습 (필요하면 자산군별로 파라미터 다르게 넣어도 됨)
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
    print(f"{note_prefix}=== Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    print(f"{note_prefix}=== Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    accuracy = float((y_pred == y_test).mean())
    print(f"{note_prefix} Validation Accuracy: {accuracy:.4f}")

    # 7) 모델 파일명/경로 생성
    now = datetime.now()
    version_str = now.strftime("%Y%m%d_%H%M%S")

    # note_prefix를 파일명에 살짝 녹여주기
    tag = note_prefix.strip("[] ").replace(" ", "_").lower()
    tag = f"{tag}_" if tag else ""
    model_filename = f"seq_model_{tag}{version_str}.pkl"

    model_path = os.path.join(model_dir, model_filename)

    # 8) 모델 저장
    joblib.dump(model, model_path)
    print(f"{note_prefix} 모델 저장 완료: {model_path}")

    note_text = f"{note_prefix}region={region_filter}" if region_filter else note_prefix
    # 9) models 테이블에 버전 기록
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO models (created_at, path, n_samples, val_accuracy, note)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                now.strftime("%Y-%m-%d %H:%M:%S"),
                model_path,
                int(len(X)),
                accuracy,
                note_text,
            ),
        )
        conn.commit()
        conn.close()
        db.log(f"{note_prefix} models 테이블에 버전 기록 완료: {model_path}")
    except Exception as e:
        db.log(f"{note_prefix} models 테이블 기록 실패: {e}")

    # 10) settings 에 model_setting_key 갱신
    try:
        db.set_setting(model_setting_key, model_path)
        db.log(f"{note_prefix} {model_setting_key} 갱신: {model_path}")
    except Exception as e:
        db.log(f"{note_prefix} {model_setting_key} 갱신 실패: {e}")

    db.log(f"{note_prefix}✅ 시퀀스 기반 ML 모델 학습/저장 완료 (accuracy={accuracy:.4f})")

    return model_path, accuracy, len(X)
