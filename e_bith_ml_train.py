# 시퀀스 기반 공용 ML 분류 모델 학습 스크립트

# - ml_seq_samples 테이블 + ohlcv_data 테이블을 기반으로 시퀀스 피처를 만들고
#   RandomForestClassifier를 학습·평가·버전 관리까지 한 번에 처리하는 공용 트레이너

# 주요 기능:
# 1) load_ml_seq_samples()
#    : trading.db 의 ml_seq_samples 테이블에서 학습 샘플(label 포함) 로드

# 2) load_all_ohlcv()
#    : ohlcv_data 테이블에서 region/symbol/interval별 OHLCV를 모두 로드해
#      (region, symbol, interval)을 key로 하는 dict로 반환

# 3) make_config_hash(cfg)
#    : 학습 설정/메타데이터 dict를 JSON 직렬화 → sha256 해시 → 앞 10자리로 압축한
#      CONFIG_HASH 생성 (모델 버전 추적용)

# 4) train_seq_model_for_universe(universe, region_filter, model_setting_key, ...)
#    : 유니버스/region 기준으로 샘플 필터링 → ml_features.build_feature_from_seq로 피처 생성
#      → RandomForestClassifier 학습/검증
#      → pkl 모델 파일 저장 + 동일 경로의 .meta.json에 학습 설정/해시/정확도 기록
#      → models 테이블에 버전 row 추가
#      → settings 테이블의 model_setting_key 값(활성 모델 경로) 업데이트

import os
import json
import hashlib
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import joblib

from c_db_manager import BotDatabase
from c_ml_features import SEQ_LEN, build_feature_from_seq  # 공통 모듈

DB_PATH = "trading.db"


# -----------------------------------------------------------
# 0) 설정 해시 유틸
# -----------------------------------------------------------
def make_config_hash(cfg: dict) -> str:
    cfg_json = json.dumps(cfg, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()[:10]


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
# 2) 코인(CR/BI) 전용 학습 함수
# -----------------------------------------------------------
def train_seq_model_for_coin_universe(
    universe: list[dict],
    *,
    region_filter: str | None,  # "CR", "BI" 또는 None(코인 전체)
    model_setting_key: str,
    note_prefix: str = "[COIN] ",
    model_dir: str = "models_coin",
    extra_config: dict | None = None,
):
    """
    코인 유니버스(CR / BI)에 대해 모델을 학습하고 저장하는 전용 함수.

    - universe: [{"region": "CR"/"BI", "symbol": "...", ...}, ...]
      * CR  : 빗썸(또는 기존 코인 엔진)
      * BI  : 바이낸스 (원하면 BI 전용으로도 학습 가능)
    - region_filter:
        - "CR": 빗썸 코인 전용 모델
        - "BI": 바이낸스 코인 전용 모델
        - None: CR + BI 전체 코인 한 모델로 학습
    - model_setting_key: settings에 저장할 키 (예: active_model_path_coin, active_model_path_bi)
    - note_prefix: 로그/파일명 접두어 (예: "[COIN_CR] ", "[COIN_BI] ")
    - model_dir: 코인 모델 저장 디렉토리 (기본: models_coin)
    - extra_config: 엔트리/청산 룰 버전 등 메타데이터 dict
    """
    os.makedirs(model_dir, exist_ok=True)
    db = BotDatabase(DB_PATH)
    db.log(f"{note_prefix}🧠 [COIN] 시퀀스 기반 ML 모델 학습 시작 (setting_key={model_setting_key})")

    # 1) 샘플 로드
    df_samples = load_ml_seq_samples()
    if df_samples.empty:
        print("ml_seq_samples 테이블이 비어 있습니다. 먼저 ml_build_seq_samples.py 를 실행하세요.")
        return

    # label 0/1만 사용
    df_samples = df_samples[df_samples["label"].isin([0, 1])].copy()
    if df_samples.empty:
        print("사용 가능한 라벨(0/1)이 없습니다.")
        return

    # 1-1) 코인 전용 region 필터 (CR, BI)
    df_samples = df_samples[df_samples["region"].isin(["CR", "BI"])].copy()
    if df_samples.empty:
        print("[COIN] CR/BI 샘플이 없습니다.")
        return

    if region_filter is not None:
        if region_filter not in ("CR", "BI"):
            print(f"[COIN] region_filter는 CR 또는 BI 또는 None만 허용됩니다. (입력: {region_filter})")
            return

        before = len(df_samples)
        df_samples = df_samples[df_samples["region"] == region_filter].copy()
        after = len(df_samples)
        print(f"[COIN] region={region_filter} 필터: {before} → {after}")
        if df_samples.empty:
            print(f"[COIN] region={region_filter} 에 해당하는 샘플이 없습니다.")
            return

    # 1-2) universe에 포함된 종목만 남기기
    universe_pairs = {(s["region"], s["symbol"]) for s in universe}
    before_cnt = len(df_samples)
    df_samples = df_samples[
        df_samples[["region", "symbol"]]
        .apply(lambda r: (r["region"], r["symbol"]) in universe_pairs, axis=1)
    ].copy()
    after_cnt = len(df_samples)

    print(f"[COIN] UNIVERSE 필터 전 샘플 수: {before_cnt}")
    print(f"[COIN] UNIVERSE 필터 후 샘플 수: {after_cnt}")

    if df_samples.empty:
        print("[COIN] UNIVERSE에 해당하는 샘플이 없습니다.")
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
        print("[COIN] 유효한 피처를 가진 샘플이 없습니다.")
        return

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)

    print(f"[COIN] 학습에 사용되는 샘플 수: {len(X)}")
    print(f"[COIN] 스킵된 샘플 수: {skip_count}")
    print(f"[COIN] 피처 차원: {X.shape[1]}")

    # 4) Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # 5) 모델 학습
    rf_params = {
        "n_estimators": 300,
        "max_depth": None,
        "min_samples_split": 5,
        "min_samples_leaf": 3,
        "random_state": 42,
        "n_jobs": -1,
    }

    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)

    # 6) 평가
    y_pred = model.predict(X_test)
    print(f"{note_prefix}=== [COIN] Classification Report ===")
    print(classification_report(y_test, y_pred, digits=4))

    print(f"{note_prefix}=== [COIN] Confusion Matrix ===")
    print(confusion_matrix(y_test, y_pred))

    accuracy = float((y_pred == y_test).mean())
    print(f"{note_prefix}[COIN] Validation Accuracy: {accuracy:.4f}")

    # 7) 모델 파일명/경로 생성
    now = datetime.now()
    version_str = now.strftime("%Y%m%d_%H%M%S")

    tag = note_prefix.strip("[] ").replace(" ", "_").lower()
    tag = f"{tag}_" if tag else ""
    model_filename = f"seq_model_coin_{tag}{version_str}.pkl"

    model_path = os.path.join(model_dir, model_filename)

    # -------------------------------------------------------
    # 7-1) 학습 설정 메타데이터 구성 + 해시 생성
    # -------------------------------------------------------
    uni_list = sorted({(u["region"], u["symbol"]) for u in universe})

    train_config = {
        "project": "kis-trading-bot",
        "asset_class": "COIN",
        "region_filter": region_filter,
        "model_setting_key": model_setting_key,
        "seq_len": SEQ_LEN,
        "rf_params": rf_params,
        "universe_size": len(uni_list),
        "universe_sample": uni_list[:50],
        "sample_table": "ml_seq_samples",
        "ohlcv_table": "ohlcv_data",
        "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
    }

    if extra_config:
        train_config.update(extra_config)

    config_hash = make_config_hash(train_config)

    # 8) 모델 저장
    joblib.dump(model, model_path)
    print(f"{note_prefix}[COIN] 모델 저장 완료: {model_path}")
    print(f"{note_prefix}[COIN] CONFIG_HASH: {config_hash}")

    # 8-1) 메타데이터 JSON 저장
    meta_path = model_path.replace(".pkl", ".meta.json")
    meta = {
        "model_path": model_path,
        "config": train_config,
        "config_hash": config_hash,
        "n_samples": int(len(X)),
        "val_accuracy": accuracy,
    }

    try:
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
        print(f"{note_prefix}[COIN] 메타데이터 저장 완료: {meta_path}")
    except Exception as e:
        print(f"{note_prefix}[COIN] 메타데이터 저장 실패: {e}")

    note_text = f"{note_prefix}region={region_filter}" if region_filter else note_prefix
    note_text = f"{note_text} cfg={config_hash}"

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
        db.log(f"{note_prefix}[COIN] models 테이블에 버전 기록 완료: {model_path}")
    except Exception as e:
        db.log(f"{note_prefix}[COIN] models 테이블 기록 실패: {e}")

    # 10) settings 에 model_setting_key 갱신
    try:
        db.set_setting(model_setting_key, model_path)
        db.log(f"{note_prefix}[COIN] {model_setting_key} 갱신: {model_path}")
    except Exception as e:
        db.log(f"{note_prefix}[COIN] {model_setting_key} 갱신 실패: {e}")

    db.log(
        f"{note_prefix}✅ [COIN] 시퀀스 기반 ML 모델 학습/저장 완료 "
        f"(accuracy={accuracy:.4f}, cfg={config_hash})"
    )

    return model_path, accuracy, len(X)