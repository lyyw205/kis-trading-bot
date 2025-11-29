# build_dataset_cr_swing.py
"""
CR 고수익 단타용 멀티호라이즌 모델 데이터셋 빌더 (빠른 버전)

- 입력: trading.db 의 ohlcv_data (region='CR', interval='5m')
- 출력: datasets/cr_swing/ 에 train/valid/test npz 파일
    - X_*: (N, SEQ_LEN_SWING, feature_dim)
    - Y_*: (N, 3)  # [r_3, r_6, r_12]
"""

import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

from numpy.lib.stride_tricks import sliding_window_view  # 🚀 벡터화용

from utils import calculate_atr

DB_PATH = "trading.db"

# ---- 하이퍼파라미터 / 설정 ----
SEQ_LEN_SWING = 60          # 입력 시퀀스 길이 (5분봉 60개 ≒ 5시간)
HORIZONS = [3, 6, 12]       # 3, 6, 12 캔들 뒤 수익률
MAX_H = max(HORIZONS)

# (선택) 데이터 기간 제한 – 너무 과거까지 안 쓰고 싶으면 START_DATE 조절
START_DATE = "2024-01-01 00:00:00"   # 이 날짜 이후만 사용
TRAIN_END = "2024-09-30 23:59:59"
VALID_END = "2024-11-15 23:59:59"

OUTPUT_DIR = "datasets/cr_swing"

# (선택) 심볼 화이트리스트 – 400개 전체 말고, universe만 쓸 거면 여기 채우기
# 예: CR_UNIVERSE = ["KRW-BTC", "KRW-ETH", "KRW-XRP"]
CR_UNIVERSE = None  # None 이면 전체 심볼 사용


def load_cr_ohlcv():
    """
    ohlcv_data 에서 CR, 5m 만 불러오기.
    컬럼 가정:
      region, symbol, interval, dt, open, high, low, close, volume
    """
    conn = sqlite3.connect(DB_PATH)
    query = """
        SELECT region, symbol, interval, dt, open, high, low, close, volume
        FROM ohlcv_data
        WHERE region = 'CR'
          AND interval = '5m'
          AND dt >= ?
        ORDER BY symbol, dt
    """
    df = pd.read_sql_query(query, conn, params=[START_DATE])
    conn.close()

    if df.empty:
        raise ValueError("CR 5m ohlcv_data 가 비어있습니다. (기간/region 확인 필요)")

    df["dt"] = pd.to_datetime(df["dt"])
    return df


def add_features_for_symbol(df_sym: pd.DataFrame) -> pd.DataFrame:
    """
    개별 symbol 에 대해 feature 컬럼을 직접 추가.
    - MA20 / MA60 / VOL_MA20
    - ATR (14)
    - hl_range, ret_1
    """
    df_sym = df_sym.sort_values("dt").reset_index(drop=True)

    df_feat = df_sym.copy()

    # 이동평균 / 거래량 평균
    df_feat["ma20"] = df_feat["close"].rolling(20).mean()
    df_feat["ma60"] = df_feat["close"].rolling(60).mean()
    df_feat["vol_ma20"] = df_feat["volume"].rolling(20).mean()

    # ATR (기존 함수 재사용)
    try:
        df_feat["atr"] = calculate_atr(df_feat, period=14)
    except Exception:
        df_feat["atr"] = np.nan

    # 파생 특징
    df_feat["hl_range"] = (df_feat["high"] - df_feat["low"]) / df_feat["close"].replace(0, np.nan)
    df_feat["ret_1"] = df_feat["close"].pct_change()

    # 🔥 여기서 중요한 부분:
    # rolling 으로 생긴 NaN 앞 구간은 통째로 버린다.
    df_feat = df_feat.dropna(
        subset=["ma20", "ma60", "vol_ma20", "atr", "hl_range", "ret_1"]
    ).reset_index(drop=True)

    # feature 컬럼 정의 (rsi는 일단 빼자)
    feature_cols = [
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ma20",
        "ma60",
        "vol_ma20",
        "atr",
        "hl_range",
        "ret_1",
    ]

    for col in feature_cols:
        if col not in df_feat.columns:
            df_feat[col] = np.nan

    df_feat = df_feat[["dt", "symbol"] + feature_cols].copy()
    return df_feat


def make_sequences_for_symbol_fast(df_feat: pd.DataFrame):
    """
    하나의 심볼에 대해, 시퀀스 + 멀티호라이즌 타겟 생성 (벡터화 버전).

    반환:
      X: (N, seq_len, feature_dim)
      Y: (N, len(HORIZONS))
      meta_list: [{symbol, entry_dt}, ...]
    """
    df_feat = df_feat.sort_values("dt").reset_index(drop=True)
    if len(df_feat) < SEQ_LEN_SWING + MAX_H + 1:
        return None, None, []

    feature_cols = [c for c in df_feat.columns if c not in ("dt", "symbol")]
    values = df_feat[feature_cols].values  # (n, feature_dim)
    closes = df_feat["close"].values      # (n,)
    dts = df_feat["dt"].values
    symbol = df_feat["symbol"].iloc[0]

    n = len(df_feat)

    # 🚀 1) 슬라이딩 윈도우로 시퀀스 한 번에 생성
    # sliding_window_view(values, (SEQ_LEN_SWING, feature_dim)) 결과 shape:
    #   (n - SEQ_LEN_SWING + 1, 1, SEQ_LEN_SWING, feature_dim)
    windows = sliding_window_view(
        values,
        window_shape=(SEQ_LEN_SWING, values.shape[1])
    )[:, 0, :, :]  # → (n - SEQ_LEN_SWING + 1, SEQ_LEN_SWING, feature_dim)

    # 시퀀스의 "끝 인덱스"는 SEQ_LEN_SWING-1 ~ n-1
    # 그 중에서 미래 MAX_H 캔들이 존재하는 인덱스까지만 유효
    max_end_idx = n - 1 - MAX_H
    start_end = SEQ_LEN_SWING - 1
    valid_len = max_end_idx - start_end + 1
    if valid_len <= 0:
        return None, None, []

    windows = windows[:valid_len]  # (valid_len, SEQ_LEN_SWING, feature_dim)

    # 🚀 2) 엔트리 시점 종가 / 날짜 벡터화
    end_indices = np.arange(start_end, start_end + valid_len)  # (valid_len,)
    entry_prices = closes[end_indices]                         # (valid_len,)
    entry_dts = dts[end_indices]                               # (valid_len,)

    # 엔트리 가격이 0 or NaN 인 것은 제거
    valid_mask = (entry_prices > 0) & ~np.isnan(entry_prices)

    # 🚀 3) 멀티호라이즌 수익률 벡터화
    future_rets_list = []
    for h in HORIZONS:
        future_indices = end_indices + h
        future_prices = closes[future_indices]
        mask_h = (future_prices > 0) & ~np.isnan(future_prices)
        r_h = future_prices / entry_prices - 1.0
        future_rets_list.append(r_h)
        valid_mask &= mask_h

    if not np.any(valid_mask):
        return None, None, []

    # 🚀 4) 1차 마스크 적용
    X = windows[valid_mask]                          # (N, seq_len, feature_dim)
    Y = np.stack([r[valid_mask] for r in future_rets_list], axis=1)  # (N, len(HORIZONS))
    entry_dts_valid = entry_dts[valid_mask]          # (N,)

    # 🚀 5) NaN 포함 샘플 제거
    nan_mask = np.isnan(X).any(axis=(1, 2)) | np.isnan(Y).any(axis=1)
    if np.any(nan_mask):
        keep = ~nan_mask
        before = X.shape[0]
        X = X[keep]
        Y = Y[keep]
        entry_dts_valid = entry_dts_valid[keep]
        print(f"    - NaN 포함 샘플 {before - X.shape[0]}개 제거, 남은 샘플 {X.shape[0]}개")

    if X.shape[0] == 0:
        return None, None, []

    # ✅ 6) meta_list 생성 (여기가 빠져 있었음)
    meta_list = [
        {"symbol": symbol, "entry_dt": dt_i}
        for dt_i in entry_dts_valid
    ]

    return X, Y, meta_list


def split_by_time(meta_list, X_array, Y_array):
    """
    meta_list 의 entry_dt 기준으로 train/valid/test 인덱스를 나눈다.
    """
    # 🔥 datetime → pandas.Timestamp 로 통일
    train_end_dt = pd.to_datetime(TRAIN_END)
    valid_end_dt = pd.to_datetime(VALID_END)

    train_idx, valid_idx, test_idx = [], [], []

    for i, meta in enumerate(meta_list):
        # meta["entry_dt"] 는 numpy.datetime64 일 수도 있고, 문자열일 수도 있음 → 전부 Timestamp로 변환
        dt_i = pd.to_datetime(meta["entry_dt"])

        if dt_i <= train_end_dt:
            train_idx.append(i)
        elif dt_i <= valid_end_dt:
            valid_idx.append(i)
        else:
            test_idx.append(i)

    def take(indices):
        if not indices:
            return (
                np.empty((0,) + X_array.shape[1:], dtype=X_array.dtype),
                np.empty((0,) + Y_array.shape[1:], dtype=Y_array.dtype),
            )
        X_sub = X_array[indices]
        Y_sub = Y_array[indices]
        return X_sub, Y_sub

    X_train, Y_train = take(train_idx)
    X_valid, Y_valid = take(valid_idx)
    X_test, Y_test = take(test_idx)

    return (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("📥 CR 5m OHLCV 데이터 로딩 중...")
    df = load_cr_ohlcv()
    print(f"- 전체 행 수: {len(df)}")

    if CR_UNIVERSE is not None:
        before = len(df)
        df = df[df["symbol"].isin(CR_UNIVERSE)].copy()
        after = len(df)
        print(f"- 심볼 화이트리스트 적용: rows {before} → {after}")

    all_X = []
    all_Y = []
    all_meta = []

    symbols = sorted(df["symbol"].unique())
    print(f"- 심볼 수: {len(symbols)}")

    for sym in symbols:
        df_sym = df[df["symbol"] == sym].copy()
        if df_sym.empty:
            continue

        print(f"  ▶ 심볼 처리: {sym} (rows={len(df_sym)})")

        # 1) feature 붙이기
        try:
            df_feat = add_features_for_symbol(df_sym)
        except Exception as e:
            print(f"    ❌ feature 계산 실패, 스킵: {e}")
            continue

        # 2) 시퀀스 + 타겟 생성 (벡터화 버전)
        X_sym, Y_sym, meta_sym = make_sequences_for_symbol_fast(df_feat)

        if X_sym is None or X_sym.shape[0] == 0:
            print(f"    - 유효한 샘플이 없어 스킵")
            continue

        all_X.append(X_sym)
        all_Y.append(Y_sym)
        all_meta.extend(meta_sym)

        print(f"    - 생성 샘플 수: {X_sym.shape[0]}")

    if not all_X:
        raise ValueError("생성된 샘플이 없습니다. 데이터/설정 확인 필요.")

    X = np.concatenate(all_X, axis=0)
    Y = np.concatenate(all_Y, axis=0)

    print(f"\n✅ 전체 샘플 수: {X.shape[0]}")
    print(f"   X shape: {X.shape}, Y shape: {Y.shape}")

    # 시간 기준으로 train/valid/test split
    (X_train, Y_train), (X_valid, Y_valid), (X_test, Y_test) = split_by_time(all_meta, X, Y)

    print("\n📊 Split 결과:")
    print(f"  - Train: {X_train.shape[0]} 샘플")
    print(f"  - Valid: {X_valid.shape[0]} 샘플")
    print(f"  - Test : {X_test.shape[0]} 샘플")

    # npz 저장
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "cr_swing_train.npz"),
        X=X_train,
        Y=Y_train,
    )
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "cr_swing_valid.npz"),
        X=X_valid,
        Y=Y_valid,
    )
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "cr_swing_test.npz"),
        X=X_test,
        Y=Y_test,
    )

    print(f"\n💾 저장 완료: {OUTPUT_DIR}/cr_swing_*.npz")


if __name__ == "__main__":
    main()
