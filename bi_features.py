# bi_features.py
# BI 코인 멀티스케일 모델용 공통 Feature 정의 + 샘플 생성 유틸

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# ==========================================================
# 1) 공통 Feature / 시퀀스 / Horizon 정의
# ==========================================================

# 모델이 보는 기본 OHLCV 컬럼 (학습/실전 공통)
RAW_FEATURE_COLS: List[str] = ["open", "high", "low", "close", "volume"]

# 파생 피처까지 포함한 최종 Feature 컬럼 (학습/실전 공통)
FEATURE_COLS: List[str] = RAW_FEATURE_COLS + [
    "log_ret_1",    # 1-step 로그 수익률
    "hl_range",     # 고저폭 / 이전 종가
    "body",         # 몸통 길이 / 이전 종가
    "upper_shadow", # 윗꼬리 / 이전 종가
    "lower_shadow", # 아랫꼬리 / 이전 종가
    "vol_chg",      # 거래량 변화율
    "rsi_14",       # 14 period RSI
    "vol_20",       # 20 period 수익률 변동성
]

# 멀티스케일 시퀀스 길이 (학습/실전 공통)
SEQ_LENS: Dict[str, int] = {
    "5m": 48,
    "15m": 24,
    "30m": 16,
    "1h": 10,
}

# 5m 기준 N개 뒤 수익률 예측 horizon (학습/실전 공통)
HORIZONS: List[int] = [3, 6, 12, 24]


# ==========================================================
# 2) 윈도우 정규화 유틸 (vectorized_window_scaling)
#    → 기존 bi_create_dataset 에 있던 구현을 여기로 옮겨 붙이면 됨
# ==========================================================

def vectorized_window_scaling(windows: np.ndarray, chunk_size: int = 20000) -> np.ndarray:
    """
    3D 배열 (N, Window, Feature)에 대해 정규화 수행
    axis=1 (시간축) 기준으로 평균/표준편차 구함

    메모리 절약을 위해 N을 한 번에 처리하지 않고
    chunk_size 단위로 나눠서 처리한다.
    """
    # 여기서는 전체를 한 번에 astype 하지 않음 (메모리 폭발 방지)
    N, L, F = windows.shape

    # 출력은 float32로 통일
    out = np.empty((N, L, F), dtype=np.float32)

    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)

        # 이 청크만 float32로 캐스팅 (copy=False라 이미 float32면 추가 메모리 없음)
        chunk = windows[start:end].astype(np.float32, copy=False)  # (C, L, F)

        mean = np.mean(chunk, axis=1, keepdims=True, dtype=np.float32)
        std = np.std(chunk, axis=1, keepdims=True, dtype=np.float32) + 1e-6

        out[start:end] = (chunk - mean) / std

    return out

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    OHLCV DataFrame에 파생 피처를 추가해서 반환.
    인덱스는 유지되고, NaN/inf는 0으로 정리.
    """
    df = df.copy()

    # 기본 컬럼 float로 캐스팅
    for col in RAW_FEATURE_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    c = df["close"]
    o = df["open"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # 1) 로그 수익률
    df["log_ret_1"] = np.log(c / c.shift(1))
    df["log_ret_1"] = df["log_ret_1"].replace([np.inf, -np.inf], 0.0)

    # 기준값: 이전 종가 (0으로 나누기 방지용)
    base = c.shift(1).replace(0, np.nan)

    # 2) 캔들 구조 관련
    df["hl_range"] = (h - l) / base
    df["body"] = (c - o) / base
    df["upper_shadow"] = (h - np.maximum(o, c)) / base
    df["lower_shadow"] = (np.minimum(o, c) - l) / base

    # 3) 거래량 변화율 (이상치 클리핑)
    df["vol_chg"] = v.pct_change()
    df["vol_chg"] = df["vol_chg"].clip(-5, 5)   # 너무 큰 값은 ±500%에서 자름

    # 4) RSI(14)
    delta = c.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    window = 14
    roll_up = up.rolling(window).mean()
    roll_down = down.rolling(window).mean()

    rs = roll_up / (roll_down + 1e-6)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # 5) 20 period 변동성 (로그수익률 표준편차)
    df["vol_20"] = df["log_ret_1"].rolling(20).std()

    # NaN / inf 정리
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(0.0, inplace=True)

    return df

# ==========================================================
# 3) 멀티스케일 샘플 생성 (학습/실전 공통)
# ==========================================================

def build_multiscale_samples_cr(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_30m: pd.DataFrame,
    df_1h: pd.DataFrame,
    feature_cols: Optional[List[str]] = None,
    seq_lens: Optional[Dict[str, int]] = None,
    horizons: Optional[List[int]] = None,
    return_index: bool = False,
) -> Tuple[
    np.ndarray,  # X_5m: (N, L5, F)
    np.ndarray,  # X_15m: (N, L15, F)
    np.ndarray,  # X_30m: (N, L30, F)
    np.ndarray,  # X_1h: (N, L1h, F)
    np.ndarray,  # Y: (N, H)
    Optional[np.ndarray],  # base_dt: (N,) or None
]:

    """
    벡터화된 멀티스케일 샘플 생성기 (속도 최적화)

    기존 bi_create_dataset.build_multiscale_samples_cr 구현을
    학습/실전 공통으로 쓰기 위해 이쪽으로 옮긴 버전.
    """

    if feature_cols is None:
        feature_cols = FEATURE_COLS
    if seq_lens is None:
        seq_lens = SEQ_LENS
    if horizons is None:
        horizons = HORIZONS

    # 1. 데이터 정렬 및 인덱스 통일
    df_5m = df_5m.sort_index()

    # 다른 타임프레임 데이터를 5분봉 인덱스에 맞춰 확장 (Forward Fill)
    common_idx = df_5m.index

    # reindex(method='ffill')이 핵심: 직전 값 참조, 미래 참조 방지
    d15 = df_15m.reindex(common_idx, method="ffill").fillna(0)[feature_cols].values
    d30 = df_30m.reindex(common_idx, method="ffill").fillna(0)[feature_cols].values
    d1h = df_1h.reindex(common_idx, method="ffill").fillna(0)[feature_cols].values
    d5 = df_5m[feature_cols].values  # (Total_Len, Feat)

    closes = df_5m["close"].values

    L5 = seq_lens["5m"]
    L15 = seq_lens["15m"]
    L30 = seq_lens["30m"]
    L1h = seq_lens["1h"]

    max_lookback = max(L5, L15, L30, L1h)
    max_h = max(horizons)
    total_len = len(d5)

    if total_len < max_lookback + max_h + 10:
        raise ValueError("데이터가 너무 짧습니다.")

    # 2. Sliding Window View 생성 (메모리 복사 없이 뷰만 생성)
    # 결과 shape: (N_windows, Window_Size, Features)
    sw5 = sliding_window_view(d5, window_shape=(L5, len(feature_cols))).squeeze()
    sw15 = sliding_window_view(d15, window_shape=(L15, len(feature_cols))).squeeze()
    sw30 = sliding_window_view(d30, window_shape=(L30, len(feature_cols))).squeeze()
    sw1h = sliding_window_view(d1h, window_shape=(L1h, len(feature_cols))).squeeze()

    # 3. 유효한 인덱스 범위 계산
    start_idx = max_lookback
    end_idx = total_len - max_h

    if start_idx >= end_idx:
        raise ValueError(f"유효 구간 없음 (len={total_len}, start={start_idx}, end={end_idx})")

    valid_range_len = end_idx - start_idx  # (사용은 안 하지만 개념적으로)

    # 각 스케일별로 필요한 구간만 잘라냄
    X_5m = sw5[start_idx - L5 : end_idx - L5]
    X_15m = sw15[start_idx - L15 : end_idx - L15]
    X_30m = sw30[start_idx - L30 : end_idx - L30]
    X_1h = sw1h[start_idx - L1h : end_idx - L1h]

    # 길이 맞추기
    min_len = min(len(X_5m), len(X_15m), len(X_30m), len(X_1h))
    X_5m = X_5m[:min_len]
    X_15m = X_15m[:min_len]
    X_30m = X_30m[:min_len]
    X_1h = X_1h[:min_len]

    # 4. 일괄 정규화
    X_5m = vectorized_window_scaling(X_5m)
    X_15m = vectorized_window_scaling(X_15m)
    X_30m = vectorized_window_scaling(X_30m)
    X_1h = vectorized_window_scaling(X_1h)

    # 5. Y(타겟) 생성
    cur_prices = closes[start_idx : start_idx + min_len]

    Y_list = []
    for h in horizons:
        target_prices = closes[start_idx + h : start_idx + min_len + h]
        ret = (target_prices[: len(cur_prices)] / cur_prices) - 1.0
        Y_list.append(ret)

    Y = np.stack(Y_list, axis=1)  # (N, H)

    # 각 샘플의 기준 시점 (5분봉 dt)
    base_dt = df_5m.index[start_idx : start_idx + min_len].to_numpy()

    if return_index:
        return X_5m, X_15m, X_30m, X_1h, Y, base_dt
    else:
        return X_5m, X_15m, X_30m, X_1h, Y


# ==========================================================
# 4) 리샘플링 (학습/실전 공통)
# ==========================================================
RESAMPLE_AGG = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
}

def resample_from_5m(df_5m: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    5분봉 원본 OHLCV(df_5m)를 받아서
    - 5m / 15m / 30m / 1h OHLCV로 리샘플
    - 각 타임프레임마다 add_derived_features()로 파생 피처까지 추가
    를 한 번에 수행해서 반환.
    """
    df_5m = df_5m.sort_index()
    df_5m = df_5m[RAW_FEATURE_COLS].apply(pd.to_numeric, errors="coerce").dropna()

    df_15m = df_5m.resample("15min").agg(RESAMPLE_AGG).dropna()
    df_30m = df_5m.resample("30min").agg(RESAMPLE_AGG).dropna()
    df_1h  = df_5m.resample("60min").agg(RESAMPLE_AGG).dropna()

    # 각 타임프레임별 파생 피처 추가
    df_5m = add_derived_features(df_5m)
    df_15m = add_derived_features(df_15m)
    df_30m = add_derived_features(df_30m)
    df_1h  = add_derived_features(df_1h)

    return df_5m, df_15m, df_30m, df_1h