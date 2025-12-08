# bi_features.py
# BI 코인 멀티스케일 모델용 공통 Feature 정의 + 샘플 생성 유틸

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


# ==========================================================
# 1) 공통 Feature / 시퀀스 / Horizon 정의
# ==========================================================

# 모델이 보는 입력 컬럼 (학습/실전 공통)
FEATURE_COLS: List[str] = ["open", "high", "low", "close", "volume"]

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

def vectorized_window_scaling(windows):
    """
    3D 배열 (N, Window, Feature)에 대해 정규화 수행
    axis=1 (시간축) 기준으로 평균/표준편차 구함
    """
    # keepdims=True를 써야 (N, 1, F) 형태가 되어 브로드캐스팅 가능
    mean = np.mean(windows, axis=1, keepdims=True)
    std = np.std(windows, axis=1, keepdims=True) + 1e-6
    return (windows - mean) / std



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
    min_future_bars: int = None,
    return_index: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray]]:
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
