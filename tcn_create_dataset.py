# dataset_multiscale_cr.py
# 학습용 멀티스케일 데이터셋 생성기
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def build_multiscale_samples_cr(
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_30m: pd.DataFrame,
    df_1h: pd.DataFrame,
    feature_cols,
    seq_lens,      # {"5m": L5, "15m": L15, "30m": L30, "1h": L1h}
    horizons,      # [3, 6, 12, 24]  # 5분봉 기준 N개 뒤
    min_future_bars: int = None,
):
    """
    코인(CR) 전용 멀티스케일 샘플 생성
    - 베이스 타임프레임: 5m
    - horizons: 5m 기준 N개 뒤 수익률 (예: 3*5m=15m, 24*5m=2h)

    반환:
        X_5m, X_15m, X_30m, X_1h, Y
        각각 shape:
          - X_5m  : (N, L_5m, F)
          - X_15m : (N, L_15m, F)
          - X_30m : (N, L_30m, F)
          - X_1h  : (N, L_1h, F)
          - Y     : (N, H)  # H=len(horizons)
    """
    df_5m = df_5m.sort_index()
    df_15m = df_15m.sort_index()
    df_30m = df_30m.sort_index()
    df_1h = df_1h.sort_index()

    L_5m = seq_lens["5m"]
    L_15m = seq_lens["15m"]
    L_30m = seq_lens["30m"]
    L_1h = seq_lens["1h"]

    max_seq_len = max(L_5m, L_15m, L_30m, L_1h)
    max_h = max(horizons)
    if min_future_bars is None:
        min_future_bars = max_h

    X5_list, X15_list, X30_list, X1h_list, Y_list = [], [], [], [], []

    closes_5m = df_5m["close"].values

    # 베이스는 5m 기준 인덱스
    for i in range(max_seq_len, len(df_5m) - min_future_bars):
        base_time = df_5m.index[i]
        cur_close = closes_5m[i]

        # ------------------------
        # 1) 스케일별 시퀀스 추출
        # ------------------------
        win_5m = df_5m.iloc[i - L_5m : i][feature_cols].values

        # base_time 이전까지 중에서 tail(L)
        win_15m = (
            df_15m[df_15m.index <= base_time]
            .iloc[-L_15m:][feature_cols].values
        )
        win_30m = (
            df_30m[df_30m.index <= base_time]
            .iloc[-L_30m:][feature_cols].values
        )
        win_1h = (
            df_1h[df_1h.index <= base_time]
            .iloc[-L_1h:][feature_cols].values
        )

        if (
            len(win_5m) < L_5m
            or len(win_15m) < L_15m
            or len(win_30m) < L_30m
            or len(win_1h) < L_1h
        ):
            continue

        # ------------------------
        # 2) 미래 수익률 (5m 기준)
        # ------------------------
        future_returns = []
        for h in horizons:
            future_idx = i + h
            if future_idx >= len(df_5m):
                future_returns = None
                break
            future_price = closes_5m[future_idx]
            r = future_price / cur_close - 1.0
            future_returns.append(r)

        if future_returns is None:
            continue

        X5_list.append(win_5m)
        X15_list.append(win_15m)
        X30_list.append(win_30m)
        X1h_list.append(win_1h)
        Y_list.append(future_returns)

    if len(X5_list) == 0:
        raise ValueError("No valid samples built. Check seq_lens / horizons / data length.")

    X_5m = np.stack(X5_list, axis=0)
    X_15m = np.stack(X15_list, axis=0)
    X_30m = np.stack(X30_list, axis=0)
    X_1h = np.stack(X1h_list, axis=0)
    Y = np.stack(Y_list, axis=0)

    return X_5m, X_15m, X_30m, X_1h, Y


class MultiScaleOhlcvDatasetCR(Dataset):
    def __init__(
        self,
        X_5m: np.ndarray,
        X_15m: np.ndarray,
        X_30m: np.ndarray,
        X_1h: np.ndarray,
        Y: np.ndarray,
    ):
        super().__init__()
        self.X_5m = torch.from_numpy(X_5m).float()
        self.X_15m = torch.from_numpy(X_15m).float()
        self.X_30m = torch.from_numpy(X_30m).float()
        self.X_1h = torch.from_numpy(X_1h).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return {
            "x_5m": self.X_5m[idx],
            "x_15m": self.X_15m[idx],
            "x_30m": self.X_30m[idx],
            "x_1h": self.X_1h[idx],
            "y": self.Y[idx],
        }
