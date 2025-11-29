# dataset_multiscale.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def build_multiscale_samples(
    df_1m: pd.DataFrame,
    df_5m: pd.DataFrame,
    df_15m: pd.DataFrame,
    df_1h: pd.DataFrame,
    feature_cols,
    seq_lens,      # dict: {"1m": 240, "5m": 120, "15m": 60, "1h": 48}
    horizons,      # list: [3, 6, 12, 24]  # 기준: 1분봉 n개 뒤
    min_future_bars: int = None,
):
    """
    멀티스케일 데이터셋을 만드는 helper.
    각 df는 datetime 인덱스 기준으로 정렬되어 있다고 가정.
    - feature_cols: 사용할 컬럼 리스트 (OHLCV + 파생피처)
    - seq_lens: 각 스케일별 시퀀스 길이
    - horizons: 미래 수익률을 볼 1분봉 기준 캔들 수
    """

    df_1m = df_1m.sort_index()
    df_5m = df_5m.sort_index()
    df_15m = df_15m.sort_index()
    df_1h = df_1h.sort_index()

    L_1m = seq_lens["1m"]
    L_5m = seq_lens["5m"]
    L_15m = seq_lens["15m"]
    L_1h = seq_lens["1h"]

    max_seq_len = max(L_1m, L_5m, L_15m, L_1h)
    max_h = max(horizons)
    if min_future_bars is None:
        min_future_bars = max_h

    X1_list, X5_list, X15_list, X1h_list, Y_list = [], [], [], [], []

    closes_1m = df_1m["close"].values

    # 유효한 인덱스 범위: 충분한 과거 + 충분한 미래가 있는 구간
    for i in range(max_seq_len, len(df_1m) - min_future_bars):
        base_time = df_1m.index[i]
        cur_close = closes_1m[i]

        # 1) 각 스케일별 시퀀스 추출
        # 1m
        win_1m = df_1m.iloc[i - L_1m : i][feature_cols].values  # (L_1m, F)

        # 5m/15m/1h는 base_time 이전까지 중에서 tail(L)
        win_5m = (
            df_5m[df_5m.index <= base_time].iloc[-L_5m:][feature_cols].values
        )
        win_15m = (
            df_15m[df_15m.index <= base_time].iloc[-L_15m:][feature_cols].values
        )
        win_1h = (
            df_1h[df_1h.index <= base_time].iloc[-L_1h:][feature_cols].values
        )

        # 길이가 부족하면 스킵
        if (
            len(win_1m) < L_1m
            or len(win_5m) < L_5m
            or len(win_15m) < L_15m
            or len(win_1h) < L_1h
        ):
            continue

        # 2) 미래 수익률 라벨 계산 (1분봉 기준)
        future_returns = []
        for h in horizons:
            future_idx = i + h
            if future_idx >= len(df_1m):
                future_returns = None
                break
            future_price = closes_1m[future_idx]
            r = future_price / cur_close - 1.0
            future_returns.append(r)

        if future_returns is None:
            continue

        X1_list.append(win_1m)
        X5_list.append(win_5m)
        X15_list.append(win_15m)
        X1h_list.append(win_1h)
        Y_list.append(future_returns)

    if len(X1_list) == 0:
        raise ValueError("No valid samples built. Check seq_lens / horizons / data length.")

    X_1m = np.stack(X1_list, axis=0)    # (N, L_1m, F)
    X_5m = np.stack(X5_list, axis=0)    # (N, L_5m, F)
    X_15m = np.stack(X15_list, axis=0)  # (N, L_15m, F)
    X_1h = np.stack(X1h_list, axis=0)   # (N, L_1h, F)
    Y = np.stack(Y_list, axis=0)        # (N, H)

    return X_1m, X_5m, X_15m, X_1h, Y


class MultiScaleOhlcvDataset(Dataset):
    def __init__(
        self,
        X_1m: np.ndarray,
        X_5m: np.ndarray,
        X_15m: np.ndarray,
        X_1h: np.ndarray,
        Y: np.ndarray,
    ):
        """
        X_*: (N, L, F)
        Y: (N, H)  # multi-horizon 수익률
        """
        super().__init__()
        self.X_1m = torch.from_numpy(X_1m).float()
        self.X_5m = torch.from_numpy(X_5m).float()
        self.X_15m = torch.from_numpy(X_15m).float()
        self.X_1h = torch.from_numpy(X_1h).float()
        self.Y = torch.from_numpy(Y).float()

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        return {
            "x_1m": self.X_1m[idx],
            "x_5m": self.X_5m[idx],
            "x_15m": self.X_15m[idx],
            "x_1h": self.X_1h[idx],
            "y": self.Y[idx],
        }
