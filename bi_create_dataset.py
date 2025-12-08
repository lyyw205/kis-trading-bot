# bi_create_dataset.py
# (최적화 버전: Python Loop 제거 및 벡터 연산 적용)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from numpy.lib.stride_tricks import sliding_window_view



class MultiScaleOhlcvDatasetCR(Dataset):
    def __init__(
        self,
        X_5m,
        X_15m,
        X_30m,
        X_1h,
        Y,
        y_trade=None,       # ✅ 추가 (numpy or None)
        trade_mask=None,    # ✅ 추가 (numpy or None)
    ):
        # 이미 numpy array 상태로 넘어오므로 바로 텐서 변환
        self.X_5m = torch.from_numpy(X_5m).float()
        self.X_15m = torch.from_numpy(X_15m).float()
        self.X_30m = torch.from_numpy(X_30m).float()
        self.X_1h = torch.from_numpy(X_1h).float()
        self.Y = torch.from_numpy(Y).float()

        # ✅ 포지션 기반 라벨이 있으면 텐서로 변환
        if y_trade is not None:
            self.y_trade = torch.from_numpy(y_trade).float()
            self.trade_mask = torch.from_numpy(trade_mask).float()
        else:
            self.y_trade = None
            self.trade_mask = None

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, idx):
        item = {
            "x_5m": self.X_5m[idx],
            "x_15m": self.X_15m[idx],
            "x_30m": self.X_30m[idx],
            "x_1h": self.X_1h[idx],
            "y": self.Y[idx],
        }

        # ✅ 포지션 라벨이 있는 경우에만 dict에 추가
        if self.y_trade is not None:
            item["y_trade"] = self.y_trade[idx]
            item["trade_mask"] = self.trade_mask[idx]

        return item