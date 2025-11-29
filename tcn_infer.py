# swing_infer_cr.py
# 멀티스케일 TCN 모델 추론 모듈 (실시간 추론 전담)
"""
CR(코인) 전용 Multi-Scale TCN + Transformer 스윙 예측 모듈

- 입력: 5분봉 OHLCV DataFrame (최근 캔들 포함)
- 내부에서 15m / 30m / 1h로 리샘플 후, 마지막 윈도우만 사용
- 출력: 5분봉 기준 [3, 6, 12] 스텝 뒤 수익률 예측 (r_3, r_6, r_12)
"""

import os
from functools import lru_cache

import numpy as np
import pandas as pd
import torch

from tcn_define_models import MultiScaleTCNTransformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATH = os.path.join("models", "multiscale_cr_model.pth")

# 학습 시와 동일하게 맞춰야 함
FEATURE_COLS = ["open", "high", "low", "close", "volume"]
HORIZONS = [3, 6, 12, 24]  

SEQ_LENS = {
    "5m": 48,
    "15m": 24,
    "30m": 16,
    "1h": 10,
}


# ----------------------------------------
# 1) 5m → 멀티스케일 리샘플
# ----------------------------------------
def _resample_ohlcv(df_5m: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = (
        df_5m.resample(rule)
        .agg(agg)
        .dropna()
    )
    return out


def _build_multiscale_windows_from_5m(df_5m: pd.DataFrame):
    """
    df_5m: datetime index, columns=[open,high,low,close,volume]
    마지막 시점 기준으로 멀티스케일 윈도우 1개를 만든다.

    반환:
        x_5m, x_15m, x_30m, x_1h  (각각 torch.Tensor, shape=(1, L, F))
    """
    if df_5m.empty:
        return None

    # datetime index 보장
    if not isinstance(df_5m.index, pd.DatetimeIndex):
        if "dt" in df_5m.columns:
            df_5m = df_5m.copy()
            df_5m["dt"] = pd.to_datetime(df_5m["dt"])
            df_5m = df_5m.set_index("dt")
        else:
            raise ValueError("df_5m에는 DatetimeIndex 또는 'dt' 컬럼이 필요합니다.")

    df_5m = df_5m.sort_index()
    df_5m = df_5m[FEATURE_COLS].apply(pd.to_numeric, errors="coerce").dropna()

    if len(df_5m) < SEQ_LENS["5m"]:
        return None

    # 5m 그대로
    win_5m = df_5m.iloc[-SEQ_LENS["5m"]:][FEATURE_COLS].values  # (L5,F)

    # 리샘플: 15m / 30m / 1h
    df_15m = _resample_ohlcv(df_5m, "15min")
    df_30m = _resample_ohlcv(df_5m, "30min")
    df_1h = _resample_ohlcv(df_5m, "60min")

    if (
        len(df_15m) < SEQ_LENS["15m"]
        or len(df_30m) < SEQ_LENS["30m"]
        or len(df_1h) < SEQ_LENS["1h"]
    ):
        return None

    win_15m = df_15m.iloc[-SEQ_LENS["15m"]:][FEATURE_COLS].values
    win_30m = df_30m.iloc[-SEQ_LENS["30m"]:][FEATURE_COLS].values
    win_1h = df_1h.iloc[-SEQ_LENS["1h"]:][FEATURE_COLS].values

    # (1,L,F) 텐서로 변환
    x_5m = torch.from_numpy(win_5m).float().unsqueeze(0)
    x_15m = torch.from_numpy(win_15m).float().unsqueeze(0)
    x_30m = torch.from_numpy(win_30m).float().unsqueeze(0)
    x_1h = torch.from_numpy(win_1h).float().unsqueeze(0)

    return x_5m, x_15m, x_30m, x_1h


# ----------------------------------------
# 2) 모델 로딩 (lazy + 캐시)
# ----------------------------------------
@lru_cache(maxsize=1)
def _load_cr_model():
    if not os.path.exists(MODEL_PATH):
        print(f"[swing_infer_cr] 모델 파일 없음: {MODEL_PATH}")
        return None

    in_features = len(FEATURE_COLS)

    model = MultiScaleTCNTransformer(
        in_features=in_features,
        horizons=HORIZONS,
        hidden_channels=64,
        tcn_layers_per_scale=4,
        transformer_layers=2,
        nhead=4,
        dropout=0.1,
        use_classification=False,
    ).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state)
    model.eval()

    print(f"[swing_infer_cr] 모델 로드 완료: {MODEL_PATH} (device={DEVICE})")
    return model


# ----------------------------------------
# 3) 외부에서 호출할 메인 함수
# ----------------------------------------
def predict_cr_swing(df_5m: pd.DataFrame):
    """
    실시간 엔트리에서 호출할 예측 함수.

    입력:
        df_5m: 5분봉 OHLCV DataFrame (datetime index 또는 dt 컬럼 포함)

    반환:
        dict 또는 None
        {
          "r_3": float,   # 3 * 5분봉 뒤 수익률
          "r_6": float,
          "r_12": float,
        }
    """
    model = _load_cr_model()
    if model is None:
        return None

    windows = _build_multiscale_windows_from_5m(df_5m)
    if windows is None:
        return None

    x_5m, x_15m, x_30m, x_1h = windows
    x_5m = x_5m.to(DEVICE)
    x_15m = x_15m.to(DEVICE)
    x_30m = x_30m.to(DEVICE)
    x_1h = x_1h.to(DEVICE)

    with torch.no_grad():
        out = model(x_5m, x_15m, x_30m, x_1h)
        reg = out["reg"].cpu().numpy()[0]  # shape: (3,)

    r_3, r_6, r_12, r_24 = reg.tolist()

    return {
        "r_3": float(r_3),
        "r_6": float(r_6),
        "r_12": float(r_12),
        "r_24": float(r_24),
    }
