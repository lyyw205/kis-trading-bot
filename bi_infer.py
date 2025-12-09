# bi_infer.py (또는 swing_infer_cr.py)
# 멀티스케일 TCN 모델 추론 모듈 (실시간 / 배치 공통)

import os
from functools import lru_cache
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import glob
from bi_define_models import MultiScaleTCNTransformer
from bi_features import (                   
    FEATURE_COLS,
    HORIZONS,
    SEQ_LENS,
    build_multiscale_samples_cr,
    resample_from_5m,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# # 학습 시와 동일 설정
# FEATURE_COLS = ["open", "high", "low", "close", "volume"]
# HORIZONS = [3, 6, 12, 24]  
# SEQ_LENS = {
#     "5m": 48,
#     "15m": 24,
#     "30m": 16,
#     "1h": 10,
# }

def get_latest_model(pattern="models/multiscale*_model_*.pth"):
    files = glob.glob(pattern)
    if not files:
        return None
    return max(files, key=os.path.getmtime)

MODEL_PATH = get_latest_model()
if MODEL_PATH is None:
    print("[bi_infer] 사용 가능한 모델 없음!")
else:
    print(f"[bi_infer] 최신 모델 사용: {MODEL_PATH}")


# ----------------------------------------
# 1) 모델 로딩
# ----------------------------------------
@lru_cache(maxsize=1)
def _load_cr_model():
    if MODEL_PATH is None:
        print("[bi_infer] MODEL_PATH=None → 모델 로드 스킵")
        return None

    if not os.path.exists(MODEL_PATH):
        print(f"[bi_infer] 모델 파일 없음: {MODEL_PATH}")
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
        use_classification=True,
        use_trade_head=True,    # ✅ 학습 코드와 동일하게
    ).to(DEVICE)

    state = torch.load(MODEL_PATH, map_location=DEVICE)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[bi_infer] state_dict mismatch | missing={missing} | unexpected={unexpected}")

    model.eval()
    return model


# ----------------------------------------
# 2) 멀티스케일 입력 생성 (df_5m → 마지막 샘플 1개)
# ----------------------------------------
def make_multiscale_inputs_from_5m_df(df_5m: pd.DataFrame):
    """
    5분봉 df_5m만 받아서:
      1) 인덱스 정리 (DatetimeIndex)
      2) resample_from_5m()으로 15m/30m/1h 생성
      3) build_multiscale_samples_cr()로 전체 샘플 생성
      4) 마지막 샘플 1개만 뽑아서 (1, L, F) 텐서로 반환
    """
    if df_5m is None or df_5m.empty:
        return None

    # 인덱스 정리
    if not isinstance(df_5m.index, pd.DatetimeIndex):
        if "dt" in df_5m.columns:
            df_5m = df_5m.copy()
            df_5m["dt"] = pd.to_datetime(df_5m["dt"])
            df_5m = df_5m.set_index("dt")
        else:
            return None

    df_5m = df_5m.sort_index()

    # 최소 길이 체크 (학습과 동일한 기준으로)
    min_len = max(SEQ_LENS["5m"], max(HORIZONS) + 10)
    if len(df_5m) < min_len:
        return None

    # 15m/30m/1h 리샘플 (공통 유틸)
    df_5m, df_15m, df_30m, df_1h = resample_from_5m(df_5m)

    try:
        X_5m, X_15m, X_30m, X_1h, Y, base_dt = build_multiscale_samples_cr(
            df_5m=df_5m,
            df_15m=df_15m,
            df_30m=df_30m,
            df_1h=df_1h,
            feature_cols=FEATURE_COLS,
            seq_lens=SEQ_LENS,
            horizons=HORIZONS,
            return_index=True,
        )
    except ValueError:
        # 데이터 길이 부족 등
        return None

    if len(X_5m) == 0:
        return None

    # 마지막 샘플만 선택 → (1, L, F) 텐서
    x_5m = torch.from_numpy(X_5m[-1:]).float().to(DEVICE)
    x_15m = torch.from_numpy(X_15m[-1:]).float().to(DEVICE)
    x_30m = torch.from_numpy(X_30m[-1:]).float().to(DEVICE)
    x_1h = torch.from_numpy(X_1h[-1:]).float().to(DEVICE)

    return x_5m, x_15m, x_30m, x_1h


# ----------------------------------------
# 3) 실시간 단일 추론
# ----------------------------------------
def predict_bi_swing(df_5m: pd.DataFrame):
    model = _load_cr_model()
    if model is None:
        return None

    windows = make_multiscale_inputs_from_5m_df(df_5m)
    if windows is None:
        return None

    x_5m, x_15m, x_30m, x_1h = windows

    with torch.no_grad():
        out = model(x_5m, x_15m, x_30m, x_1h)
        reg = out["reg"].cpu().numpy()[0]

    return {
        "r_3": float(reg[0]),
        "r_6": float(reg[1]),
        "r_12": float(reg[2]),
        "r_24": float(reg[3]),
    }

# # ----------------------------------------
# # 4) [NEW] 백테스트용 대용량 배치 추론
# # ----------------------------------------
# def predict_cr_swing_batch(df_5m: pd.DataFrame, batch_size=512) -> dict:
#     """
#     df_5m 전체에 대해 Sliding Window 방식으로 입력을 만들어
#     한번에(Batch) 추론하고 결과를 딕셔너리로 반환합니다.
#     """
#     model = _load_cr_model()
#     # 결과 담을 배열 (NaN으로 초기화)
#     n = len(df_5m)
#     res = {
#         "r_3": np.full(n, np.nan),
#         "r_6": np.full(n, np.nan),
#         "r_12": np.full(n, np.nan)
#     }
    
#     if model is None or n < 200:
#         return res

#     # 1. 데이터 정리
#     df = df_5m.copy()
#     if not isinstance(df.index, pd.DatetimeIndex):
#         if "dt" in df.columns:
#             df["dt"] = pd.to_datetime(df["dt"])
#             df = df.set_index("dt")
#     df = df.sort_index()
    
#     df_raw = df[FEATURE_COLS].apply(pd.to_numeric, errors='coerce').fillna(0)
    
#     # 5m Data (numpy)
#     data_5m = df_raw.values
    
#     # 15m Data
#     df_15m = _resample_ohlcv(df, "15min")
#     data_15m = df_15m[FEATURE_COLS].reindex(df.index, method='ffill').fillna(0).values
    
#     # 30m Data
#     df_30m = _resample_ohlcv(df, "30min")
#     data_30m = df_30m[FEATURE_COLS].reindex(df.index, method='ffill').fillna(0).values
    
#     # 1h Data
#     df_1h = _resample_ohlcv(df, "60min")
#     data_1h = df_1h[FEATURE_COLS].reindex(df.index, method='ffill').fillna(0).values
    
#     # Sliding Window 생성 기준점 (가장 긴 윈도우)
#     min_idx = max(SEQ_LENS["5m"], SEQ_LENS["15m"], SEQ_LENS["30m"], SEQ_LENS["1h"])
    
#     valid_indices = np.arange(min_idx, n)
#     if len(valid_indices) == 0:
#         return res

#     # Pytorch Dataset
#     class MultiScaleDataset(torch.utils.data.Dataset):
#         def __init__(self, idxs, d5, d15, d30, d1h):
#             self.idxs = idxs
#             self.d5 = d5
#             self.d15 = d15
#             self.d30 = d30
#             self.d1h = d1h
            
#             self.l5 = SEQ_LENS["5m"]
#             self.l15 = SEQ_LENS["15m"]
#             self.l30 = SEQ_LENS["30m"]
#             self.l1h = SEQ_LENS["1h"]
            
#         def __len__(self):
#             return len(self.idxs)
        
#         def __getitem__(self, i):
#             curr = self.idxs[i]
#             # Slicing
#             x5 = self.d5[curr-self.l5 : curr]
#             x15 = self.d15[curr-self.l15 : curr]
#             x30 = self.d30[curr-self.l30 : curr]
#             x1h = self.d1h[curr-self.l1h : curr]
            
#             # [수정] 배치 추론 시에도 정규화 적용!
#             x5 = apply_window_scaling(x5)
#             x15 = apply_window_scaling(x15)
#             x30 = apply_window_scaling(x30)
#             x1h = apply_window_scaling(x1h)
            
#             return (
#                 torch.tensor(x5, dtype=torch.float32),
#                 torch.tensor(x15, dtype=torch.float32),
#                 torch.tensor(x30, dtype=torch.float32),
#                 torch.tensor(x1h, dtype=torch.float32)
#             )

#     ds = MultiScaleDataset(valid_indices, data_5m, data_15m, data_30m, data_1h)
#     loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)

#     all_preds = []
    
#     with torch.no_grad():
#         for x5, x15, x30, x1h in loader:
#             x5 = x5.to(DEVICE)
#             x15 = x15.to(DEVICE)
#             x30 = x30.to(DEVICE)
#             x1h = x1h.to(DEVICE)
            
#             out = model(x5, x15, x30, x1h)
#             preds = out["reg"].cpu().numpy()
#             all_preds.append(preds)
            
#     if not all_preds:
#         return res
        
#     all_preds = np.concatenate(all_preds, axis=0) 
    
#     res["r_3"][valid_indices] = all_preds[:, 0]
#     res["r_6"][valid_indices] = all_preds[:, 1]
#     res["r_12"][valid_indices] = all_preds[:, 2]
    
#     return res