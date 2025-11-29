# cr_swing_infer.py
"""
CR Swing 모델 실전 예측 헬퍼

- 역할:
  1) 학습된 모델(checkpoint) 로드
  2) 최근 5m OHLCV DataFrame에서 feature 생성
  3) 마지막 60개 시퀀스로 [r_3, r_6, r_12] 예측
"""

import os
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch

from swing_model_cr import CrSwingModel, CrSwingConfig
from utils import calculate_atr  # 이미 거기 정의돼있으면 재사용


# -----------------------------
# 전역 캐시
# -----------------------------
_CR_SWING_MODEL: Optional[CrSwingModel] = None
_CR_SWING_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 1) 모델 로드
# -----------------------------
def load_cr_swing_model(
    ckpt_path: str = "models/cr_swing/cr_swing_best.pt",
) -> Optional[CrSwingModel]:
    global _CR_SWING_MODEL

    if _CR_SWING_MODEL is not None:
        return _CR_SWING_MODEL

    if not os.path.exists(ckpt_path):
        print(f"[CR_SWING] 체크포인트 없음: {ckpt_path}")
        return None

    state = torch.load(ckpt_path, map_location=_CR_SWING_DEVICE)
    cfg_dict = state.get("config", {})

    cfg = CrSwingConfig(**cfg_dict)
    model = CrSwingModel(cfg).to(_CR_SWING_DEVICE)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    _CR_SWING_MODEL = model
    print(f"[CR_SWING] 모델 로드 완료: {ckpt_path} (device={_CR_SWING_DEVICE})")
    return _CR_SWING_MODEL


# -----------------------------
# 2) feature 생성 (실전용)
# -----------------------------
def build_cr_swing_features(df_raw: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    df_raw: 최소 60 + 롤링워밍 몇 개 이상 5m OHLCV (symbol 단일)
      - 필수 컬럼: dt, open, high, low, close, volume

    반환: feature 붙은 df_feat (NaN 정리 후)
    """
    if df_raw is None or df_raw.empty:
        return None

    df = df_raw.copy()
    if "dt" not in df.columns:
        raise ValueError("df_raw 에 'dt' 컬럼이 필요합니다.")
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.sort_values("dt").reset_index(drop=True)

    # 심볼은 하나라고 가정
    symbol = df.get("symbol", None)
    if symbol is None:
        df["symbol"] = "CR_UNKNOWN"
    else:
        # 전부 같은 값인지 한 번 체크
        if df["symbol"].nunique() > 1:
            # 심볼 섞여 있으면 일단 마지막 심볼만 사용
            last_sym = df["symbol"].iloc[-1]
            df = df[df["symbol"] == last_sym].copy()

    # === 데이터셋 빌더와 동일한 feature 계산 ===
    df_feat = df.copy()

    df_feat["ma20"] = df_feat["close"].rolling(20).mean()
    df_feat["ma60"] = df_feat["close"].rolling(60).mean()
    df_feat["vol_ma20"] = df_feat["volume"].rolling(20).mean()

    try:
        df_feat["atr"] = calculate_atr(df_feat, period=14)
    except Exception:
        df_feat["atr"] = np.nan

    df_feat["hl_range"] = (df_feat["high"] - df_feat["low"]) / df_feat["close"].replace(0, np.nan)
    df_feat["ret_1"] = df_feat["close"].pct_change()

    # rolling NaN 구간 제거
    df_feat = df_feat.dropna(
        subset=["ma20", "ma60", "vol_ma20", "atr", "hl_range", "ret_1"]
    ).reset_index(drop=True)

    if len(df_feat) < 60:
        # 시퀀스 길이 부족
        return None

    # feature 순서 (데이터셋과 동일)
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


# -----------------------------
# 3) 예측 함수
# -----------------------------
@torch.no_grad()
def predict_cr_swing(
    df_raw: pd.DataFrame,
    ckpt_path: str = "models/cr_swing/cr_swing_best.pt",
) -> Optional[Dict[str, Any]]:
    """
    최근 5m OHLCV df_raw에서 마지막 60개 시퀀스로 [r_3, r_6, r_12] 예측.

    반환 예:
      {
        "r_3":  0.0031,
        "r_6":  0.0062,
        "r_12": 0.0105,
        "raw": np.array([...]),  # (3,)
      }
    """
    model = load_cr_swing_model(ckpt_path)
    if model is None:
        return None

    df_feat = build_cr_swing_features(df_raw)
    if df_feat is None or len(df_feat) < 60:
        return None

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
    values = df_feat[feature_cols].values  # (n, feat_dim)
    seq = values[-60:]  # 마지막 60개

    X = torch.from_numpy(seq).float().unsqueeze(0).to(_CR_SWING_DEVICE)  # (1, 60, feat_dim)
    pred = model(X)  # (1, 3)
    pred = pred.squeeze(0).cpu().numpy()  # (3,)

    return {
        "r_3": float(pred[0]),
        "r_6": float(pred[1]),
        "r_12": float(pred[2]),
        "raw": pred,
    }
