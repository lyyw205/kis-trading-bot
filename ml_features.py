# ml_features.py
import numpy as np
import pandas as pd

# trader.py와 동일하게 맞춰야 함
SEQ_LEN = 30   # 시퀀스 길이


def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI(상대강도지수) 계산.
    trader.py의 calculate_rsi와 동일 로직.
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # 초기값은 50으로 처리


def build_feature_from_seq(df_seq: pd.DataFrame, seq_len: int = SEQ_LEN):
    """
    최근 seq_len개 캔들을 기반으로 feature 생성.
    trader.py의 build_feature_from_seq와 동일 로직.
    """
    if len(df_seq) != seq_len:
        return None

    close = df_seq["close"].values
    high = df_seq["high"].values
    low = df_seq["low"].values
    vol = df_seq["volume"].values

    base = close[0]
    if base <= 0:
        return None

    close_rel = close / base - 1.0
    high_rel = high / base - 1.0
    low_rel = low / base - 1.0

    vol_mean = np.mean(vol) if np.mean(vol) > 0 else 1.0
    vol_norm = vol / vol_mean

    feat = np.concatenate([close_rel, high_rel, low_rel, vol_norm])
    return feat
