# ta_utils.py
"""
공용 기술적 지표 유틸 모듈 (ATR 등)
"""

from typing import Optional
import pandas as pd
import numpy as np


def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    ATR(Average True Range) 계산

    df: columns = ["high", "low", "close"] 필수
    period: ATR 기간 (기본 14)
    """
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift(1)).abs()
    low_close = (df["low"] - df["close"].shift(1)).abs()

    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    return atr
