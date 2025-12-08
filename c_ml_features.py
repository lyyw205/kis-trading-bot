# "시계열 ML 입력 피처 생성 유틸 (공통)

#  - 트레이더 / 학습 스크립트에서 공통으로 사용하는
#    RSI 계산 + 캔들 시퀀스(feature vector) 생성 모듈.

# 주요 기능:

# 1) SEQ_LEN = 30
#    - ML 모델에 넣을 시퀀스 길이 (캔들 개수)
#    - trader.py, build_ml_seq_samples.py 등과 동일 값 유지해야 함.

# 2) calculate_rsi(series, period=14) -> pd.Series
#    - 종가 시계열을 입력으로 받아 RSI(상대강도지수)를 계산.
#    - delta = diff, 양수 구간만 평균 → gain, 음수 구간만 평균 → loss
#    - RSI = 100 - 100 / (1 + RS), RS = gain / loss
#    - 초기 구간 NaN은 50으로 채워서 중립값으로 사용.

# 3) build_feature_from_seq(df_seq, seq_len=SEQ_LEN) -> np.ndarray | None
#    - 길이가 seq_len인 캔들 DataFrame을 받아서 1D feature vector로 변환.
#    - 기대 컬럼: ["close", "high", "low", "volume"]
#    - 첫 번째 close를 기준 가격(base)으로 두고:
#        · close_rel = close/base - 1
#        · high_rel  = high/base - 1
#        · low_rel   = low/base - 1
#    - volume은 평균으로 나눠서 정규화:
#        · vol_norm = volume / vol_mean (vol_mean<=0이면 1.0으로 방어)
#    - 최종 feature:
#        [close_rel 30개, high_rel 30개, low_rel 30개, vol_norm 30개] → 길이 120짜리 벡터
#    - 길이가 맞지 않거나 base<=0이면 None 반환 (유효하지 않은 시퀀스 방어).

# → 결과적으로, 이 파일은
#    '시계열 캔들을 ML 모델이 바로 먹을 수 있는 벡터로 변환해주는 공용 유틸' 역할을 한다."

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
