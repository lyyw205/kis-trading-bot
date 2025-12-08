# "멀티 자산군(KR/US/CR) 공통 ML 시퀀스 라벨링 스크립트

#  - ohlcv_data 테이블에 저장된 5분봉(기타 interval도 가능)을 기반으로
#    엔트리 신호가 발생한 시점마다 향후 수익(TP/SL)을 보고 0/1 라벨을 붙여
#    ml_seq_samples 테이블에 학습용 레이블 데이터를 만드는 유틸 스크립트.

# 주요 기능:

# 1) OHLCV 전체 로드
#    - load_ohlcv_all()
#      · sqlite3 로 trading.db의 ohlcv_data 전체 조회
#      · region, symbol, interval로 groupby
#      · 각 그룹을 datetime index + [open, high, low, close, volume] DataFrame으로 정리
#      · 딕셔너리 key = (region, symbol, interval) 형태로 반환

# 2) TP/SL 기반 라벨링
#    - calc_label(future_df, entry_price) -> int
#      · FUTURE_WINDOW(20봉) 동안의 close를 보면서
#        · 수익률 >= TP_RATE(2%) 도달 → 1 (성공)
#        · 수익률 <= SL_RATE(-2%) 도달 → 0 (실패)
#        · 둘 다 안 나오면 기본 0 처리

# 3) 유니버스별 샘플 생성
#    - build_samples_for_universe(ohlcv_dict, universe, interval='5m')
#      · universe: config의 KR_UNIVERSE_STOCKS / US_UNIVERSE_STOCKS / CR_UNIVERSE_STOCKS 등
#      · region에 따라 라벨링 방식 분기:

#      [KR / US]
#        - add_common_entry_columns(df, params) 로 공통 엔트리 신호 컬럼 추가
#        - entry_signal == True 인 시점만 엔트리 포인트로 사용
#        - 각 엔트리 시점마다 미래 FUTURE_WINDOW 윈도우를 보고 calc_label()로 0/1 라벨 생성
#        - samples에 {region, symbol, interval, dt_entry, label} dict 추가

#      [CR]
#        - 코인은 시퀀스 기반 엔트리 사용
#        - df를 슬라이딩 윈도우(길이 SEQ_LEN)로 잘라서 df_seq 구성
#        - 각 df_seq에 대해 make_entry_signal_coin_ms(df_seq, params) 호출
#        - entry_signal=True 인 구간만 엔트리로 채택
#        - 시퀀스 마지막 캔들의 close를 entry_price로 보고,
#          뒤 FUTURE_WINDOW 캔들로 TP/SL 라벨 계산
#        - 동일하게 samples 리스트에 dict 형태로 적재

# 4) 메인 실행 플로우 (__main__)
#    - BotDatabase로 로그 기록
#    - ohlcv_dict = load_ohlcv_all()
#    - KR/US/CR 유니버스 각각에 대해 build_samples_for_universe 호출:
#        · kr_samples / us_samples / cr_samples
#    - save_samples(region_code, samples)
#        · sqlite3.connect(DB_PATH)로 ml_seq_samples 테이블에 저장
#        · 같은 region 샘플은 기존 것 DELETE 후 INSERT로 덮어쓰기
#    - 최종적으로
#        · KR / US / CR 별 샘플 개수 로그
#        · 'ML 시퀀스 샘플 생성 + 저장 완료' 로그 출력

# → 결과적으로 이 파일은
#    'KR/US 룰 기반 엔트리 + CR TCN 엔트리'를 기준으로
#    미래 TP/SL 성과를 라벨링한 시퀀스 엔트리 데이터셋을 만들어주는
#    공통 ML 학습용 전처리 스크립트."


import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd

from c_config import KR_UNIVERSE_STOCKS, US_UNIVERSE_STOCKS, CR_UNIVERSE_STOCKS
from c_db_manager import BotDatabase
from c_st_entry import add_common_entry_columns
from bi_entry_lib import make_entry_signal_coin_ms
from c_ml_features import SEQ_LEN

DB_PATH = "trading.db"

LOOKBACK = 20
BAND_PCT = 0.005

FUTURE_WINDOW = 20
TP_RATE = 0.02
SL_RATE = -0.02


# -------------------------------------------------------
# OHLCV 전체 로드 (그대로 사용)
# -------------------------------------------------------
def load_ohlcv_all():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT region, symbol, interval, dt,
               open, high, low, close, volume
        FROM ohlcv_data
        ORDER BY region, symbol, interval, dt
        """,
        conn,
    )
    conn.close()

    if df.empty:
        return {}

    df["dt"] = pd.to_datetime(df["dt"])

    groups = {}
    for (region, symbol, interval), g in df.groupby(
        ["region", "symbol", "interval"], sort=False
    ):
        g = g.copy().sort_values("dt")
        g.set_index("dt", inplace=True)

        g = g[["open", "high", "low", "close", "volume"]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()

        if len(g) == 0:
            continue

        groups[(region, symbol, interval)] = g

    return groups


# -------------------------------------------------------
# TP/SL 기반 라벨 계산 (그대로)
# -------------------------------------------------------
def calc_label(future_df: pd.DataFrame, entry_price: float) -> int:
    for _, row in future_df.iterrows():
        price = row["close"]
        profit = (price - entry_price) / entry_price

        if profit >= TP_RATE:
            return 1
        if profit <= SL_RATE:
            return 0
    return 0


# -------------------------------------------------------
# ⭐ 유니버스별 샘플 생성 코어 함수
# -------------------------------------------------------
def build_samples_for_universe(ohlcv_dict: dict, universe, *, interval="5m"):
    """
    하나의 유니버스(KR / US / CR)에 대해 샘플을 생성한다.

    - universe: [{"region": "...", "symbol": "...", "excd": "..."}, ...]
    - 반환: samples 리스트 (기존과 동일 포맷)
    """
    samples: list[dict] = []

    base_params = {
        "lookback": LOOKBACK,
        "band_pct": BAND_PCT,
    }

    for t in universe:
        region = t["region"]
        symbol = t["symbol"]

        key = (region, symbol, interval)
        if key not in ohlcv_dict:
            continue

        df = ohlcv_dict[key].copy()
        if len(df) < max(60, SEQ_LEN + FUTURE_WINDOW):
            # 너무 짧으면 건너뜀
            continue

        # ================================
        # 1) KR / US : 기존 공통 엔트리 그대로
        # ================================
        if region in ("KR", "US"):
            params = base_params

            df_k = add_common_entry_columns(df.copy(), params)
            df_k = df_k.dropna(subset=["entry_signal", "close"])

            entry_points = df_k[df_k["entry_signal"]]
            if entry_points.empty:
                continue

            for dt_entry, row in entry_points.iterrows():
                entry_price = row["close"]

                try:
                    idx = df_k.index.get_loc(dt_entry)
                except KeyError:
                    continue

                if idx + 1 + FUTURE_WINDOW > len(df_k):
                    continue

                future_df = df_k.iloc[idx + 1 : idx + 1 + FUTURE_WINDOW]
                if future_df.empty:
                    continue

                label = calc_label(future_df, entry_price)

                samples.append(
                    {
                        "region": region,
                        "symbol": symbol,
                        "interval": interval,
                        "dt_entry": dt_entry.strftime("%Y-%m-%d %H:%M:%S"),
                        "label": int(label),
                    }
                )

        # ================================
        # 2) CR : st_entry_coin 기반 강화 엔트리 사용
        # ================================
        elif region == "CR":
            params = base_params  # CR에서도 동일 파라미터 전달 (ATR, HL 필터는 내부 default 사용)

            # 인덱스 기준으로 슬라이딩 시퀀스
            df_c = df.copy()

            # idx: 시퀀스의 마지막 캔들 인덱스 (엔트리 시점)
            # 이후 FUTURE_WINDOW 만큼 미래 캔들로 라벨 계산
            last_possible_idx = len(df_c) - FUTURE_WINDOW - 1
            for end_idx in range(SEQ_LEN - 1, last_possible_idx + 1):
                start_idx = end_idx - SEQ_LEN + 1
                df_seq = df_c.iloc[start_idx : end_idx + 1].copy()
                if len(df_seq) < SEQ_LEN:
                    continue

                sig = make_entry_signal_coin_ms(df_seq, params)
                if not sig.get("entry_signal", False):
                    continue

                dt_entry = df_seq.index[-1]
                entry_price = float(df_seq["close"].iloc[-1])

                future_df = df_c.iloc[end_idx + 1 : end_idx + 1 + FUTURE_WINDOW]
                if future_df.empty:
                    continue

                label = calc_label(future_df, entry_price)

                samples.append(
                    {
                        "region": region,
                        "symbol": symbol,
                        "interval": interval,
                        "dt_entry": dt_entry.strftime("%Y-%m-%d %H:%M:%S"),
                        "label": int(label),
                    }
                )

        # 혹시 다른 region 이 추가되면 일단 스킵
        else:
            continue

    return samples


# -------------------------------------------------------
# (옵션) 메인: KR / US / CR 각각 호출 예시
# -------------------------------------------------------
if __name__ == "__main__":
    db = BotDatabase(DB_PATH)
    db.log("📦 ML 시퀀스 샘플 생성 시작 (KR/US/CR 분리)")

    ohlcv_dict = load_ohlcv_all()
    if not ohlcv_dict:
        db.log("⚠️ ohlcv_data 비어 있음. 먼저 OHLCV 백필 필요.")
        exit(0)

    # 1) KR
    kr_samples = build_samples_for_universe(ohlcv_dict, KR_UNIVERSE_STOCKS)
    db.log(f"✅ KR 샘플 수: {len(kr_samples)}")

    # 2) US
    us_samples = build_samples_for_universe(ohlcv_dict, US_UNIVERSE_STOCKS)
    db.log(f"✅ US 샘플 수: {len(us_samples)}")

    # 3) CR (코인)
    cr_samples = build_samples_for_universe(ohlcv_dict, CR_UNIVERSE_STOCKS)
    db.log(f"✅ CR 샘플 수: {len(cr_samples)}")

    # 2) DB 저장 함수
    def save_samples(region_code: str, samples: list[dict]):
        if not samples:
            db.log(f"⚠️ {region_code} 샘플 없음 → 저장 스킵")
            return

        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()

        # region별로 기존 샘플 삭제 (매번 새로 덮어쓰는 방식)
        cur.execute(
            "DELETE FROM ml_seq_samples WHERE region = ?",
            (region_code,),
        )

        cur.executemany(
            """
            INSERT INTO ml_seq_samples (region, symbol, interval, dt_entry, label)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    s["region"],
                    s["symbol"],
                    s["interval"],
                    s["dt_entry"],
                    s["label"],
                )
                for s in samples
            ],
        )

        conn.commit()
        conn.close()
        db.log(f"💾 {region_code} 샘플 {len(samples)}개 저장 완료")

    # 3) KR / US / CR 각각 저장
    save_samples("KR", kr_samples)
    save_samples("US", us_samples)
    save_samples("CR", cr_samples)

    db.log("🎉 ML 시퀀스 샘플 생성 + 저장 완료")
