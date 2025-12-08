# market_regime_coin.py
"""
코인 마켓 레짐(BULL / BEAR / NEUTRAL) 계산 모듈 (BI / Binance 전용)

- BTCUSDT + 상위 코인들 4시간봉 기준으로 시장 상태를 판단
- 6시간마다 run_update_market_regime_coin.py 같은 스크립트에서 호출해서
  settings 테이블에 결과를 저장하는 용도로 사용

저장 키:
  - market_regime_coin               : "BULL" | "BEAR" | "NEUTRAL"
  - market_regime_coin_updated_at    : "YYYY-MM-DD HH:MM:SS"
  - market_regime_coin_avg_return_1d : 상위 코인 1일 평균 수익률 (소수, 예: 0.012345 → 1.23%)
  - market_regime_coin_breadth_ma50  : MA50 하회 비율 (0.0 ~ 1.0)
"""

from datetime import datetime
from typing import Dict, Any, List

import numpy as np
import pandas as pd

from c_db_manager import BotDatabase
from bi_multiscale_loader import load_ohlcv_multiscale_for_symbol


# ============================================================
# 헬퍼: 1시간봉 → 4시간봉 리샘플
# ============================================================
def _ensure_dt_index(df: pd.DataFrame, col: str = "dt") -> pd.DataFrame:
    """
    - df.index가 DatetimeIndex가 아니면 col을 기준으로 DatetimeIndex로 변환
    - 정렬까지 수행
    """
    if df is None or df.empty:
        return df

    if isinstance(df.index, pd.DatetimeIndex):
        return df.sort_index()

    if col in df.columns:
        df = df.copy()
        df[col] = pd.to_datetime(df[col])
        df = df.set_index(col)
        df = df.sort_index()
        return df

    raise ValueError("DatetimeIndex 또는 'dt' 컬럼이 필요합니다.")


def resample_1h_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
    """
    1시간봉 DF를 4시간봉으로 리샘플링
    index: DatetimeIndex
    columns: open, high, low, close, volume
    """
    if df_1h is None or df_1h.empty:
        return pd.DataFrame()

    df_1h = _ensure_dt_index(df_1h)

    agg_dict = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    df_4h = df_1h.resample("4h").agg(agg_dict).dropna()
    return df_4h


# ============================================================
# 레짐 조건 계산
# ============================================================
def is_btc_downtrend(df_4h: pd.DataFrame) -> bool:
    """
    BTCUSDT 4h 기준 하락 추세 여부
      - close < ema200
      - ema50 < ema200
    """
    df = df_4h.copy()
    df["ema50"] = df["close"].ewm(span=50).mean()
    df["ema200"] = df["close"].ewm(span=200).mean()
    last = df.iloc[-1]

    cond_price_below = last["close"] < last["ema200"]
    cond_ma_down = last["ema50"] < last["ema200"]

    return bool(cond_price_below and cond_ma_down)


def calc_breadth_below_ma(
    df_dict_4h: Dict[str, pd.DataFrame],
    ma_window: int = 50,
) -> float:
    """
    상위 N개 코인 중 몇 %가 해당 MA 아래에 있는지 계산
    - 각 심볼별로 마지막 close < MA(window) 인지 체크
    """
    total = 0
    below = 0

    for sym, df in df_dict_4h.items():
        if df is None or df.empty:
            continue
        if len(df) < ma_window + 5:
            continue

        ma = df["close"].rolling(ma_window).mean()
        last_close = df["close"].iloc[-1]
        last_ma = ma.iloc[-1]
        if np.isnan(last_ma):
            continue

        total += 1
        if last_close < last_ma:
            below += 1

    if total == 0:
        return 0.0

    return below / total  # 0~1


def calc_universe_avg_return_1d(df_dict_4h: Dict[str, pd.DataFrame]) -> float:
    """
    상위 코인들에 대해 4h 기준 1일 수익률 평균 계산
      - 심볼별: 마지막 close / 6개 전 close - 1  (4h * 6 = 24h)
      - 전체 평균 리턴 (없으면 0.0)
    """
    rets: List[float] = []

    for sym, df in df_dict_4h.items():
        if df is None or df.empty:
            continue
        # 4h * 6 = 24h
        if len(df) < 7:
            continue

        last_close = float(df["close"].iloc[-1])
        prev_close = float(df["close"].iloc[-7])
        if prev_close <= 0:
            continue

        ret = last_close / prev_close - 1.0
        rets.append(ret)

    if not rets:
        return 0.0

    return float(np.mean(rets))


def detect_market_regime_coin(
    btc_df_4h: pd.DataFrame,
    universe_4h_dict: Dict[str, pd.DataFrame],
) -> Dict[str, Any]:
    """
    마켓 레짐 계산 로직

    반환:
        {
          "regime": "BULL/BEAR/NEUTRAL",
          "score": int,
          "breadth": float,
          "cond_detail": {...}
        }
    """
    # 조건1: BTC 4h 하락 추세
    cond1 = is_btc_downtrend(btc_df_4h)

    # 조건2: 상위 코인들의 MA50 하회 비율
    breadth = calc_breadth_below_ma(universe_4h_dict, ma_window=50)
    cond2 = breadth >= 0.7  # 예: 70% 이상이 MA 아래면 하락장

    # 조건3: 최근 N개 4h 봉 동안 BTC 종가가 ema20 아래
    df = btc_df_4h.copy()
    df["ema20"] = df["close"].ewm(span=20).mean()
    recent = df.tail(18)  # 4h * 18 ≒ 3일
    cond3 = (recent["close"] < recent["ema20"]).all()

    score = int(cond1) + int(cond2) + int(cond3)

    if score >= 2:
        regime = "BEAR"
    elif score == 0:
        regime = "BULL"
    else:
        regime = "NEUTRAL"

    return {
        "regime": regime,
        "score": score,
        "breadth": breadth,
        "cond_detail": {
            "cond1_btc_down": cond1,
            "cond2_breadth_ge_0_7": cond2,
            "cond3_btc_below_ema20_recent": cond3,
        },
    }


# ============================================================
# 데이터 로딩 헬퍼 (BI / Binance 전용)
# ============================================================
def load_universe_4h_data(
    db: BotDatabase,
    symbols: List[str],
    region: str = "BI",
) -> Dict[str, pd.DataFrame]:
    """
    trading.db 의 ohlcv_data에서 4h 데이터 로드
    - 내부적으로 load_ohlcv_multiscale_for_symbol(region, symbol, base_interval="5m") 사용
    - 1h를 4h로 리샘플
    """
    result: Dict[str, pd.DataFrame] = {}

    for sym in symbols:
        try:
            df_5m, df_15m, df_30m, df_1h = load_ohlcv_multiscale_for_symbol(
                region=region,
                symbol=sym,
                base_interval="5m",
            )
        except Exception as e:
            db.log(f"⚠️ [REGIME] {region} {sym} 4h 로딩 실패: {e}")
            continue

        if df_1h is None or df_1h.empty:
            continue

        df_4h = resample_1h_to_4h(df_1h)
        if df_4h.empty:
            continue

        result[sym] = df_4h

    return result


# ============================================================
# 메인 엔트리: 레짐 계산 + settings 저장
# ============================================================
def update_market_regime_coin(
    db: BotDatabase,
    top_symbols: List[str],
    region: str = "BI",
) -> Dict[str, Any]:
    """
    - BTCUSDT + top_symbols 4h 데이터 로드
    - 레짐 계산
    - settings 테이블에 저장
    - info dict 반환
    """
    # 1) BTCUSDT 4h 데이터
    try:
        _, _, _, btc_1h = load_ohlcv_multiscale_for_symbol(
            region=region,
            symbol="BTCUSDT",
            base_interval="5m",
        )
    except Exception as e:
        raise RuntimeError(f"BTCUSDT 4h 데이터 로딩 실패: {e}")

    if btc_1h is None or btc_1h.empty:
        raise RuntimeError("BTCUSDT 1h 데이터가 없습니다.")

    btc_4h = resample_1h_to_4h(btc_1h)
    if btc_4h is None or len(btc_4h) < 100:
        raise RuntimeError("BTCUSDT 4h 데이터가 부족합니다. (len < 200)")

    # 2) 유니버스 상위 코인 4h 로드
    uni_4h = load_universe_4h_data(db, top_symbols, region=region)

    # 3) 레짐 계산
    info = detect_market_regime_coin(btc_4h, uni_4h)
    avg_ret_1d = calc_universe_avg_return_1d(uni_4h)
    info["avg_return_1d"] = avg_ret_1d

    regime = info["regime"]
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 4) settings 테이블에 저장
    db.set_setting("market_regime_coin", regime)
    db.set_setting("market_regime_coin_updated_at", now)
    db.set_setting("market_regime_coin_avg_return_1d", f"{avg_ret_1d:.6f}")
    db.set_setting("market_regime_coin_breadth_ma50", f"{info['breadth']:.6f}")

    # 5) 디버그 로그
    db.log(
        "[MARKET_REGIME_COIN] "
        f"region={region} | symbols={len(uni_4h)} | regime={regime} "
        f"| score={info['score']} "
        f"| breadth_ma50={info['breadth']:.3f} "
        f"| avg_1d={avg_ret_1d*100:.2f}%"
    )

    return info


if __name__ == "__main__":
    # ✅ Config에서 실제 백필된 유니버스 가져오기
    from c_config import BI_UNIVERSE_STOCKS
    
    db = BotDatabase()
    
    # 1. Config에 있는 딕셔너리 리스트에서 'symbol'만 추출
    # 예: [{'symbol': 'BTCUSDT', ...}, ...] -> ['BTCUSDT', ...]
    top_symbols = [
        t["symbol"] 
        for t in BI_UNIVERSE_STOCKS 
        if t.get("region") == "BI"
    ]

    print(f"🧪 테스트 시작")
    print(f"   - 대상: Config 유니버스 내 {len(top_symbols)}개 종목")
    # print(f"   - 목록: {top_symbols[:5]} ...") 

    try:
        # 2. 레짐 업데이트 실행
        info = update_market_regime_coin(db, top_symbols, region="BI")
        
        print("\n📊 결과 확인 (Settings 저장 완료):")
        print(f"   - 시장 레짐: {info['regime']}")
        print(f"   - 레짐 점수: {info['score']} (0=Bull, 1=Neutral, 2+=Bear)")
        print(f"   - 하락 종목 비율(Breadth): {info['breadth']*100:.2f}%")
        print(f"   - 평균 1일 수익률: {info.get('avg_return_1d', 0)*100:.4f}%")
        print(f"   - 상세 조건: {info['cond_detail']}")

    except Exception as e:
        print(f"\n❌ 에러 발생: {e}")
        print("   -> 힌트: DB에 데이터가 충분하지 않거나(백필 필요), BTCUSDT 데이터가 없을 수 있습니다.")