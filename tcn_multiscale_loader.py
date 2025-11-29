#멀티스케일 ohlcv 로더
#trading.db에서 5분봉을 가져와 멀티스케일(5m/15m/30m/1h)로 변환하는 로더

# ms_loader_cr.py
import sqlite3
import pandas as pd

DB_PATH = "trading.db"


def _resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """
    5분봉 → 15m/30m/1h 등으로 리샘플
    """
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    out = (
        df.resample(rule)
        .agg(agg)
        .dropna()
    )
    return out


def load_ohlcv_multiscale_for_symbol(
    region: str,
    symbol: str,
    base_interval: str = "5m",
    db_path: str = DB_PATH,
):
    """
    ohlcv_data 테이블에서 5분봉을 불러와서
    5m / 15m / 30m / 1h 시계열을 한 번에 만들어준다.

    반환:
        df_5m, df_15m, df_30m, df_1h
        (각각 datetime index, columns=[open,high,low,close,volume])
    """
    conn = sqlite3.connect(db_path)
    query = """
        SELECT dt, open, high, low, close, volume
        FROM ohlcv_data
        WHERE region = ? AND symbol = ? AND interval = ?
        ORDER BY dt
    """
    df = pd.read_sql_query(
        query,
        conn,
        params=(region, symbol, base_interval),
    )
    conn.close()

    if df.empty:
        raise ValueError(
            f"ohlcv_data 비어 있음: region={region}, symbol={symbol}, interval={base_interval}"
        )

    # dt를 datetime index로 변환
    df["dt"] = pd.to_datetime(df["dt"])
    df = df.set_index("dt").sort_index()

    # 숫자 컬럼 정리 + 결측 제거
    df = df[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    if df.empty:
        raise ValueError(
            f"유효한 OHLCV가 없음: region={region}, symbol={symbol}, interval={base_interval}"
        )

    # 5m는 그대로 사용
    df_5m = df

    # 5m → 15m/30m/1h 리샘플
    df_15m = _resample_ohlcv(df_5m, "15min")
    df_30m = _resample_ohlcv(df_5m, "30min")
    df_1h  = _resample_ohlcv(df_5m, "60min")

    return df_5m, df_15m, df_30m, df_1h
