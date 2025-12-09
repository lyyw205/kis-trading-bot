#멀티스케일 ohlcv 로더
#DB에서 5분봉을 가져와 멀티스케일(5m/15m/30m/1h)로 변환하는 로더
# bi_multiscale_loader.py
from c_db_manager import BotDatabase
import pandas as pd
import time
import psycopg2
from pandas.errors import DatabaseError
import warnings
from bi_features import resample_from_5m 

warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable .*",
    category=UserWarning,
)


def load_ohlcv_multiscale_for_symbol(
    region: str,
    symbol: str,
    base_interval: str = "5m",
):
    """
    ohlcv_data 테이블에서 5분봉을 불러와서
    5m / 15m / 30m / 1h 시계열을 한 번에 만들어준다.

    반환:
        df_5m, df_15m, df_30m, df_1h
        (각각 datetime index, columns=[open,high,low,close,volume])
    """

    # -----------------------------
    # 0) DB 조회 + 재시도 로직
    # -----------------------------
    query = """
        SELECT dt, open, high, low, close, volume
        FROM ohlcv_data
        WHERE region = %s AND symbol = %s AND interval = %s
        ORDER BY dt
    """

    max_retry_seconds = 180   # 약 3분
    sleep_seconds = 5         # 실패 시 재시도 간격
    start_ts = time.time()
    attempt = 0
    df = None

    while True:
        attempt += 1
        db = BotDatabase()
        conn = db.get_connection()

        try:
            df = pd.read_sql_query(
                query,
                conn,
                params=(region, symbol, base_interval),
            )
            # ✅ 여기까지 오면 성공, 루프 탈출
            break

        except (DatabaseError, psycopg2.OperationalError, psycopg2.InterfaceError) as e:
            msg = str(e)

            retriable = (
                "SSL connection has been closed" in msg
                or "connection already closed" in msg
                or "server closed the connection" in msg
            )

            if not retriable:
                raise

            elapsed = time.time() - start_ts
            if elapsed >= max_retry_seconds:
                raise RuntimeError(
                    f"[load_ohlcv_multiscale_for_symbol] "
                    f"DB 연결 오류로 약 {int(elapsed)}초 동안 {attempt}회 시도했지만 실패했습니다. "
                    f"(region={region}, symbol={symbol}, interval={base_interval})"
                ) from e

            print(
                f"[load_ohlcv_multiscale_for_symbol] DB 연결 끊김 감지 → "
                f"{sleep_seconds}초 후 재시도 예정 "
                f"(attempt={attempt}, elapsed={int(elapsed)}s, symbol={symbol})"
            )
            time.sleep(sleep_seconds)

        finally:
            try:
                conn.close()
            except Exception:
                pass

    # -----------------------------
    # 1) 이후 로직
    # -----------------------------
    if df is None or df.empty:
        raise ValueError(
            f"ohlcv_data 비어 있음: region={region}, symbol={symbol}, interval={base_interval}"
        )

    df["dt"] = pd.to_datetime(
        df["dt"],
        format="%Y-%m-%d %H:%M:%S",
        errors="coerce"
    )

    df.set_index("dt", inplace=True)
    df.sort_index(inplace=True)

    # 숫자 컬럼 정리 + 결측 제거
    df_5m = df[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    if df_5m.empty:
        raise ValueError(
            f"유효한 OHLCV가 없음: region={region}, symbol={symbol}, interval={base_interval}"
        )

    # ✅ 여기서부터는 bi_features.resample_from_5m()에 위임
    df_5m, df_15m, df_30m, df_1h = resample_from_5m(df_5m)

    return df_5m, df_15m, df_30m, df_1h