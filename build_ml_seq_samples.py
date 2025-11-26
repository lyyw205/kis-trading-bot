# build_ml_seq_samples.py
import sqlite3
from datetime import datetime

import pandas as pd

from db import BotDatabase
from config import TARGET_STOCKS

DB_PATH = "trading.db"

# 시퀀스 길이 / 미래 구간 / TP·SL 기준
SEQ_LEN = 30          # 과거 60 캔들 사용
FUTURE_BARS_KR = 60   # 3분봉 기준 약 36분 (필요하면 조절 가능)
FUTURE_BARS_US = 5    # 일봉 기준 5일
TP_PCT = 0.03         # +3% 익절
SL_PCT = -0.04        # -4% 손절


def ensure_ml_seq_table(reset=False):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ml_seq_samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region TEXT,
            symbol TEXT,
            interval TEXT,
            dt_entry TEXT,          -- 진입 기준 시점 (시퀀스 마지막 캔들 시각)
            entry_price REAL,

            seq_len INTEGER,
            future_bars INTEGER,

            tp_hit INTEGER,          -- 1 or 0
            sl_hit INTEGER,          -- 1 or 0
            label INTEGER,           -- 1: 좋은 진입, 0: 나쁜 진입

            future_max_ret REAL,     -- (미래 최고가 / entry - 1)
            future_min_ret REAL      -- (미래 최저가 / entry - 1)
        )
    """)

    # 🔥 매번 전체를 다시 만들고 싶으면 reset=True로 호출 → 기존 샘플 삭제
    if reset:
        cur.execute("DELETE FROM ml_seq_samples")

    conn.commit()
    conn.close()


def load_ohlcv(region, symbol, interval):
    """ohlcv_data에서 해당 심볼+타임프레임의 전체 데이터를 불러옴."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT dt, open, high, low, close, volume
        FROM ohlcv_data
        WHERE region = ? AND symbol = ? AND interval = ?
        ORDER BY dt
        """,
        conn,
        params=(region, symbol, interval),
    )
    conn.close()

    if df.empty:
        return pd.DataFrame()

    df["dt"] = pd.to_datetime(df["dt"])
    df.set_index("dt", inplace=True)
    df = df.astype(float)
    return df


def build_samples_for_symbol(db, region, symbol, interval):
    """
    한 심볼에 대해:
      - ohlcv_data에서 시계열을 불러오고
      - 롤링 윈도우로 SEQ_LEN + FUTURE_BARS 만큼 묶어서
      - label을 계산해 ml_seq_samples에 INSERT
    """
    df = load_ohlcv(region, symbol, interval)
    if df.empty:
        db.log(f"⚠️ ML 시퀀스 샘플 생성 실패: 데이터 없음 {region} {symbol} ({interval})")
        return 0

    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    index = df.index

    if region == "KR":
        future_bars = FUTURE_BARS_KR
    else:
        future_bars = FUTURE_BARS_US

    if len(df) < SEQ_LEN + future_bars + 1:
        db.log(
            f"⚠️ 샘플 생성 불가 (데이터 부족): {region} {symbol} "
            f"({interval}) len={len(df)}, 필요>={SEQ_LEN + future_bars + 1}"
        )
        return 0

    rows = []

    # i: 시퀀스 마지막 캔들 인덱스 (진입 기준 시점)
    # i 뒤로 future_bars 만큼 미래 캔들이 있어야 하므로 range 끝 조정
    for i in range(SEQ_LEN - 1, len(df) - future_bars):
        entry_price = closes[i]
        dt_entry = index[i]

        future_slice = slice(i + 1, i + 1 + future_bars)
        future_high = highs[future_slice].max()
        future_low = lows[future_slice].min()

        future_max_ret = (future_high / entry_price) - 1.0
        future_min_ret = (future_low / entry_price) - 1.0

        tp_hit = future_max_ret >= TP_PCT
        sl_hit = future_min_ret <= SL_PCT

        # 둘 다 안 맞으면 애매한 샘플 → 스킵
        if not tp_hit and not sl_hit:
            continue

        # 둘 다 맞은 경우: 보수적으로 0 (위험한 구간)
        if tp_hit and sl_hit:
            label = 0
        else:
            label = 1 if tp_hit else 0

        rows.append((
            region,
            symbol,
            interval,
            dt_entry.strftime("%Y-%m-%d %H:%M:%S"),
            float(entry_price),
            SEQ_LEN,
            future_bars,
            int(tp_hit),
            int(sl_hit),
            int(label),
            float(future_max_ret),
            float(future_min_ret),
        ))

    if not rows:
        db.log(f"⚠️ 유효 샘플 없음: {region} {symbol} ({interval})")
        return 0

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.executemany("""
        INSERT INTO ml_seq_samples (
            region, symbol, interval, dt_entry, entry_price,
            seq_len, future_bars,
            tp_hit, sl_hit, label,
            future_max_ret, future_min_ret
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)
    conn.commit()
    conn.close()

    db.log(f"✅ ML 시퀀스 샘플 생성 완료: {region} {symbol} ({interval}) {len(rows)}개")
    return len(rows)


if __name__ == "__main__":
    db = BotDatabase(DB_PATH)
    db.log("📚 ML 시퀀스 샘플 생성 시작")

    # 🔥 매번 전체 리빌드: reset=True
    ensure_ml_seq_table(reset=True)

    total = 0
    for t in TARGET_STOCKS:
        region = t["region"]
        symbol = t["symbol"]
        excd = t.get("excd")  # 필요시 확장용

        # ohlcv_data에서 trader와 동일한 interval 사용
        interval = "5m" if region == "KR" else "1d"

        try:
            n = build_samples_for_symbol(db, region, symbol, interval)
            total += n
        except Exception as e:
            db.log(f"⚠️ {region} {symbol} 샘플 생성 중 에러: {e}")
            continue

    db.log(f"🎉 ML 시퀀스 샘플 생성 완료 (총 {total}개)")
