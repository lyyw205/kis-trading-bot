# backtest_multiscale_cr.py
# Multi-Scale TCN + Transformer 엔트리 백테스트 엔진
"""
CR(코인) 전용 Multi-Scale TCN + Transformer 엔트리 백테스트 엔진

- 데이터 소스: trading.db 의 ohlcv_data (region='CR', interval='5m')
- 엔트리 로직: st_entry_coin_ms.make_entry_signal_coin_ms
- 포지션: 심볼별 최대 1개 롱
- 청산:
    - TP / SL 퍼센트
    - 최대 보유 봉 수 (timeout)

결과:
    - 심볼별 성능 요약
    - 전체 성능 요약
    - 개별 트레이드 CSV 저장 (옵션)
"""

import os
import sqlite3
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import numpy as np
import pandas as pd

from config import CR_UNIVERSE_STOCKS
from tcn_entry import make_entry_signal_coin_ms
from ai_helpers import make_model_update_advice
from tcn_exit import CrPosition, decide_exit_cr_ms, DEFAULT_EXIT_PARAMS


DB_PATH = "trading.db"

# -----------------------------------------
# 백테스트 파라미터
# -----------------------------------------
TP_RATE = 0.02          # +2% 이익 시 청산
SL_RATE = -0.02         # -2% 손실 시 청산
MAX_HOLD_BARS = 24      # 최대 보유 24개 5분봉 (약 2시간)

EXIT_PARAMS_MS = DEFAULT_EXIT_PARAMS.copy()
EXIT_PARAMS_MS.update({
    "tp_rate": TP_RATE,              # 여기서 백테스트용 TP/SL/보유봉 설정
    "sl_rate": SL_RATE,
    "max_hold_bars": MAX_HOLD_BARS,
    # 필요하면 ml_recheck_min_bars, interval 등도 여기서 덮어쓰기 가능
    # "ml_recheck_min_bars": 3,
    # "ml_recheck_interval": 2,
})

# 엔트리 함수에 넘길 기본 params (필요에 따라 튜닝)
ENTRY_PARAMS_BASE: Dict[str, Any] = {
    "min_len": 120,
    "atr_period": 14,

    # 🔻 변동성 필터 완화
    "atr_max_ratio": 0.06,   # 3% → 6%
    "hl_max_ratio": 0.08,    # 5% → 8%

    "use_rsi_filter": False,  # 일단 꺼두고 모델만 보자
    "use_trend_filter": False,

    "rsi_min": 35.0,
    "rsi_max": 75.0,

    # 🔻 ML 컷 완전 느슨하게
    "ml_min_r3": 0.0,
    "ml_min_r6": 0.0,
    "ml_min_r12": 0.0,
    "ml_min_score": 0.0,      # score > 0만 허용하는 단계로 보고싶으면 0.0
    "ml_max_worst": -0.03,    # -1% → -3% (최악 허용 넓게)
    "ml_min_pos_ratio": 0.34, # 3개 중 1개만 양수여도 통과

    "ml_horizon_weights": [0.4, 0.35, 0.25],

    # STRONG / NORMAL 기준도 내려줘
    "ml_strong_score": 0.0020,  # 0.3% → 0.2%
    "ml_weak_score": 0.0005,    # 0.15% → 0.05%

    "atr_for_strong": 0.03,

    # 공통 엔트리용
    "lookback": 20,
    "band_pct": 0.005,
}


# -----------------------------------------
# 데이터 로더
# -----------------------------------------
def load_cr_ohlcv_5m(symbol: str) -> pd.DataFrame:
    """
    ohlcv_data에서 CR 5분봉 전체 로드
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT dt, open, high, low, close, volume
        FROM ohlcv_data
        WHERE region='CR' AND symbol=? AND interval='5m'
        ORDER BY dt
        """,
        conn,
        params=(symbol,),
        parse_dates=["dt"],
    )
    conn.close()

    if df.empty:
        return df

    df = df.set_index("dt").sort_index()
    df = df[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    return df


# -----------------------------------------
# 트레이드 기록용 데이터클래스
# -----------------------------------------
@dataclass
class TradeRecord:
    region: str
    symbol: str
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    pnl_pct: float
    holding_bars: int
    reason: str           # "TP" / "SL" / "TIMEOUT"
    strategy_name: str    # CR_MS_STRONG / NORMAL / LIGHT ...
    note: str             # st_entry_coin_ms 의 note
    ml_r3: Optional[float]
    ml_r6: Optional[float]
    ml_r12: Optional[float]
    ml_score: Optional[float]
    ml_worst: Optional[float]
    ml_pos_ratio: Optional[float]
    atr_ratio: Optional[float]


# -----------------------------------------
# 한 심볼에 대한 백테스트
# -----------------------------------------
def backtest_symbol(symbol_info: Dict[str, Any]) -> List[TradeRecord]:
    """
    symbol_info: {"region": "CR", "symbol": "KRW-BTC", "excd": "BITHUMB"}
    """
    region = symbol_info["region"]
    symbol = symbol_info["symbol"]

    print(f"\n[BACKTEST] {region} {symbol} 시작")

    df = load_cr_ohlcv_5m(symbol)
    if df.empty:
        print(f"  ⚠️ 데이터 없음 → 스킵")
        return []

    params = ENTRY_PARAMS_BASE.copy()

    trades: List[TradeRecord] = []
    open_pos = None  # {"entry_idx": int, "entry_price": float, "info": dict}

    # 인덱스 리스트 (datetime)
    times = df.index.to_list()
    closes = df["close"].values

    # 최소 길이 이후부터 루프
    min_len = params.get("min_len", 120)

    for i in range(min_len, len(df) - 1):  # 마지막 한 봉은 진입만, 청산은 그 뒤에서
        t = times[i]

        # 현재까지의 데이터 슬라이스
        df_slice = df.iloc[: i + 1]

        # ---------------------------------
        # 1) 오픈 포지션이 있으면 공통 청산 로직(decide_exit_cr_ms)으로 체크
        # ---------------------------------
        if open_pos is not None:
            entry_idx = open_pos["entry_idx"]
            entry_price = open_pos["entry_price"]
            pos_obj = open_pos["position"]

            # 지금 시점까지의 5분봉 슬라이스
            df_slice = df.iloc[: i + 1]
            cur_price = float(closes[i])     # 실시간에선 last/ticker, 여기선 현재 봉 close 사용
            now_dt = times[i]

            decision = decide_exit_cr_ms(
                pos=pos_obj,
                df_5m=df_slice,
                cur_price=cur_price,
                now_dt=now_dt.to_pydatetime() if isinstance(now_dt, pd.Timestamp) else now_dt,
                params=EXIT_PARAMS_MS,
            )

            if decision.get("should_exit", False):
                exit_price = float(decision.get("exit_price", cur_price))
                pnl_pct = (exit_price - entry_price) / entry_price
                holding = i - entry_idx

                # exit 모듈의 note도 같이 남겨두고 싶으면 합쳐서 저장
                exit_note = decision.get("note", "")
                note_combined = f"{open_pos['note']} | {exit_note}" if exit_note else open_pos["note"]

                tr = TradeRecord(
                    region=region,
                    symbol=symbol,
                    entry_time=times[entry_idx],
                    entry_price=entry_price,
                    exit_time=now_dt,
                    exit_price=exit_price,
                    pnl_pct=float(pnl_pct),
                    holding_bars=int(holding),
                    reason=decision.get("reason", "UNKNOWN"),
                    strategy_name=open_pos["strategy_name"],
                    note=note_combined,
                    ml_r3=open_pos["ml_r3"],
                    ml_r6=open_pos["ml_r6"],
                    ml_r12=open_pos["ml_r12"],
                    ml_score=open_pos["ml_score"],
                    ml_worst=open_pos["ml_worst"],
                    ml_pos_ratio=open_pos["ml_pos_ratio"],
                    atr_ratio=open_pos["atr_ratio"],
                )
                trades.append(tr)
                open_pos = None

        # 포지션이 여전히 열려 있다면, 새 진입은 안 한다
        if open_pos is not None:
            continue

        # ---------------------------------
        # 2) 새 엔트리 시그널 탐색
        #    (진입가는 "다음 봉"의 open 으로 가정 → 룩어헤드 방지)
        # ---------------------------------
        # 이 시점에서의 시그널을 보고, 다음 봉 open에 진입
        sig = make_entry_signal_coin_ms(df_slice, params)

        if not sig.get("entry_signal", False):
            continue

        # 다음 봉 index (진입 시점) 체크
        entry_idx = i + 1
        if entry_idx >= len(df):
            break  # 더 이상 진입 불가

        entry_time = times[entry_idx]
        entry_price = float(df["open"].iloc[entry_idx])

        ml_pred = sig.get("ml_pred") or {}
        risk = sig.get("risk") or {}

        # 🔹 실시간용과 동일한 구조의 포지션 객체 생성
        pos_obj = CrPosition(
            region=region,
            symbol=symbol,
            side="BUY",   # 코인은 롱만 쓰는 전제
            qty=1.0,      # 백테스트에선 수량은 의미 없으니 1.0으로 고정
            entry_price=entry_price,
            entry_time=entry_time.to_pydatetime() if isinstance(entry_time, pd.Timestamp) else entry_time,
            ml_score_entry=ml_pred.get("score"),
            ml_worst_entry=ml_pred.get("worst"),
            atr_ratio_entry=risk.get("atr_ratio"),
        )

        open_pos = {
            "entry_idx": entry_idx,
            "entry_price": entry_price,
            "position": pos_obj,  # 🔹 decide_exit_cr_ms 에 넘길 포지션 객체
            "strategy_name": sig.get("strategy_name", "CR_MS"),
            "note": sig.get("note", ""),
            "ml_r3": ml_pred.get("r_3"),
            "ml_r6": ml_pred.get("r_6"),
            "ml_r12": ml_pred.get("r_12"),
            "ml_score": ml_pred.get("score"),
            "ml_worst": ml_pred.get("worst"),
            "ml_pos_ratio": ml_pred.get("pos_ratio"),
            "atr_ratio": risk.get("atr_ratio"),
        }

    # 루프 끝까지 갔는데 포지션이 남아있으면 마지막 종가에 강제 청산
    if open_pos is not None:
        entry_idx = open_pos["entry_idx"]
        entry_price = open_pos["entry_price"]
        exit_idx = len(df) - 1
        exit_price = float(closes[exit_idx])
        pnl_pct = (exit_price - entry_price) / entry_price
        holding = exit_idx - entry_idx

        tr = TradeRecord(
            region=region,
            symbol=symbol,
            entry_time=times[entry_idx],
            entry_price=entry_price,
            exit_time=times[exit_idx],
            exit_price=exit_price,
            pnl_pct=float(pnl_pct),
            holding_bars=int(holding),
            reason="FORCE_CLOSE",
            strategy_name=open_pos["strategy_name"],
            note=open_pos["note"],
            ml_r3=open_pos["ml_r3"],
            ml_r6=open_pos["ml_r6"],
            ml_r12=open_pos["ml_r12"],
            ml_score=open_pos["ml_score"],
            ml_worst=open_pos["ml_worst"],
            ml_pos_ratio=open_pos["ml_pos_ratio"],
            atr_ratio=open_pos["atr_ratio"],
        )
        trades.append(tr)

    print(f"  → 트레이드 수: {len(trades)}")
    return trades


# -----------------------------------------
# 요약 / 통계 출력
# -----------------------------------------
def summarize_trades(trades: List[TradeRecord], title: str = ""):
    if not trades:
        print(f"\n[{title}] 트레이드 없음")
        return

    df = pd.DataFrame([asdict(t) for t in trades])

    n = len(df)
    wins = df[df["pnl_pct"] > 0]
    n_win = len(wins)
    win_rate = n_win / n if n > 0 else 0.0

    avg_ret = df["pnl_pct"].mean()
    med_ret = df["pnl_pct"].median()

    # 누적 수익률 (단순 합산 말고, 곱연산 기준)
    cum_ret = (df["pnl_pct"] + 1.0).prod() - 1.0

    print(f"\n==============================")
    print(f"[{title}] 성능 요약")
    print(f"==============================")
    print(f"총 트레이드 수   : {n}")
    print(f"승률             : {win_rate*100:.2f}% ({n_win}/{n})")
    print(f"평균 수익률(%)   : {avg_ret*100:.3f}%")
    print(f"중앙값 수익률(%) : {med_ret*100:.3f}%")
    print(f"누적 수익률(%)   : {cum_ret*100:.3f}%")

    # 전략 이름별
    print(f"\n전략별 성능 (strategy_name)")
    for name, g in df.groupby("strategy_name"):
        nn = len(g)
        ww = (g["pnl_pct"] > 0).sum()
        wr = ww / nn if nn > 0 else 0.0
        avg_r = g["pnl_pct"].mean()
        print(f"[{name:15}] 트레이드 {nn:4d} | 승률 {wr*100:6.2f}% | 평균 {avg_r*100:7.3f}%")

    # ML score 구간별 (선택)
    if "ml_score" in df.columns:
        bins = [-1.0, 0.0, 0.002, 0.004, 0.01]
        labels = ["<=0", "0~0.2%", "0.2~0.4%", ">=0.4%"]
        df["score_bucket"] = pd.cut(df["ml_score"].fillna(-1.0), bins=bins, labels=labels, include_lowest=True)
        print(f"\nML Score 구간별 성능")
        for b, g in df.groupby("score_bucket", observed=False):
            nn = len(g)
            if nn == 0:
                continue
            ww = (g["pnl_pct"] > 0).sum()
            wr = ww / nn
            avg_r = g["pnl_pct"].mean()
            print(f"[{b}] 트레이드 {nn:4d} | 승률 {wr*100:6.2f}% | 평균 {avg_r*100:7.3f}%")


# -----------------------------------------
# 메인
# -----------------------------------------
def main():
    # 1) 유니버스 일부만 골라서 테스트하고 싶으면 여기서 조정
    universe = [t for t in CR_UNIVERSE_STOCKS if t["symbol"] in ("KRW-BTC", "KRW-ETH", "KRW-XRP")]
    if not universe:
        universe = CR_UNIVERSE_STOCKS

    all_trades: List[TradeRecord] = []

    for t in universe:
        sym_trades = backtest_symbol(t)
        if sym_trades:
            # 심볼별 CSV도 저장하고 싶으면 이 부분 활성화
            df_sym = pd.DataFrame([asdict(tr) for tr in sym_trades])
            save_dir = "backtests_cr"
            os.makedirs(save_dir, exist_ok=True)
            csv_path = os.path.join(save_dir, f"bt_cr_{t['symbol']}.csv")
            df_sym.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print(f"   {t['symbol']} 결과 CSV 저장: {csv_path}")

        all_trades.extend(sym_trades)

    # 2) 전체 요약
    summarize_trades(all_trades, title="CR MultiScale TCN+Transformer")

if __name__ == "__main__":
    main()
