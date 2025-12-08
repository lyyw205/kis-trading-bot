""" Binance 코인 멀티전략 + 우주 1포지션 백테스트 엔진

 - BI_UNIVERSE_STOCKS(바이낸스 코인 유니버스)를 대상으로
   멀티 엔트리 전략(MS / MS_SHORT)을 한 번에 평가하는 전용 백테스트 스크립트.
 - 한 시점에 우주 전체에서 단 하나의 포지션만 보유(우주 1포지션)하는 구조.

주요 기능:
1) load_cr_ohlcv_5m(symbol)
   - ohlcv_data 테이블에서 region='BI', interval='5m'인 바이낸스 코인 데이터를
     PostgreSQL을 통해 로드하고, 인덱스를 dt로 설정해 정제된 DataFrame 반환

2) TradeRecord dataclass
   - 백테스트 결과를 한 트레이드(포지션 단위)마다 구조화해서 보관하는 자료구조
   - 필드: symbol, side(BUY/SHORT), entry/exit 시각/가격, pnl_pct, 전략명, ML 스코어 등

3) backtest_universe(universe)
   - 주어진 universe(BI_UNIVERSE_STOCKS)를 순회해 각 심볼의 5분봉을 로딩
   - 전체 공통 타임라인(all_times)을 만들고, 시점별로:
       · 보유 중 포지션은 tcn_exit_hub.decide_exit_cr 로 청산 여부 판단
       · 포지션이 없을 때는 tcn_entry_hub.pick_best_entry_across_universe로
         멀티전략(MS, MS_SHORT) 후보 중 최종 엔트리 하나만 선택
       · 선택된 심볼/전략은 다음 봉 시가에 롱(BUY) 또는 숏(SHORT) 진입
   - 루프 종료 후 포지션이 남아 있으면 마지막 종가 기준 강제 청산
   - 최종적으로 TradeRecord 리스트를 반환

4) summarize_trades(trades, title)
   - 백테스트 결과(TradeRecord 리스트)를 집계하여
     전체 승률, 평균 수익률, 누적 수익률, MDD, 전략별 성능 등을 콘솔에 출력
   - ml_score 분포에 대한 기본 통계도 함께 출력

5) main()
   - BI_UNIVERSE_STOCKS를 universe로 사용해 backtest_universe 실행
   - 소요 시간과 성능 요약, 상위 수익 트레이드 몇 건을 콘솔에 표시

※ 전제:
 - 엔트리 쪽: tcn_entry_ms / tcn_entry_ms_short (+ 필요 시 REV, MOMO)
 - 멀티전략 허브: tcn_entry_hub.pick_best_entry_across_universe
 - 청산 로직: tcn_exit_hub.decide_exit_cr / CrPosition
 - 시세 데이터: ohlcv_data(region='BI', interval='5m')에 사전 백필되어 있어야 함.
"""

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.append(root_dir)

from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

import numpy as np
import pandas as pd

from c_config import BI_UNIVERSE_STOCKS

# ✅ 공통 코어: 기본 파라미터
from bi_entry_lib import DEFAULT_ENTRY_PARAMS_MS

# ✅ 전략 모듈
from bi_entry_ms import make_entry_signal_coin_ms
from bi_entry_rev import make_entry_signal_coin_rev
from bi_entry_momo import make_entry_signal_coin_momo
from bi_entry_short import make_entry_signal_coin_ms_short

# ✅ 멀티전략 허브
from bi_entry_hub import pick_best_entry_across_universe

from bi_exit_hub  import decide_exit_cr
from bi_exit_lib import CrPosition

from c_db_manager import BotDatabase


import warnings

warnings.filterwarnings(
    "ignore",
    message="pandas only supports SQLAlchemy connectable .*",
    category=UserWarning,
)


# 멀티전략 + 우주 1포지션일 때 최종 엔트리 최소 점수
MIN_FINAL_SCORE = 0.015


# -----------------------------------------
# 데이터 로더
# -----------------------------------------
def load_cr_ohlcv_5m(symbol: str) -> pd.DataFrame:
    """
    ohlcv_data에서 BI 5분봉 전체 로드 (PostgreSQL 연동)
    """
    db = BotDatabase()
    conn = db.get_connection()

    try:
        df = pd.read_sql_query(
            """
            SELECT dt, open, high, low, close, volume
            FROM ohlcv_data
            WHERE region='BI' AND symbol=%s AND interval='5m'
            ORDER BY dt
            """,
            conn,
            params=(symbol,),
            parse_dates=["dt"],
        )

        if df.empty:
            return df

        df["dt"] = pd.to_datetime(df["dt"])
        df = df.set_index("dt").sort_index()
        df = df[["open", "high", "low", "close", "volume"]].apply(
            pd.to_numeric, errors="coerce"
        ).dropna()

        return df

    finally:
        conn.close()


def get_current_market_regime_coin():
    """
    settings 테이블에 저장된 코인 레짐/평균 수익률을 읽어온다.
    run_update_market_regime_coin.py 를 미리 돌려서
    market_regime_coin_* 값이 세팅되어 있어야 함.
    """
    db = BotDatabase()

    regime = db.get_setting("market_regime_coin", default="NEUTRAL")
    avg_ret_str = db.get_setting("market_regime_coin_avg_return_1d", default="0.0")

    try:
        avg_ret = float(avg_ret_str)
    except Exception:
        avg_ret = 0.0

    return regime, avg_ret

# -----------------------------------------
# 트레이드 기록용 데이터클래스
# -----------------------------------------
@dataclass
class TradeRecord:
    region: str
    symbol: str
    side: str             # BUY / SHORT
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: pd.Timestamp
    exit_price: float
    pnl_pct: float
    holding_bars: int
    reason: str           # "TP" / "SL" / "TIMEOUT" / ...
    strategy_name: str    # CR_MS_STRONG / CR_REV_ENTRY / CR_MOMO_ENTRY ...
    note: str             # exit note or entry note
    ml_r3: Optional[float]
    ml_r6: Optional[float]
    ml_r12: Optional[float]
    ml_score: Optional[float]       # 멀티전략 공통 entry_score
    ml_worst: Optional[float]
    ml_pos_ratio: Optional[float]
    atr_ratio: Optional[float]


# -----------------------------------------
# 우주(여러 코인) 1포지션 백테스트
# -----------------------------------------
def backtest_universe(universe: List[Dict[str, Any]]) -> List[TradeRecord]:
    """
    universe: [{"region": "BI", "symbol": "KRW-BTC"}, ...]
    """
    print("\n[BACKTEST] BI 멀티전략 + 우주 1포지션 시작")

    # ✅ 0) 현재 settings 기준 코인 레짐 읽기
    regime, avg_ret_1d = get_current_market_regime_coin()
    print(f"[BACKTEST] settings 기준 market_regime_coin = {regime}, "
          f"avg_return_1d = {avg_ret_1d*100:.2f}%")
    
    # 1) 각 심볼별로 5분봉 로드
    dfs: Dict[str, pd.DataFrame] = {}
    for info in universe:
        symbol = info["symbol"]
        region = info["region"]
        if region != "BI":
            continue

        df_raw = load_cr_ohlcv_5m(symbol)
        if df_raw.empty:
            print(f"  - {symbol}: 데이터 없음, 스킵")
            continue

        dfs[symbol] = df_raw

    if not dfs:
        print("⚠️ 유효한 BI 데이터 없음, 종료")
        return []

    # 2) 전체 타임라인 생성
    all_times = sorted(set().union(*[df.index for df in dfs.values()]))
    if not all_times:
        print("⚠️ 전체 타임라인 비어 있음, 종료")
        return []

    all_times = pd.to_datetime(all_times)

    # ✅ 전략 셋업 (설정하고 싶으면 하단 멀티전략 허브에서 strategies 인자로 넘기기)
    entry_strategies = {
        # "MS": make_entry_signal_coin_ms,
        # "REV": make_entry_signal_coin_rev,
        # "MOMO": make_entry_signal_coin_momo,
        # "MS_SHORT": make_entry_signal_coin_ms_short,
    }

    min_len = DEFAULT_ENTRY_PARAMS_MS["min_len"]

    trades: List[TradeRecord] = []
    open_pos: Optional[Dict[str, Any]] = None  # 현재 보유 포지션 (우주 1개)

    # -----------------------------------------
    # 메인 루프
    # -----------------------------------------
    for t in all_times:
        # 1) 포지션 보유 중이면 청산 여부 먼저 확인
        if open_pos is not None:
            sym = open_pos["symbol"]
            df_sym = dfs.get(sym)
            
            # 데이터 정합성 체크
            if df_sym is None or t not in df_sym.index:
                pass
            else:
                # exit 로직에 넘길 슬라이스
                df_slice_small = df_sym.loc[:t].iloc[-50:]
                cur_price = df_slice_small["close"].iloc[-1]

                pos_obj: CrPosition = open_pos["position"]
                side = pos_obj.side # BUY or SHORT

                decision = decide_exit_cr(
                    pos=pos_obj,
                    df_5m=df_slice_small,
                    cur_price=cur_price,
                    now_dt=t,
                    strategy_name=open_pos["strategy_name"],
                    params_by_strategy={},
                )

                if decision.get("should_exit", False):
                    exit_price = float(decision.get("exit_price", cur_price))
                    entry_time = open_pos["entry_time"]
                    entry_price = open_pos["entry_price"]
                    
                    # 🔽 [수정] Side에 따른 PnL 계산
                    if side == "SHORT":
                        pnl_pct = (entry_price - exit_price) / entry_price
                    else:
                        pnl_pct = (exit_price - entry_price) / entry_price

                    holding_minutes = (t - entry_time).total_seconds() / 60.0
                    holding_bars = int(round(holding_minutes / 5.0))

                    tr = TradeRecord(
                        region="BI",
                        symbol=sym,
                        side=side,
                        entry_time=entry_time,
                        entry_price=entry_price,
                        exit_time=t,
                        exit_price=exit_price,
                        pnl_pct=float(pnl_pct),
                        holding_bars=holding_bars,
                        reason=decision.get("reason", "UNKNOWN"),
                        strategy_name=open_pos["strategy_name"],
                        note=decision.get("note", ""),
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

        # 2) 포지션이 아직 있으면 새 엔트리는 못 들어감
        if open_pos is not None:
            continue

        # 3) 포지션이 없으면: 이 시점에서 엔트리 후보 찾기
        df_by_symbol: Dict[str, pd.DataFrame] = {}

        for sym, df_sym in dfs.items():
            if t not in df_sym.index:
                continue

            df_slice = df_sym.loc[:t]
            if len(df_slice) < min_len:
                continue

            df_by_symbol[sym] = df_slice

        if not df_by_symbol:
            continue

        # ✅ 멀티전략 허브 (백테스트에선 BEAR/BULL 구분 없이 다 돌려봄)
        # 만약 특정 레짐을 가정하고 싶다면 market_regime="BEAR" 등을 추가 가능
        result = pick_best_entry_across_universe(
            df_by_symbol=df_by_symbol,
            strategies=None,
            params_by_strategy={},
            min_final_score=MIN_FINAL_SCORE,
            market_regime=regime,
            per_strategy_min_score={
                "MS": 0.012,        # 롱은 점수 높은 것만
                "MS_SHORT": 0.01,  # 숏은 조금 더 느슨하게
            },
        )

        if not result.get("has_final_entry", False):
            continue

        # 4) 최종 선정된 심볼 / 전략에 대해 → 다음 봉 시가에 진입
        best_sym = result["symbol"]
        entry_ctx = result["entry"] or {}
        df_sym = dfs[best_sym]

        try:
            idx = df_sym.index.get_loc(t)
        except KeyError:
            continue

        entry_idx = idx + 1
        if entry_idx >= len(df_sym):
            continue

        entry_time = df_sym.index[entry_idx]
        entry_price = float(df_sym["open"].iloc[entry_idx])
        
        # 🔽 [수정] 진입 방향(Side) 파싱
        entry_side = entry_ctx.get("side", "BUY")
        if entry_side == "SELL": # 혹시라도 SELL로 넘어오면 SHORT로 통일
            entry_side = "SHORT"

        ml_pred = entry_ctx.get("ml_pred") or {}
        risk = entry_ctx.get("risk") or {}

        pos_obj = CrPosition(
            region="BI",
            symbol=best_sym,
            side=entry_side, # BUY or SHORT
            qty=1.0,
            entry_price=entry_price,
            entry_time=entry_time,
            ml_score_entry=ml_pred.get("score"),
            ml_worst_entry=ml_pred.get("worst"),
            atr_ratio_entry=risk.get("atr_ratio"),
        )

        open_pos = {
            "symbol": best_sym,
            "entry_time": entry_time,
            "entry_price": entry_price,
            "position": pos_obj,
            "strategy_name": entry_ctx.get("strategy_name", result.get("strategy", "UNKNOWN")),
            "note": entry_ctx.get("note", ""),
            "ml_r3": ml_pred.get("r_3"),
            "ml_r6": ml_pred.get("r_6"),
            "ml_r12": ml_pred.get("r_12"),
            "ml_score": entry_ctx.get("final_score") or entry_ctx.get("entry_score"),
            "ml_worst": ml_pred.get("worst"),
            "ml_pos_ratio": ml_pred.get("pos_ratio"),
            "atr_ratio": risk.get("atr_ratio"),
        }

    # 5) 루프 끝났는데 포지션 남아 있으면 강제 청산
    if open_pos is not None:
        sym = open_pos["symbol"]
        df_sym = dfs[sym]
        exit_time = df_sym.index[-1]
        exit_price = float(df_sym["close"].iloc[-1])
        
        pos_obj = open_pos["position"]
        side = pos_obj.side

        if side == "SHORT":
            pnl_pct = (open_pos["entry_price"] - exit_price) / open_pos["entry_price"]
        else:
            pnl_pct = (exit_price - open_pos["entry_price"]) / open_pos["entry_price"]

        holding_minutes = (exit_time - open_pos["entry_time"]).total_seconds() / 60.0
        holding_bars = int(round(holding_minutes / 5.0))

        tr = TradeRecord(
            region="BI",
            symbol=sym,
            side=side,
            entry_time=open_pos["entry_time"],
            entry_price=open_pos["entry_price"],
            exit_time=exit_time,
            exit_price=exit_price,
            pnl_pct=float(pnl_pct),
            holding_bars=holding_bars,
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

    print(f"\n[BACKTEST DONE] 트레이드 수: {len(trades)}")
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
    cum_ret = (df["pnl_pct"] + 1.0).prod() - 1.0
    
    # MDD 계산 (누적 수익 곡선 기준)
    df = df.sort_values("exit_time")
    df["cum_idx"] = (df["pnl_pct"] + 1.0).cumprod()
    df["peak"] = df["cum_idx"].cummax()
    df["dd"] = (df["cum_idx"] - df["peak"]) / df["peak"]
    mdd = df["dd"].min()

    print(f"\n==============================")
    print(f"[{title}] 성능 요약")
    print(f"==============================")
    print(f"총 트레이드 수   : {n}")
    print(f"승률             : {win_rate*100:.2f}% ({n_win}/{n})")
    print(f"평균 수익률(%)   : {avg_ret*100:.3f}%")
    print(f"중앙값 수익률(%) : {med_ret*100:.3f}%")
    print(f"누적 수익률(%)   : {cum_ret*100:.3f}%")
    print(f"MDD(%)           : {mdd*100:.3f}%")

    print(f"\n전략별 성능 (strategy_name)")
    for name, g in df.groupby("strategy_name"):
        nn = len(g)
        ww = (g["pnl_pct"] > 0).sum()
        wr = ww / nn if nn > 0 else 0.0
        avg_r = g["pnl_pct"].mean()
        
        # 숏/롱 구분 표시
        side_hint = "SHORT" if "SHORT" in name else "LONG"
        print(f"[{name:20}] {side_hint} | 트레이드 {nn:4d} | 승률 {wr*100:6.2f}% | 평균 {avg_r*100:7.3f}%")

    if "ml_score" in df.columns:
        # 숏 점수는 음수일 수도 있고 양수일 수도 있어서 절대값으로 보거나 별도 처리 필요하지만
        # 여기선 단순 분포만 확인
        print(f"\nML Score 통계")
        print(df["ml_score"].describe())


# -----------------------------------------
# 메인
# -----------------------------------------
def main():
    universe = BI_UNIVERSE_STOCKS

    start_time = time.time()
    trades = backtest_universe(universe)
    end_time = time.time()

    print(f"\n[완료] 소요 시간: {end_time - start_time:.2f}초")
    summarize_trades(trades, title="BI 멀티전략(L/S) + 우주 1포지션 백테스트")

    if trades:
        df = pd.DataFrame([asdict(t) for t in trades])
        print("\n📌 상위 수익 트레이드 상세")
        print(
            df.sort_values("pnl_pct", ascending=False)[
                ["symbol", "side", "entry_time", "exit_time",
                 "pnl_pct", "reason", "strategy_name", "holding_bars"]
            ].head(5)
        )


if __name__ == "__main__":
    main()
    