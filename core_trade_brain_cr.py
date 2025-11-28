# trader_cr.py
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from ai_helpers import make_entry_comment, make_exit_comment
from ml_features import SEQ_LEN, build_feature_from_seq
from st_entry_coin import make_entry_signal_coin
from st_exit_common import decide_exit

from brk_bithumb_client import BithumbDataFetcher
from db_manager import BotDatabase
from config import CR_UNIVERSE_STOCKS  # [{"region": "CR", "symbol": "KRW-BTC", ...}, ...]

DB_PATH = "trading.db"


class CoinRealTimeTrader:
    """
    코인(CR) 전용 실시간 트레이더

    - 인터페이스를 GlobalRealTimeTrader 에 최대한 맞춘 버전
      __init__(fetcher, targets, params, db, model=None, ml_threshold=0.55)

    - 차이점
      * region: "CR" 만 사용
      * fetcher: BithumbDataFetcher
      * 잔고/현금: 코인 전용 메서드 사용
      * 주문: send_coin_order 사용
      * 시장 시간: 24시간 열려 있다고 가정
    """

    def __init__(
        self,
        fetcher: BithumbDataFetcher,
        targets,
        params,
        db: BotDatabase,
        model=None,
        ml_threshold: float = 0.55,
        dry_run=True,
    ):
        self.fetcher = fetcher
        self.targets = targets          # 보통 CR_UNIVERSE_STOCKS
        self.params = params            # lookback, band_pct 등
        self.db = db
        self.trade_state = {}           # 심볼별 TP/SL 상태 저장 (st_exit_common과 호환)

        # ML 모델 (RandomForest 등)
        self.model = model
        self.ml_threshold = ml_threshold

        # 코인용 최소 주문 기준 (KRW)
        self.min_order_amount_krw = 5_000

        # 포지션 최대 개수 (코인 전용)
        self.max_pos = 3

    # ------------------------------------------------
    # 코인은 24시간 장이라 장 마감 체크는 간단하게
    # ------------------------------------------------
    def is_market_open(self, region: str) -> bool:
        # CR 은 항상 True
        if region == "CR":
            return True
        return False

    # ------------------------------------------------
    # 매수 집행 로직 (주식 트레이더와 최대한 비슷하게)
    # ------------------------------------------------
    def execute_buys(self, candidates, coin_balance, cash_krw):
        """
        candidates: [
          {
            "region": "CR",
            "symbol": "KRW-BTC",
            "current_price": float,
            "ml_proba": float,
            "signal_id": int,
            "strategy_name": str,
          },
          ...
        ]
        """
        max_pos = self.max_pos

        held = len(coin_balance)
        total_held = held
        remain_slots = max(0, max_pos - total_held)

        self.db.log(
            f"🧮 [COIN 매수집행] 후보:{len(candidates)} | "
            f"보유:{held}/{max_pos} | 남은슬롯:{remain_slots} | "
            f"KRW:{cash_krw:,.0f}원"
        )

        if remain_slots <= 0:
            self.db.log("⏭️ [COIN 슬롯없음] 신규 매수 전부 스킵")
            return

        # 이미 보유 중인 마켓은 후보에서 제외
        held_markets = set(coin_balance.keys())
        filtered = [c for c in candidates if c["symbol"] not in held_markets]

        if not filtered:
            self.db.log("⏭️ [COIN 후보없음] 신규 매수 대상 없음")
            return

        # ML 점수 높은 순 정렬
        filtered.sort(key=lambda x: (x["ml_proba"] or 0), reverse=True)
        targets_to_buy = filtered[:remain_slots]

        available_krw = cash_krw
        slots_left = remain_slots
        success_new = 0

        for c in targets_to_buy:
            if slots_left <= 0:
                break

            region = c["region"]
            market = c["symbol"]          # "KRW-BTC"
            price = c["current_price"]
            ml_proba = c["ml_proba"]
            signal_id = c["signal_id"]
            strategy_name = c.get("strategy_name", "UNKNOWN")

            # 분할 비율은 기존 주식 로직과 비슷하게
            buy_index = success_new
            if buy_index == 0:
                min_ratio, max_ratio = 0.30, 0.40
            elif buy_index == 1:
                min_ratio, max_ratio = 0.40, 0.60
            else:
                min_ratio, max_ratio = 1.0, 1.0
            ratio = (min_ratio + max_ratio) / 2.0

            # 코인은 KRW 기준
            if available_krw <= 0:
                continue

            budget = available_krw * ratio
            if budget < self.min_order_amount_krw:
                self.db.log(
                    f"⚠️ [COIN금액컷] {market} Budget={budget:.0f}원 (< {self.min_order_amount_krw}원)"
                )
                continue

            volume = budget / price
            amount = volume * price

            if volume <= 0 or amount < self.min_order_amount_krw:
                self.db.log(
                    f"⚠️ [COIN수량컷] {market} volume={volume:.6f}, amount={amount:.0f}원"
                )
                continue

            # 실제 주문 보내기 (빗썸: 매수 side='bid')
            success = self.fetcher.send_coin_order(
                market=market,
                side="bid",
                volume=volume,
                price=None,        # 시장가 매수라면 price=None + ord_type="market"
                ord_type="market",
            )

            if success:
                available_krw -= amount
                slots_left -= 1
                total_held += 1
                success_new += 1

                # trade 기록 (profit=0, 향후 청산시 계산)
                trade_id = self.db.save_trade(
                    market,
                    "BUY",
                    price,
                    volume,
                    0,
                    signal_id=signal_id,
                    ml_proba=ml_proba,
                    entry_allowed=True,
                    region=region
                )

                self.db.log(
                    f"✅🚀[COIN매수] {market} {volume:.6f} | ML:{ml_proba:.3f} | 약 {amount:,.0f}원"
                )

                # AI 진입 코멘트
                try:
                    entry_ctx = {
                        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "region": region,
                        "symbol": market,
                        "exchange": "BITHUMB",
                        "side": "BUY",
                        "qty": volume,
                        "price": float(price),
                        "ml_proba": ml_proba,
                        "strategy": strategy_name,
                    }
                    comment = make_entry_comment(entry_ctx)
                    self.db.update_trade_entry_comment(trade_id, comment)
                except Exception as e:
                    self.db.log(f"⚠️ [COIN AI진입코멘트 실패] {market} | {e}")

    # ------------------------------------------------
    # 메인 체크 루프 (주식 trader.run_check 와 비슷한 구조)
    # ------------------------------------------------
    def run_check(self):
        # 1. 잔고 및 현금 조회 (코인 전용)
        try:
            coin_balance = self.fetcher.get_coin_balance()
            cash_krw = self.fetcher.get_coin_buyable_cash()
        except Exception as e:
            self.db.log(f"❌ [COIN] 잔고 불러오기 실패: {e}")
            return

        self.db.log(
            f"💰 [COIN 잔고스냅샷] 보유코인:{len(coin_balance)} | "
            f"KRW:{cash_krw:,.0f}원"
        )

        entry_candidates = []
        count_checked = 0
        count_skipped = 0
        count_signals = 0
        ml_scores = []

        skip_market_closed = 0
        skip_no_price = 0
        skip_no_df = 0
        skip_short_df = 0

        # 2. 코인 유니버스 스캔
        for t in self.targets:
            region = t["region"]
            market = t["symbol"]   # "KRW-BTC" 형식

            if region != "CR":
                # 혹시 섞여있어도 안전하게 방어
                count_skipped += 1
                continue

            time.sleep(0.2)  # API 과부하 방지

            # (1) 시장 열렸는지 체크 (CR은 거의 항상 True)
            if not self.is_market_open(region):
                skip_market_closed += 1
                count_skipped += 1
                continue

            # (2) 현재가 조회
            price = self.fetcher.get_coin_current_price(market)
            has_coin = (market in coin_balance) or (market in self.trade_state)
            my_info = coin_balance.get(market)

            if not price:
                skip_no_price += 1
                count_skipped += 1
                continue

            # (3) OHLCV 조회 (5분봉)
            interval = "5m"  # DB 저장용 interval 명
            df = self.fetcher.get_coin_ohlcv(
                market=market,
                interval="minute5",
                count=120,
            )

            if df is None or df.empty:
                skip_no_df += 1
                count_skipped += 1
                continue

            if len(df) < SEQ_LEN:
                skip_short_df += 1
                count_skipped += 1
                continue

            count_checked += 1

            # OHLCV DB 저장 (원한다면)
            try:
                self.db.save_ohlcv_df(region, market, interval, df)
            except Exception:
                pass

            # (4) 코인 엔트리 전략 로직
            sig = make_entry_signal_coin(df, self.params)
            entry_signal = sig["entry_signal"]
            strategy_name = sig["strategy_name"]
            at_support = sig.get("at_support", False)
            is_bullish = sig.get("is_bullish", False)
            price_up = sig.get("price_up", False)

            if entry_signal:
                count_signals += 1

            # (5) ML 점수 계산
            df_seq = df.iloc[-SEQ_LEN:]
            seq_feat = build_feature_from_seq(df_seq)

            ml_proba = None
            if self.model is not None and seq_feat is not None:
                try:
                    ml_proba = float(self.model.predict_proba([seq_feat])[0][1])
                    ml_scores.append((market, ml_proba))
                except Exception as e:
                    self.db.log(f"⚠️ [COIN ML예외] {market}: {e}")
                    ml_proba = None

            # (6) 최종 진입 허용 여부
            entry_allowed = entry_signal and (
                (ml_proba is not None) and (ml_proba >= self.ml_threshold)
            )

            # (7) 신호 DB 저장 (주식과 동일 포맷)
            signal_id = self.db.save_signal(
                region=region,
                symbol=market,
                price=float(df["close"].iloc[-1]),
                at_support=bool(at_support),
                is_bullish=bool(is_bullish),
                price_up=bool(price_up),
                lookback=self.params["lookback"],
                band_pct=self.params["band_pct"],
                has_stock=has_coin,          # 코인도 has_stock에 기록 (컬럼 재사용)
                entry_signal=entry_signal,
                ml_proba=ml_proba,
                entry_allowed=entry_allowed,
                note=strategy_name,
            )

            # (8) 매수 후보 등록 (미보유만)
            if entry_allowed and not has_coin:
                entry_candidates.append(
                    {
                        "region": region,
                        "symbol": market,
                        "current_price": price,
                        "ml_proba": ml_proba,
                        "signal_id": signal_id,
                        "strategy_name": strategy_name,
                    }
                )

            # (9) 매도/청산 로직
            if has_coin and my_info:
                avg_price = my_info["avg_price"]
                qty = my_info["qty"]

                state = self.trade_state.setdefault(
                    market,
                    {
                        "tp1": False,
                        "tp2": False,
                        "entry_time": datetime.utcnow(),
                        "max_profit": 0.0,
                    },
                )

                now = datetime.utcnow()

                sell_qty, sell_type, new_state, profit_rate, elapsed_min = decide_exit(
                    symbol=market,
                    region=region,
                    price=price,
                    avg_price=avg_price,
                    qty=qty,
                    state=state,
                    now=now,
                )

                # 상태 업데이트 / 삭제
                if new_state.get("delete"):
                    if market in self.trade_state:
                        del self.trade_state[market]
                else:
                    self.trade_state[market] = new_state

                # 실제 매도
                if sell_qty > 0:
                    success = self.fetcher.send_coin_order(
                        market=market,
                        side="ask",
                        volume=sell_qty,
                        price=None,
                        ord_type="market",
                    )

                    if success:
                        trade_id = self.db.save_trade(
                            market,
                            sell_type,
                            price,
                            sell_qty,
                            profit_rate * 100,
                            region=region
                        )
                        self.db.log(
                            f"📉[COIN매도] {market}: {sell_type} {sell_qty:.6f} "
                            f"({profit_rate*100:.2f}%)"
                        )

                        # AI 청산 코멘트
                        try:
                            exit_ctx = {
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "region": region,
                                "symbol": market,
                                "exchange": "BITHUMB",
                                "side": "SELL",
                                "qty": sell_qty,
                                "avg_entry": float(avg_price),
                                "exit_price": float(price),
                                "pnl_pct": profit_rate * 100,
                                "reason": sell_type,
                                "holding_minutes": elapsed_min,
                            }
                            comment = make_exit_comment(exit_ctx)
                            self.db.update_trade_exit_comment(trade_id, comment)
                        except Exception as e:
                            self.db.log(f"⚠️ [COIN AI청산코멘트 실패] {market} | {e}")

        # 3. 매수 집행 (전체 스캔 끝난 뒤 한 번만)
        self.execute_buys(entry_candidates, coin_balance, cash_krw)

        # 4. 요약 로그 (주식 트레이더와 형식 통일)
        ml_scores.sort(key=lambda x: x[1], reverse=True)
        top_ml_str = ", ".join([f"{s}({p:.2f})" for s, p in ml_scores[:3]])

        summary_msg = (
            f"📊 [COIN 스캔완료] 대상:{count_checked} 스킵:{count_skipped} "
            f"(장마감:{skip_market_closed}, 가격없음:{skip_no_price}, "
            f"데이터없음:{skip_no_df}, 캔들부족:{skip_short_df}) "
            f"| 매수후보:{len(entry_candidates)} "
            f"| 🔥ML Top3: [{top_ml_str}]"
        )
        self.db.log(summary_msg)
