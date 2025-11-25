# trader.py
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

SEQ_LEN = 60   # 시퀀스 길이 (train_seq_model.py와 동일)


def build_feature_from_seq(df_seq):
    """
    최근 60개 캔들을 기반으로 feature 생성
    - close/high/low → 첫 종가 대비 변화율
    - volume → 평균 대비 정규화
    """
    if len(df_seq) != SEQ_LEN:
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


class GlobalRealTimeTrader:
    def __init__(self, fetcher, targets, params, db, model=None, ml_threshold=0.55):
        self.fetcher = fetcher
        self.targets = targets
        self.params = params
        self.db = db
        self.trade_state = {}

        # 시퀀스 기반 ML 모델
        self.model = model
        self.ml_threshold = ml_threshold


    def is_market_open(self, region):
        now = datetime.now()
        if region == "KR":
            return (
                now.hour == 9 or
                (9 < now.hour < 15) or
                (now.hour == 15 and now.minute <= 20)
            )
        else:
            return (now.hour >= 21) or (now.hour < 6)


    # ------------------------------
    # 매수 집행(루프 마지막에 Top3만)
    # ------------------------------
    def execute_buys(self, candidates, kr_balance, us_balance, cash_krw, cash_usd):
        max_pos = 3
        held_kr = len(kr_balance)
        held_us = len(us_balance)

        remain_kr = max(0, max_pos - held_kr)
        remain_us = max(0, max_pos - held_us)

        # ML 점수가 높은 순
        candidates = sorted(
            candidates,
            key=lambda x: (x["ml_proba"] or 0),
            reverse=True
        )

        available_krw = cash_krw
        available_usd = cash_usd

        for c in candidates:
            region = c["region"]
            symbol = c["symbol"]
            excd = c["excd"]
            price = c["current_price"]
            ml_proba = c["ml_proba"]
            signal_id = c["signal_id"]

            if region == "KR" and remain_kr > 0:
                budget = available_krw / remain_kr
                qty = int(budget / price)
                if qty <= 0:
                    continue

                if self.fetcher.send_kr_order(symbol, "buy", qty):
                    cost = qty * price
                    available_krw -= cost
                    remain_kr -= 1

                    self.db.save_trade(
                        symbol, "BUY", price, qty,
                        profit=0,
                        signal_id=signal_id,
                        ml_proba=ml_proba,
                        entry_allowed=True
                    )
                    self.db.log(f"🚀[매수] {symbol} {qty}주 | ML={ml_proba:.3f}")

            elif region == "US" and remain_us > 0:
                budget = available_usd / remain_us
                qty = int(budget / price)
                if qty <= 0:
                    continue

                if self.fetcher.send_us_order(excd, symbol, "buy", qty, price):
                    cost = qty * price
                    available_usd -= cost
                    remain_us -= 1

                    self.db.save_trade(
                        symbol, "BUY", price, qty,
                        profit=0,
                        signal_id=signal_id,
                        ml_proba=ml_proba,
                        entry_allowed=True
                    )
                    self.db.log(f"🚀[매수] {symbol} {qty}주 | ML={ml_proba:.3f}")


    # ------------------------------
    # 메인 루프
    # ------------------------------
    def run_check(self):
        self.db.log("⏰ 감시 시작")

        # 잔고
        try:
            kr_balance = self.fetcher.get_kr_balance()
            us_balance = self.fetcher.get_us_balance()
            cash_krw = self.fetcher.get_kr_buyable_cash()
            cash_usd = self.fetcher.get_us_buyable_cash()
        except Exception as e:
            self.db.log(f"❌ 잔고 불러오기 실패: {e}")
            return

        entry_candidates = []
        count_checked = 0
        count_signals = 0

        for t in self.targets:
            region = t["region"]
            symbol = t["symbol"]
            excd = t.get("excd")

            time.sleep(0.2)

            if not self.is_market_open(region):
                continue

            # 현재가
            if region == "KR":
                price = self.fetcher.get_kr_current_price(symbol)
                has_stock = symbol in kr_balance
                my_info = kr_balance.get(symbol)
            else:
                price = self.fetcher.get_us_current_price(excd, symbol)
                has_stock = symbol in us_balance
                my_info = us_balance.get(symbol)

            if not price:
                continue

            df = self.fetcher.get_ohlcv(region, symbol, excd)
            if df is None or df.empty or len(df) < SEQ_LEN:
                continue

            interval = "5m" if region == "KR" else "1d"
            try:
                self.db.save_ohlcv_df(region, symbol, interval, df)
            except Exception as e:
                self.db.log(f"⚠️ OHLCV 저장 오류 {symbol}: {e}")

            count_checked += 1

            # 룰 기반 신호 계산
            df["support"] = df["low"].rolling(self.params["lookback"]).min()
            df["at_support"] = df["low"] <= df["support"] * (1 + self.params["band_pct"])
            df["is_bullish"] = df["close"] > df["open"]
            df["price_up"] = df["close"] > df["close"].shift(1)
            last = df.iloc[-1]

            entry_signal = bool(
                last["at_support"] and last["is_bullish"] and last["price_up"]
            )
            if entry_signal:
                count_signals += 1

            # 시퀀스 기반 ML feature
            df_seq = df.iloc[-SEQ_LEN:]
            seq_feat = build_feature_from_seq(df_seq)

            ml_proba = None
            if self.model is not None and seq_feat is not None:
                try:
                    ml_proba = float(self.model.predict_proba([seq_feat])[0][1])
                    self.db.log(f"🤖 ML[{symbol}] = {ml_proba:.3f}")
                except Exception as e:
                    self.db.log(f"⚠️ ML 오류 {symbol}: {e}")
                    ml_proba = None

            # 최종 진입 여부
            entry_allowed = entry_signal and (
                (ml_proba is not None) and (ml_proba >= self.ml_threshold)
            )

            # save_signal 저장
            signal_id = self.db.save_signal(
                region=region,
                symbol=symbol,
                price=float(last["close"]),
                at_support=bool(last["at_support"]),
                is_bullish=bool(last["is_bullish"]),
                price_up=bool(last["price_up"]),
                lookback=self.params["lookback"],
                band_pct=self.params["band_pct"],
                has_stock=has_stock,
                entry_signal=entry_signal,
                ml_proba=ml_proba,
                entry_allowed=entry_allowed,
                note="seq_model_check"
            )

            # 매수 후보 저장
            if entry_allowed and not has_stock:
                entry_candidates.append({
                    "region": region,
                    "symbol": symbol,
                    "excd": excd,
                    "current_price": price,
                    "ml_proba": ml_proba,
                    "signal_id": signal_id
                })

            # 매도 로직 (기존 동일)
            if has_stock and my_info:
                avg_price = my_info["avg_price"]
                qty = my_info["qty"]
                profit_rate = (price - avg_price) / avg_price

                state = self.trade_state.setdefault(symbol, {"tp1": False, "tp2": False})
                sell_qty = 0
                sell_type = ""

                if profit_rate >= 0.03 and not state["tp1"]:
                    sell_qty = max(1, int(qty * 0.5))
                    sell_type = "PROFIT_3%"
                    state["tp1"] = True

                elif profit_rate >= 0.05 and not state["tp2"]:
                    sell_qty = max(1, int(qty * 0.5))
                    sell_type = "PROFIT_5%"
                    state["tp2"] = True

                elif profit_rate <= -0.04:
                    sell_qty = qty
                    sell_type = "CUT_LOSS"
                    del self.trade_state[symbol]

                if sell_qty > 0:
                    if region == "KR":
                        success = self.fetcher.send_kr_order(symbol, "sell", sell_qty)
                    else:
                        success = self.fetcher.send_us_order(excd, symbol, "sell", sell_qty, price)

                    if success:
                        self.db.save_trade(symbol, sell_type, price, sell_qty, profit_rate * 100)
                        self.db.log(f"📉[매도] {symbol}: {sell_type} {sell_qty}주")


        # 루프 종료 → 후보들 중 상위만 매수
        self.execute_buys(entry_candidates, kr_balance, us_balance, cash_krw, cash_usd)

        self.db.log(
            f"📊 [종료] 조회 {count_checked}개 | 룰신호 {count_signals}개 | 후보 {len(entry_candidates)}개"
        )
