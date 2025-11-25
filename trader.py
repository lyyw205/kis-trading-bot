# trader.py
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

SEQ_LEN = 40   # 시퀀스 길이 (train_seq_model.py와 동일)


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

                # ✅ 시장별 매수 라운드 (0=첫 주문, 1=두번째, 2=세번째)
        self.buy_round = {"KR": 0, "US": 0}
        self.max_round = 3

        # ✅ 최소 주문 금액(브로커가 요구하는 수준에 맞춰 조정해도 됨)
        self.min_us_order_amount = 0.0   # 예: 30달러 이상만 주문
        self.min_kr_order_amount = 5000   # 예: 5천원 이상만 주문

    def is_market_open(self, region):
        now = datetime.now()
        if region == "KR":
            return (
                now.hour == 9 or
                (9 < now.hour < 15) or
                (now.hour == 15 and now.minute <= 20)
            )
        else:
            return (now.hour >= 23) or (now.hour < 6)


    # ------------------------------
    # 매수 집행(루프 마지막에 Top3만)
    # ------------------------------
    def execute_buys(self, candidates, kr_balance, us_balance, cash_krw, cash_usd):
        max_pos = 3
        held_kr = len(kr_balance)
        held_us = len(us_balance)

        remain_kr = max(0, max_pos - held_kr)
        remain_us = max(0, max_pos - held_us)

        self.db.log(
            f"🧮 [매수집행 진입] 후보:{len(candidates)} | "
            f"KR보유종목:{held_kr} / US보유종목:{held_us} | "
            f"남은슬롯 KR:{remain_kr}, US:{remain_us} | "
            f"KRW현금:{cash_krw}원 / USD매수파워:{cash_usd:.2f}$ | "
            f"라운드 KR:{self.buy_round['KR']} / US:{self.buy_round['US']}"
        )

        # ML 점수 높은 순으로 정렬
        candidates = sorted(
            candidates,
            key=lambda x: (x["ml_proba"] or 0),
            reverse=True
        )

        # 현재 가용 현금
        available_krw = cash_krw
        available_usd = cash_usd

        kr_bought_this_round = False
        us_bought_this_round = False

        for c in candidates:
            region = c["region"]
            symbol = c["symbol"]
            excd = c["excd"]
            price = c["current_price"]
            ml_proba = c["ml_proba"]
            signal_id = c["signal_id"]

            if region == "KR":
                # 슬롯 다 찼거나, 라운드 다 썼으면 신규 매수 안함
                if remain_kr <= 0:
                    self.db.log(f"⏭️ [KR슬롯없음] {symbol} 매수 스킵")
                    continue
                if self.buy_round["KR"] >= self.max_round:
                    self.db.log(f"⏭️ [KR라운드완료] {symbol} 매수 스킵 (buy_round={self.buy_round['KR']})")
                    continue
                if kr_bought_this_round:
                    # 한 번에 KR 신규 매수는 1종목만
                    continue

                # ✅ 라운드별 예산 비율
                r = self.buy_round["KR"]
                if r == 0:
                    budget = available_krw * 0.35   # 1차: 35%
                elif r == 1:
                    budget = available_krw * 0.60   # 2차: 남은 것의 60%
                else:
                    budget = available_krw          # 3차: 전액

                qty = int(budget / price)
                order_amount = qty * price

                self.db.log(
                    f"🧾 [KR수량계산] 라운드:{r+1} {symbol} | "
                    f"budget:{budget:.0f}원 / price:{price:.0f}원 → qty:{qty}, amount:{order_amount:.0f}"
                )

                if qty <= 0:
                    self.db.log(
                        f"⚠️ [US금액컷] {symbol}: "
                        f"예산으로 1주도 못 삼 (budget={budget:.2f}$, price={price:.2f}$)"
                    )
                    continue

                success = self.fetcher.send_kr_order(symbol, "buy", qty)
                if success:
                    available_krw -= order_amount
                    remain_kr -= 1
                    kr_bought_this_round = True

                    self.buy_round["KR"] = min(self.buy_round["KR"] + 1, self.max_round)
                    self.db.save_trade(
                        symbol, "BUY", price, qty,
                        profit=0,
                        signal_id=signal_id,
                        ml_proba=ml_proba,
                        entry_allowed=True
                    )
                    self.db.log(
                        f"✅🚀[KR매수체결] 라운드:{self.buy_round['KR']} {symbol} {qty}주 "
                        f"| 체결가:{price} | ML={ml_proba:.3f}"
                    )
                else:
                    self.db.log(
                        f"❌ [KR매수실패] {symbol} {qty}주 | 주문 API 실패 (amount={order_amount:.0f})"
                    )

            elif region == "US":
                if remain_us <= 0:
                    self.db.log(f"⏭️ [US슬롯없음] {symbol} 매수 스킵")
                    continue
                if self.buy_round["US"] >= self.max_round:
                    self.db.log(f"⏭️ [US라운드완료] {symbol} 매수 스킵 (buy_round={self.buy_round['US']})")
                    continue
                if us_bought_this_round:
                    # 한 번에 US 신규 매수는 1종목만
                    continue

                r = self.buy_round["US"]
                if r == 0:
                    budget = available_usd * 0.35   # 1차: 35%
                elif r == 1:
                    budget = available_usd * 0.60   # 2차: 남은 것의 60%
                else:
                    budget = available_usd          # 3차: 전액

                qty = int(budget / price)
                order_amount = qty * price

                self.db.log(
                    f"🧾 [US수량계산] 라운드:{r+1} {symbol} | "
                    f"budget:{budget:.2f}$ / price:{price:.2f}$ → qty:{qty}, amount:{order_amount:.2f}$"
                )

                if qty <= 0 or order_amount < self.min_us_order_amount:
                    self.db.log(
                        f"⚠️ [US금액컷] {symbol}: "
                        f"주문금액 ${order_amount:.2f} < 최소 ${self.min_us_order_amount:.2f} (qty={qty})"
                    )
                    continue

                success = self.fetcher.send_us_order(excd, symbol, "buy", qty, price)
                if success:
                    available_usd -= order_amount
                    remain_us -= 1
                    us_bought_this_round = True

                    self.buy_round["US"] = min(self.buy_round["US"] + 1, self.max_round)
                    self.db.save_trade(
                        symbol, "BUY", price, qty,
                        profit=0,
                        signal_id=signal_id,
                        ml_proba=ml_proba,
                        entry_allowed=True
                    )
                    self.db.log(
                        f"✅🚀[US매수체결] 라운드:{self.buy_round['US']} {symbol} {qty}주 "
                        f"| 체결가:{price} | ML={ml_proba:.3f}"
                    )
                else:
                    self.db.log(
                        f"❌ [US매수실패] {symbol} {qty}주 | 주문 API 실패 (amount={order_amount:.2f}$)"
                    )


    # ------------------------------
    # 메인 루프
    # ------------------------------
    def run_check(self):
        # 시작 로그도 시끄러우면 주석 처리 가능
        # self.db.log("⏰ 감시 시작") 

        # 잔고
        try:
            kr_balance = self.fetcher.get_kr_balance()
            us_balance = self.fetcher.get_us_balance()
            cash_krw = self.fetcher.get_kr_buyable_cash()
            cash_usd = self.fetcher.get_us_buyable_cash()
        except Exception as e:
            self.db.log(f"❌ 잔고 불러오기 실패: {e}")
            return
        
        self.db.log(
            f"💰 [잔고스냅샷] KR보유종목:{len(kr_balance)} / US보유종목:{len(us_balance)} | "
            f"KRW현금:{cash_krw}원 | USD매수파워:{cash_usd:.2f}$"
        )

        entry_candidates = []
        count_checked = 0
        count_skipped = 0
        count_signals = 0
        
        # ML 점수 기록용 리스트 (심볼, 점수)
        ml_scores = []

        for t in self.targets:
            region = t["region"]
            symbol = t["symbol"]
            excd = t.get("excd")

            time.sleep(0.2) # API 부하 조절

            if not self.is_market_open(region):
                self.db.log(
                    f"⏱️ [장외스킵] {region} {symbol} | "
                    f"현재시간={datetime.now().strftime('%H:%M')} → is_market_open=False"
                )
                count_skipped += 1
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
                # self.db.log(f"🚫 [Skip] {symbol}: 현재가 조회 실패")
                count_skipped += 1
                continue

            # 5분봉 조회
            interval = "5m"
            df = self.fetcher.get_ohlcv(region, symbol, excd, interval=interval, count=120)

            if df is None or df.empty:
                self.db.log(f"🚫 [Skip] {symbol}: 데이터 없음")
                count_skipped += 1
                continue
            
            if len(df) < SEQ_LEN:
                self.db.log(f"⚠️ [Skip] {symbol}: 데이터 부족")
                count_skipped += 1
                continue

            try:
                self.db.save_ohlcv_df(region, symbol, interval, df)
            except Exception as e:
                pass # 저장 에러는 조용히 넘어감

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
                    # 📝 개별 로그 대신 리스트에 추가
                    ml_scores.append((symbol, ml_proba))
                except Exception as e:
                    self.db.log(f"⚠️ [ML예외] {region} {symbol}: {e}")
                    ml_proba = None

            # 최종 진입 여부
            entry_allowed = entry_signal and (
                (ml_proba is not None) and (ml_proba >= self.ml_threshold)
            )

            # DB 저장 (Signals)
            signal_id = self.db.save_signal(
                region=region, symbol=symbol, price=float(last["close"]),
                at_support=bool(last["at_support"]), is_bullish=bool(last["is_bullish"]),
                price_up=bool(last["price_up"]), lookback=self.params["lookback"],
                band_pct=self.params["band_pct"], has_stock=has_stock,
                entry_signal=entry_signal, ml_proba=ml_proba, entry_allowed=entry_allowed,
                note="seq_model_check"
            )

            # 매수 후보 등록
            if entry_allowed and not has_stock:
                entry_candidates.append({
                    "region": region, "symbol": symbol, "excd": excd,
                    "current_price": price, "ml_proba": ml_proba, "signal_id": signal_id
                })

            # 매도 로직 (기존 동일)
            if has_stock and my_info:
                avg_price = my_info["avg_price"]
                qty = my_info["qty"]
                profit_rate = (price - avg_price) / avg_price
                state = self.trade_state.setdefault(symbol, {"tp1": False, "tp2": False})
                sell_qty = 0
                sell_type = ""

                if profit_rate >= 0.02 and not state["tp1"]:
                    sell_qty = max(1, int(qty * 0.6))
                    sell_type = "PROFIT_3%"
                    state["tp1"] = True
                elif profit_rate >= 0.05 and not state["tp2"]:
                    sell_qty = max(1, int(qty * 0.4))
                    sell_type = "PROFIT_5%"
                    state["tp2"] = True
                elif profit_rate <= -0.02:
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

        # 루프 종료 후 매수 집행
        self.execute_buys(entry_candidates, kr_balance, us_balance, cash_krw, cash_usd)

        # ✨ [깔끔해진 최종 로그]
        # ML 점수 상위 3개만 추려서 보여줌
        ml_scores.sort(key=lambda x: x[1], reverse=True)
        top_ml_str = ", ".join([f"{s}({p:.2f})" for s, p in ml_scores[:3]])
        
        summary_msg = (
            f"📊 [스캔완료] 대상:{count_checked} 스킵:{count_skipped} "
            f"| 매수후보:{len(entry_candidates)} "
            f"| 🔥ML Top3: [{top_ml_str}]"
        )
        self.db.log(summary_msg)
