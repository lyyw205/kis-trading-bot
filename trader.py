# trader.py
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd

SEQ_LEN = 30   # 시퀀스 길이 (train_seq_model.py와 동일)

# -----------------------------------------------------------
# 헬퍼 함수 (RSI 계산, Feature 생성)
# -----------------------------------------------------------
def calculate_rsi(series, period=14):
    """RSI(상대강도지수) 계산"""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50) # 초기값은 50으로 처리

def build_feature_from_seq(df_seq):
    """
    최근 30개 캔들을 기반으로 feature 생성
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


# -----------------------------------------------------------
# 트레이더 클래스
# -----------------------------------------------------------
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

        # 최소 주문 금액 설정
        self.min_us_order_amount = 0.0    
        self.min_kr_order_amount = 5000   

    def is_market_open(self, region):
        now = datetime.now()
        if region == "KR":
            # 09:00 ~ 15:20
            return (
                now.hour == 9 or
                (9 < now.hour < 15) or
                (now.hour == 15 and now.minute <= 20)
            )
        else:  # US (서머타임 고려 필요 시 별도 로직 추가, 기본값 유지)
            return (
                (now.hour == 23 and now.minute >= 30) or   # 23:30 ~ 23:59
                (0 <= now.hour < 6)                        # 00:00 ~ 05:59
            )

    # ------------------------------
    # 매수 집행 로직
    # ------------------------------
    def execute_buys(self, candidates, kr_balance, us_balance, cash_krw, cash_usd):
        max_pos = 3

        held_kr = len(kr_balance)
        held_us = len(us_balance)
        total_held = held_kr + held_us
        remain_slots = max(0, max_pos - total_held)

        self.db.log(
            f"🧮 [매수집행] 후보:{len(candidates)} | "
            f"KR보유:{held_kr} / US보유:{held_us} | "
            f"총보유:{total_held}/{max_pos} | 남은슬롯:{remain_slots} | "
            f"KRW:{cash_krw}원 / USD:{cash_usd:.2f}$"
        )

        if remain_slots <= 0:
            self.db.log("⏭️ [슬롯없음] 신규 매수 전부 스킵")
            return

        # 중복 제거
        held_symbols = set(kr_balance.keys()) | set(us_balance.keys())
        filtered = [c for c in candidates if c["symbol"] not in held_symbols]

        if not filtered:
            self.db.log("⏭️ [후보없음] 신규 매수 대상 없음")
            return

        # ML 점수 높은 순 정렬
        filtered.sort(key=lambda x: (x["ml_proba"] or 0), reverse=True)
        targets_to_buy = filtered[:remain_slots]

        available_krw = cash_krw
        available_usd = cash_usd
        slots_left = remain_slots
        success_new = 0 

        for c in targets_to_buy:
            if slots_left <= 0:
                break

            region = c["region"]
            symbol = c["symbol"]
            excd = c["excd"]
            price = c["current_price"]
            ml_proba = c["ml_proba"]
            signal_id = c["signal_id"]

            buy_index = success_new

            # 분할 매수 비율 설정
            if buy_index == 0:
                min_ratio, max_ratio = 0.30, 0.40
            elif buy_index == 1:
                min_ratio, max_ratio = 0.40, 0.60
            else:
                min_ratio, max_ratio = 1.0, 1.0

            ratio = (min_ratio + max_ratio) / 2.0

            # --- KR 매수 ---
            if region == "KR":
                if available_krw <= 0:
                    continue
                
                budget = available_krw * ratio
                qty = int(budget / price)
                amount = qty * price

                if qty <= 0:
                    self.db.log(f"⚠️ [KR금액컷] {symbol} QTY=0 (Budget:{budget:.0f})")
                    continue

                success = self.fetcher.send_kr_order(symbol, "buy", qty)
                if success:
                    available_krw -= amount
                    slots_left -= 1
                    total_held += 1
                    success_new += 1
                    self.db.save_trade(symbol, "BUY", price, qty, 0, signal_id, ml_proba, True)
                    self.db.log(f"✅🚀[KR매수] {symbol} {qty}주 | ML:{ml_proba:.3f}")

            # --- US 매수 ---
            elif region == "US":
                if available_usd <= 0:
                    continue

                budget = available_usd * ratio
                qty = int(budget / price)
                amount = qty * price

                if qty <= 0 or amount < self.min_us_order_amount:
                    self.db.log(f"⚠️ [US금액컷] {symbol} QTY={qty}, Amt=${amount:.2f}")
                    continue

                success = self.fetcher.send_us_order(excd, symbol, "buy", qty, price)
                if success:
                    available_usd -= amount
                    slots_left -= 1
                    total_held += 1
                    success_new += 1
                    self.db.save_trade(symbol, "BUY", price, qty, 0, signal_id, ml_proba, True)
                    self.db.log(f"✅🚀[US매수] {symbol} {qty}주 | ML:{ml_proba:.3f}")

    # ------------------------------
    # 메인 체크 루프 (수정 완료)
    # ------------------------------
    def run_check(self):
        self.db.log(f"🔍 [DEBUG] KIS 모드: {self.fetcher.mode}")
        # 1. 잔고 및 현금 조회
        try:
            kr_balance = self.fetcher.get_kr_balance()
            us_balance = self.fetcher.get_us_balance()
            cash_krw = self.fetcher.get_kr_buyable_cash()
            cash_usd = self.fetcher.get_us_buyable_cash()
        except Exception as e:
            self.db.log(f"❌ 잔고 불러오기 실패: {e}")
            return

        self.db.log(
            f"💰 [잔고스냅샷] KR보유:{len(kr_balance)} / US보유:{len(us_balance)} | "
            f"KRW:{cash_krw}원 | USD:{cash_usd:.2f}$"
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

        # 2. 종목 스캔 루프
        for t in self.targets:
            region = t["region"]
            symbol = t["symbol"]
            excd = t.get("excd")

            time.sleep(0.2)  # API 과부하 방지

            # ------------------------------
            # (1) 장 운영 시간 확인
            # ------------------------------
            if not self.is_market_open(region):
                skip_market_closed += 1
                count_skipped += 1
                # self.db.log(f"⏱️ [Skip:장마감] {symbol}")
                continue

            # ------------------------------
            # (2) 현재가 조회
            # ------------------------------
            if region == "KR":
                price = self.fetcher.get_kr_current_price(symbol)
                has_stock = (symbol in kr_balance) or (symbol in self.trade_state)
                my_info = kr_balance.get(symbol)
            else:
                price = self.fetcher.get_us_current_price(excd, symbol)
                has_stock = (symbol in us_balance) or (symbol in self.trade_state)
                my_info = us_balance.get(symbol)

            if not price:
                skip_no_price += 1
                count_skipped += 1
                # self.db.log(f"🚫 [Skip:가격없음] {symbol}")
                continue

            # ------------------------------
            # (3) 캔들(5분봉) 조회
            # ------------------------------
            interval = "5m"
            if region == "KR":
                df = self.fetcher.get_ohlcv(
                    region,
                    symbol,
                    interval=interval,
                    count=120,
                )
            else:
                df = self.fetcher.get_ohlcv(
                    region,
                    symbol,
                    excd,
                    interval=interval,
                    count=120,
                )

            if df is None or df.empty:
                skip_no_df += 1
                count_skipped += 1
                # self.db.log(f"🚫 [Skip:캔들없음] {symbol}")
                continue

            if len(df) < SEQ_LEN:
                skip_short_df += 1
                count_skipped += 1
                # self.db.log(f"🚫 [Skip:캔들부족] {symbol} len={len(df)}")
                continue

            # ✅ 여기까지 통과한 종목만 진짜로 "대상"으로 카운트
            count_checked += 1

            # 데이터 저장
            try:
                self.db.save_ohlcv_df(region, symbol, interval, df)
            except Exception:
                pass

            # -----------------------------------------------------------
            # [전략 로직 수정 구간] - Momentum & Reversal
            # -----------------------------------------------------------
            
            # (1) 보조지표 계산
            # - Support: 과거 N봉 최저가
            df["support"] = df["low"].rolling(self.params["lookback"]).min()
            
            # - MA (이동평균선)
            df["ma20"] = df["close"].rolling(20).mean()
            df["ma60"] = df["close"].rolling(60).mean()  # 60선 추가 (장기 추세용)

            # - RSI (상대강도지수)
            df["rsi"] = calculate_rsi(df["close"], 14)

            # - Volume MA (거래량 이평)
            df["vol_ma20"] = df["volume"].rolling(20).mean()

            # (2) 마지막 봉 데이터 및 상태 플래그
            last = df.iloc[-1]
            prev = df.iloc[-2]

            # 기본 조건
            is_bullish = last["close"] > last["open"]       # 양봉
            price_up = last["close"] > prev["close"]        # 전일 대비 상승

            # -------------------------------------------------------
            # A. 역추세 전략 (Reversal)
            # 조건: 바닥권(Support 근처) + 양봉 + 상승반전
            # -------------------------------------------------------
            at_support = last["low"] <= last["support"] * (1 + self.params["band_pct"])
            
            sig_reversal = bool(at_support and is_bullish and price_up)

            # -------------------------------------------------------
            # B. 추세추종 전략 (Momentum Strong)
            # 조건: 정배열(20>60) + RSI적절(50~75) + 거래량증가
            # -------------------------------------------------------
            
            # 1. 정배열: 가격 > 20선 > 60선 (완벽한 상승 추세)
            cond_align = (last["close"] > last["ma20"]) and (last["ma20"] > last["ma60"])
            
            # 2. RSI: 50 이상(힘 좋음) ~ 75 이하(아직 꼭지는 아님)
            cond_rsi = (50 <= last["rsi"] <= 75)
            
            # 3. 거래량: 현재 거래량이 20이평보다 많음 (수급 들어옴)
            cond_vol = (prev["volume"] > last["vol_ma20"]) or \
                       (last["volume"] > last["vol_ma20"] * 0.4)

            sig_momentum = bool(
                cond_align and cond_rsi and cond_vol and is_bullish
            )

            # -------------------------------------------------------

            # 최종 신호: 둘 중 하나라도 만족
            entry_signal = sig_reversal or sig_momentum

            strategy_name = "NONE"
            if sig_reversal:
                strategy_name = "REVERSAL"
            elif sig_momentum:
                strategy_name = "MOMENTUM_STRONG"
            
            # -----------------------------------------------------------

            if entry_signal:
                count_signals += 1

            # (4) 머신러닝(ML) 점수 계산
            df_seq = df.iloc[-SEQ_LEN:]
            seq_feat = build_feature_from_seq(df_seq)

            ml_proba = None
            if self.model is not None and seq_feat is not None:
                try:
                    ml_proba = float(self.model.predict_proba([seq_feat])[0][1])
                    ml_scores.append((symbol, ml_proba))
                except Exception as e:
                    self.db.log(f"⚠️ [ML예외] {region} {symbol}: {e}")
                    ml_proba = None

            # (5) 최종 진입 허용 여부 (Rule Signal + ML Score)
            entry_allowed = entry_signal and (
                (ml_proba is not None) and (ml_proba >= self.ml_threshold)
            )

            # (6) 신호 DB 저장
            signal_id = self.db.save_signal(
                region=region,
                symbol=symbol,
                price=float(last["close"]),
                at_support=bool(at_support),
                is_bullish=bool(is_bullish),
                price_up=bool(price_up),
                lookback=self.params["lookback"],
                band_pct=self.params["band_pct"],
                has_stock=has_stock,
                entry_signal=entry_signal,
                ml_proba=ml_proba,
                entry_allowed=entry_allowed,
                note=strategy_name, 
            )

            # 매수 후보 등록 (미보유 종목만)
            if entry_allowed and not has_stock:
                entry_candidates.append(
                    {
                        "region": region,
                        "symbol": symbol,
                        "excd": excd,
                        "current_price": price,
                        "ml_proba": ml_proba,
                        "signal_id": signal_id,
                    }
                )

            # (7) 매도 로직 (이익실현 및 손절)
            if has_stock and my_info:
                avg_price = my_info["avg_price"]
                qty = my_info["qty"]
                profit_rate = (price - avg_price) / avg_price   # 소수 (0.03 = 3%)

                # 심볼별 상태 초기화 / 가져오기
                state = self.trade_state.setdefault(symbol, {
                    "tp1": False,
                    "tp2": False,
                    "entry_time": datetime.utcnow(),   # 매수 시점에 세팅해두는 게 제일 좋고,
                    "max_profit": 0.0                  # 여기서는 안전용으로 기본값만 둠
                })

                now = datetime.utcnow()
                # 익절/횡보 판단을 위해 최고 수익률 갱신
                state["max_profit"] = max(state.get("max_profit", 0.0), profit_rate)

                elapsed_min = (now - state.get("entry_time", now)).total_seconds() / 60.0

                sell_qty = 0
                sell_type = ""

                # 1) 손절 (-2% 도달 시 전량 매도)
                if profit_rate <= -0.02:
                    sell_qty = qty
                    sell_type = "CUT_LOSS"
                    if symbol in self.trade_state:
                        del self.trade_state[symbol]

                # 2) 익절 2차 (5% 수익 시 남은 것의 40% 매도)
                #    * 만약 처음부터 5%를 한방에 넘겼다면, tp1 안 찍고 바로 여기로 들어올 수도 있음
                elif profit_rate >= 0.05 and not state["tp2"]:
                    # 현재 수량 기준으로 40% 익절
                    sell_qty = max(1, int(qty * 0.4))
                    sell_type = "PROFIT_5%"
                    state["tp2"] = True
                    # tp1 안 찍고 바로 5% 온 경우, tp1도 찍힌 것으로 처리
                    state["tp1"] = True

                # 3) 익절 1차 (3% 수익 시 60% 매도)
                elif profit_rate >= 0.03 and not state["tp1"]:
                    sell_qty = max(1, int(qty * 0.6))
                    sell_type = "PROFIT_3%"
                    state["tp1"] = True
                    # 1차 익절 시점에 기준 시간 리셋 (원하면 유지해도 됨)
                    state["entry_time"] = now

                # 4) 1차 익절 이후, 5% 못 가고 3%로 되돌림 → 전량 매도
                elif state["tp1"] and not state["tp2"]:
                    # 어느 정도 위로 갔다가 (예: 4% 이상)
                    # 다시 3% 근처(약간 여유를 둬서 2.8% 이하)로 내려오면 정리
                    if state.get("max_profit", 0.0) >= 0.04 and profit_rate <= 0.028:
                        sell_qty = qty
                        sell_type = "EXIT_RETRACE_TP1"
                        if symbol in self.trade_state:
                            del self.trade_state[symbol]

                # 5) 2차 익절 이후, 상승 끊기거나 5% 되돌림 → 전량 매도
                elif state["tp2"]:
                    max_p = state.get("max_profit", 0.0)
                    # (a) 7% 이상 갔다가 5% 부근까지 되돌림
                    if max_p >= 0.07 and profit_rate <= 0.052:
                        sell_qty = qty
                        sell_type = "EXIT_RETRACE_TP2"
                        if symbol in self.trade_state:
                            del self.trade_state[symbol]

                    # 필요하다면 여기서 "상승 캔들 끊김" 조건(이전 종가 대비 하락 등)도
                    # state에 이전 가격/종가를 저장해두고 추가로 체크 가능

                # 6) 횡보 정리 조건 (익절/손절 사이에서 애매하게 기는 애들 강제 퇴장)

                # 6-1) 진입 후 60분 동안 -1.5% ~ +2.5% 박스 안에서만 움직이면 전량 매도
                if sell_qty == 0 and elapsed_min >= 60 and -0.015 <= profit_rate <= 0.025:
                    sell_qty = qty
                    sell_type = "TIMEOUT_NO_TP"
                    if symbol in self.trade_state:
                        del self.trade_state[symbol]

                # 6-2) 1차 익절 후 45분 안에 5% 도달 못하고 +2.5% ~ +4.5%에서 횡보 → 전량 매도
                if sell_qty == 0 and state["tp1"] and not state["tp2"] and elapsed_min >= 45:
                    if 0.025 <= profit_rate <= 0.045:
                        sell_qty = qty
                        sell_type = "TIMEOUT_AFTER_TP1"
                        if symbol in self.trade_state:
                            del self.trade_state[symbol]

                # === 실제 주문 전송 ===
                if sell_qty > 0:
                    if region == "KR":
                        success = self.fetcher.send_kr_order(symbol, "sell", sell_qty)
                    else:
                        success = self.fetcher.send_us_order(excd, symbol, "sell", sell_qty, price)
                    
                    if success:
                        self.db.save_trade(symbol, sell_type, price, sell_qty, profit_rate * 100)
                        self.db.log(f"📉[매도] {symbol}: {sell_type} {sell_qty}주 ({profit_rate*100:.2f}%)")

        # ── for t in self.targets: 끝 ──

        # 3. 매수 집행 (전체 스캔 끝난 뒤에 한 번만)
        self.execute_buys(entry_candidates, kr_balance, us_balance, cash_krw, cash_usd)

        # 4. 요약 로그
        ml_scores.sort(key=lambda x: x[1], reverse=True)
        top_ml_str = ", ".join([f"{s}({p:.2f})" for s, p in ml_scores[:3]])

        summary_msg = (
            f"📊 [스캔완료] 대상:{count_checked} 스킵:{count_skipped} "
            f"(장마감:{skip_market_closed}, 가격없음:{skip_no_price}, "
            f"데이터없음:{skip_no_df}, 캔들부족:{skip_short_df}) "
            f"| 매수후보:{len(entry_candidates)} "
            f"| 🔥ML Top3: [{top_ml_str}]"
        )
        self.db.log(summary_msg)