# trader.py
import time
from datetime import datetime
import joblib
import numpy as np
import pandas as pd
from ai_helpers import make_entry_comment, make_exit_comment

from ml_features import SEQ_LEN, calculate_rsi, build_feature_from_seq
from st_entry_kr import make_entry_signal_kr
from st_entry_us import make_entry_signal_us
from tcn_entry_cr import make_entry_signal_coin_ms
from st_exit_common import decide_exit




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
        elif region == "US":  # US (서머타임 고려 필요 시 별도 로직 추가, 기본값 유지)
            return (
                (now.hour == 23 and now.minute >= 30) or   # 23:30 ~ 23:59
                (0 <= now.hour < 6)                        # 00:00 ~ 05:59
            )
        elif region == "COIN":
            return True  # 24시간 오픈
        else:
            return False

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
            strategy_name = c.get("strategy_name", "UNKNOWN")

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

                    # ✅ trade_id 받기
                    trade_id = self.db.save_trade(
                        symbol,
                        "BUY",
                        price,
                        qty,
                        0,
                        signal_id=signal_id,
                        ml_proba=ml_proba,
                        entry_allowed=True,
                        region=region
                    )
                    self.db.log(f"✅🚀[KR매수] {symbol} {qty}주 | ML:{ml_proba:.3f}")

                    # 🔹 AI 진입 코멘트 생성 + DB 저장
                    try:
                        entry_ctx = {
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "region": region,
                            "symbol": symbol,
                            "exchange": None,
                            "side": "BUY",
                            "qty": qty,
                            "price": float(price),
                            "ml_proba": ml_proba,
                            "strategy": strategy_name,
                        }
                        comment = make_entry_comment(entry_ctx)
                        self.db.update_trade_entry_comment(trade_id, comment)
                    except Exception as e:
                        self.db.log(f"⚠️ [AI진입코멘트 실패] {symbol} | {e}")

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

                    trade_id = self.db.save_trade(
                        symbol,
                        "BUY",
                        price,
                        qty,
                        0,
                        signal_id=signal_id,
                        ml_proba=ml_proba,
                        entry_allowed=True,
                        region=region
                    )
                    self.db.log(f"✅🚀[US매수] {symbol} {qty}주 | ML:{ml_proba:.3f}")

                    try:
                        entry_ctx = {
                            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "region": region,
                            "symbol": symbol,
                            "exchange": excd,
                            "side": "BUY",
                            "qty": qty,
                            "price": float(price),
                            "ml_proba": ml_proba,
                            "strategy": strategy_name,
                        }
                        comment = make_entry_comment(entry_ctx)
                        self.db.update_trade_entry_comment(trade_id, comment)
                    except Exception as e:
                        self.db.log(f"⚠️ [AI진입코멘트 실패] {symbol} | {e}")

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

            # -===========================================================
            # [전략 로직] - 자산군별 엔트리 함수 호출
            # -----------------------------------------------------------
            if region == "KR":
                sig = make_entry_signal_kr(df, self.params)
            elif region == "US":
                sig = make_entry_signal_us(df, self.params)
            elif region == "COIN":
                sig = make_entry_signal_coin_ms(df, self.params)
            else:
                # 혹시 모르는 region 값에 대한 방어
                sig = {
                    "entry_signal": False,
                    "strategy_name": "NONE",
                    "at_support": False,
                    "is_bullish": False,
                    "price_up": False,
                }

            entry_signal = sig["entry_signal"]
            strategy_name = sig["strategy_name"]
            at_support = sig["at_support"]
            is_bullish = sig["is_bullish"]
            price_up = sig["price_up"]

            if entry_signal:
                count_signals += 1

            #===========================================================

            

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
                price=float(df["close"].iloc[-1]),
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
                        "strategy_name": strategy_name,
                    }
                )

            # (7) 매도 로직 (이익실현 및 손절)
            if has_stock and my_info:
                avg_price = my_info["avg_price"]
                qty = my_info["qty"]

                # 심볼별 상태 초기화 / 가져오기
                state = self.trade_state.setdefault(
                    symbol,
                    {
                        "tp1": False,
                        "tp2": False,
                        "entry_time": datetime.utcnow(),   # 매수 시점에 따로 세팅하면 더 좋음
                        "max_profit": 0.0,
                    },
                )

                now = datetime.utcnow()

                # 공통 청산 로직 호출
                sell_qty, sell_type, new_state, profit_rate, elapsed_min = decide_exit(
                    symbol=symbol,
                    region=region,
                    price=price,
                    avg_price=avg_price,
                    qty=qty,
                    state=state,
                    now=now,
                )

                # 상태 업데이트 / 삭제
                if new_state.get("delete"):
                    if symbol in self.trade_state:
                        del self.trade_state[symbol]
                else:
                    self.trade_state[symbol] = new_state

                # === 실제 주문 전송 ===
                if sell_qty > 0:
                    if region == "KR":
                        success = self.fetcher.send_kr_order(symbol, "sell", sell_qty)
                    else:
                        success = self.fetcher.send_us_order(excd, symbol, "sell", sell_qty, price)

                    if success:
                        trade_id = self.db.save_trade(
                            symbol,
                            sell_type,                # type
                            price,
                            sell_qty,
                            profit_rate * 100,        # profit (퍼센트)
                        )
                        self.db.log(
                            f"📉[매도] {symbol}: {sell_type} {sell_qty}주 ({profit_rate*100:.2f}%)"
                        )

                        # 🔹 AI 청산 코멘트 생성 + DB 저장
                        try:
                            exit_ctx = {
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "region": region,
                                "symbol": symbol,
                                "exchange": excd if region == "US" else None,
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
                            self.db.log(f"⚠️ [AI청산코멘트 실패] {symbol} | {e}")

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