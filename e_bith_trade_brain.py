"""코인(CR) 전용 실시간 트레이더 (Bithumb, Multi-Strategy TCN 엔진)

 - BithumbDataFetcher + trades/signals DB를 사용해서 코인 자동매매를 수행하는 메인 트레이딩 브레인
 - GlobalRealTimeTrader 인터페이스를 최대한 맞춘 CR 전용 버전 (region='CR' 고정)

주요 기능:
1) 초기화
   - fetcher(BithumbDataFetcher), targets(CR 유니버스), params(DEFAULT_ENTRY_PARAMS_MS 기반), BotDatabase, ML 모델, ML threshold 설정
   - 엔트리 전략 세트(MS/REV/MOMO) 등록, 코인 최소 주문금액/최대 포지션 수/재진입 쿨다운 시간 설정
   - 현재 사용 중인 ENTRY/EXIT 버전 로그 출력

2) 상태 관리
   - trade_state: 심볼별 진입 시점, ML score, ATR 비율, 전략 이름 등을 저장
   - pending_orders: 미체결 주문(주문 ID, 생성시간, 방향) 관리
   - last_exit_time: 심볼별 마지막 청산 시각 저장 → 재진입 쿨다운 로직에 사용
   - DB 재실행 복구: _restore_entry_state_from_db()로 trades 테이블의 마지막 BUY 기록을 읽어 entry_time / ml_proba를 복구

3) 주문/쿨다운 유틸
   - is_market_open(): CR(코인)은 항상 장 열림으로 처리
   - _is_in_cooldown(): 마지막 청산 이후 reentry_cooldown_min(기본 60분) 이내면 해당 코인 신규 진입 차단
   - cancel_stale_orders(): 일정 시간(max_wait_sec, 기본 30초) 이상 체결 안 된 pending 주문은 취소 시도 후 관리 목록에서 제거

4) 매수 엔진 (execute_buys)
   - '전액 1종목' 모드:
     · 이미 포지션이나 미체결 주문이 있으면 신규 매수 전체 스킵
     · 엔트리 후보 중 ML 점수가 가장 높은 1개만 선택
     · Bithumb get_order_chance()로 최소 주문 금액(min_total) 및 실제 주문 가능 잔액 확인
     · 예산(safety_factor 0.98 적용) < 최소 주문 기준이면 매수 스킵
     · 조건을 만족하면 지정가(limit) 매수 주문 전송
   - 주문 성공 시:
     · trade_state에 엔트리 정보 저장 (entry_time, ml_score_entry, ml_worst_entry, atr_ratio_entry, strategy_name)
     · pending_orders에 주문 정보 저장
     · trades 테이블에 BUY 기록 저장 + AI 엔트리 코멘트(make_entry_comment) 생성 후 trade에 업데이트

5) 메인 루프 (run_check)
   - ① pending 주문 정리: cancel_stale_orders()
   - ② Bithumb에서 코인 잔고 및 매수 가능 현금(KRW) 조회
   - ③ 현재 보유 포지션 기준으로 entry_state 복구(_restore_entry_state_from_db)
   - ④ 유니버스 targets 순회:
     · region != "CR" 은 방어 차원으로 스킵
     · 포지션 보유 중이면 내가 들고 있는 코인만 청산 감시, 나머지 심볼은 스캔 스킵
     · 현재가/5분봉 OHLCV 조회, 캔들이 SEQ_LEN 미만이면 스킵
     · OHLCV를 DB에 save_ohlcv_df로 저장 (오류는 무시)
   - ⑤ 보유 포지션에 대한 청산 판단:
     · CrPosition 생성 후 decide_exit_cr 호출 → should_exit이면 마켓가 매도 전송
     · 청산 성공 시 trades에 SELL 기록 저장, last_exit_time 갱신, trade_state에서 제거
     · AI 청산 코멘트(make_exit_comment) 생성 후 trade에 업데이트
   - ⑥ 포지션이 없을 때의 엔트리 스캔:
     · 재진입 쿨다운 중인 코인은 후보 제외
     · 심볼별 df/현재가를 모아두고, pick_best_entry_across_universe() 호출
       - MS / REV / MOMO 중 최종 entry 1개 선택
       - AI_PARAMS_COIN["ms_min_final_score"] 이상일 때만 엔트리 허용
     · 선택된 엔트리 신호는 signals 테이블에 저장(save_signal)
     · execute_buys()에서 요구하는 형태의 후보 딕셔너리를 만들어 전액 매수 로직 호출
   - ⑦ 스캔 요약 로그:
     · 스캔/스킵 카운트, 매수 후보 수, ML Top3 목록, 엔트리 요약(상위 N개 코인에 대해 score/ATR) 로그 출력"""


import time
from datetime import datetime
import math
import joblib
import numpy as np
import pandas as pd

from ai_helpers import make_entry_comment, make_exit_comment
from c_ml_features import SEQ_LEN
from bi_entry_hub import (
    make_entry_signal_coin_ms,
    make_entry_signal_coin_rev,
    make_entry_signal_coin_momo,
    pick_best_entry_across_universe,
    ENTRY_VERSION,
    DEFAULT_ENTRY_PARAMS_MS,
)
from bi_exit_lib import CrPosition, EXIT_VERSION
from bi_exit_hub import decide_exit_cr
from e_bithumb_client import BithumbDataFetcher
from c_db_manager import BotDatabase
from c_config import AI_PARAMS_COIN


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
        base_params = DEFAULT_ENTRY_PARAMS_MS.copy()
        if params:
            base_params.update(params)
        self.params = base_params            # lookback, band_pct 등
        self.db = db
        self.trade_state = {}           # 심볼별 TP/SL 상태 저장 (st_exit_common과 호환)
        self.pending_orders: dict[str, dict] = {}
        # ML 모델 (RandomForest 등)
        self.model = model
        self.ml_threshold = ml_threshold

        # 코인용 최소 주문 기준 (KRW)
        self.min_order_amount_krw = 5_000

        # 포지션 최대 개수 (코인 전용)
        self.max_pos = 1

        # ✅ 재진입 쿨다운 (분 단위, 예: 같은 코인 60분 동안 재진입 금지)
        self.reentry_cooldown_min = 60
        self.last_exit_time: dict[str, datetime] = {}

        self.entry_strategies = {
            "MS": make_entry_signal_coin_ms,
            "REV": make_entry_signal_coin_rev,
            "MOMO": make_entry_signal_coin_momo,
        }

        self.min_final_score = 0.006

        self.db.log(
            "🔄 [COIN] Entry engine = Multi-Scale TCN+Transformer "
            "(tcn_entry_cr.make_entry_signal_coin_ms 사용 중)"
        )
        log_fn = getattr(self.db, "log", print)

        log_fn(
            f"📦 [CR_ENTRY_LOADED] {ENTRY_VERSION} "
            f"({make_entry_signal_coin_ms.__code__.co_filename})"
        )
        log_fn(
            f"📦 [CR_EXIT_LOADED] {EXIT_VERSION} "
            f"({decide_exit_cr.__code__.co_filename})"
        )

    def _truncate_qty(self, qty: float, precision: int = 4) -> float:
        factor = 10 ** precision
        return math.floor(qty * factor) / factor

    # ------------------------------------------------
    # 코인은 24시간 장이라 장 마감 체크는 간단하게
    # ------------------------------------------------
    def is_market_open(self, region: str) -> bool:
        # CR 은 항상 True
        if region == "CR":
            return True
        return False

    def _is_in_cooldown(self, market: str) -> bool:
        last = self.last_exit_time.get(market)
        if not last:
            return False

        elapsed_min = (datetime.now() - last).total_seconds() / 60.0
        if elapsed_min < self.reentry_cooldown_min:
            self.db.log(
                f"⏸️ [COIN 재진입쿨다운] {market} 마지막 청산 후 {elapsed_min:.1f}분 경과 "
                f"(쿨다운 {self.reentry_cooldown_min}분)"
            )
            return True
        return False
    
    # ------------------------------------------------
    # 재실행 시 DB에서 엔트리 정보 복구
    # ------------------------------------------------
    def _restore_entry_state_from_db(self, coin_balance: dict):
        """
        거래소 잔고(coin_balance)를 기준으로,
        trades 테이블에서 마지막 BUY 트레이드를 찾아서
        self.trade_state[market]["entry_time"] 등을 복구.

        - 프로그램 재실행 이후에도 TIMEOUT, ML exit가 정상 동작하도록 하기 위함.
        """
        if not coin_balance:
            return

        # 이미 복구되어 있으면 다시 안 함 (한 번만)
        if self.trade_state:
            return

        try:
            conn = self.db.get_connection()
        except Exception as e:
            self.db.log(f"⚠️ [COIN 복구실패] DB 연결 실패: {e}")
            return

        try:
            cur = conn.cursor()
            for market, info in coin_balance.items():
                # 이미 trade_state에 있으면 건너뜀
                if market in self.trade_state:
                    continue

                try:
                    # ✅ PostgreSQL 스타일 플레이스홀더(%s) 사용
                    cur.execute(
                        """
                        SELECT time, ml_proba
                        FROM trades
                        WHERE region = %s
                          AND symbol = %s
                        ORDER BY time DESC
                        LIMIT 1
                        """,
                        ("CR", market),
                    )
                    row = cur.fetchone()
                except Exception as e:
                    self.db.log(f"⚠️ [COIN 복구쿼리실패] {market} | {e}")
                    continue

                if not row:
                    # 해당 코인에 대한 BUY 기록이 없으면 복구할 게 없음
                    self.db.log(f"ℹ️ [COIN 복구대상없음] {market} BUY 트레이드 없음")
                    continue

                raw_time = row[0]
                ml_proba = row[1] if len(row) > 1 else None

                # time 컬럼을 datetime으로 변환
                try:
                    entry_time = pd.to_datetime(raw_time)
                except Exception:
                    try:
                        entry_time = datetime.fromisoformat(raw_time)
                    except Exception:
                        entry_time = datetime.now()

                self.trade_state[market] = {
                    "entry_time": entry_time,
                    "ml_score_entry": ml_proba,   # DB에 저장된 ml_proba 재사용
                    "ml_worst_entry": None,       # 아직 컬럼 없으니 None
                    "atr_ratio_entry": None,
                }

                self.db.log(
                    f"🔁 [COIN 포지션복구] {market} "
                    f"entry_time={entry_time}, ml_score_entry={ml_proba}"
                )

            cur.close()
        finally:
            try:
                conn.close()
            except Exception:
                pass
    

    # ------------------------------------------------
    # 매수 집행 로직 (주식 트레이더와 최대한 비슷하게)
    # ------------------------------------------------
    def execute_buys(self, candidates, coin_balance, cash_krw):
        """
        이제는 '전액 1종목 매수' 전략:

        - 이미 코인 보유 중이면 신규 매수 전부 스킵
        - 후보들 중 ML 점수 가장 높은 1개만 선택
        - 사용 가능 KRW 거의 전부를 그 코인에 넣어서 매수
        """

        has_position = bool(coin_balance)
        has_pending = bool(self.pending_orders)
        
        # 0) 이미 보유 중이면 신규 매수 금지
        if has_position or has_pending:
            self.db.log("⏭️ [COIN 전액모드] 실제 포지션/주문 존재 → 신규 매수 스킵")
            return

        if not candidates:
            self.db.log("⏭️ [COIN 후보없음] 신규 매수 대상 없음")
            return

        # 1) ML 점수 높은 순으로 정렬 후 최상위 1개만 선택
        candidates.sort(key=lambda x: (x["ml_proba"] or 0), reverse=True)
        c = candidates[0]

        region = c["region"]
        market = c["symbol"]
        price = c["current_price"]
        ml_proba = c["ml_proba"]
        signal_id = c["signal_id"]
        strategy_name = c.get("strategy_name", "UNKNOWN")

        available_krw = cash_krw
        if available_krw <= 0:
            self.db.log(f"⚠️ [COIN잔액없음] {market} 사용 가능 KRW 없음")
            return

        # 2) 마켓 주문 가능 정보 조회
        try:
            chance = self.fetcher.get_order_chance(market)
        except Exception as e:
            self.db.log(f"❌ [COIN chance 조회 실패] {market} | {e}")
            return

        market_info = chance.get("market") or {}
        bid_info = chance.get("bid_account") or {}
        bid_constraints = market_info.get("bid", {}) or {}

        min_total = float(bid_constraints.get("min_total", "0"))
        exchange_balance = float(bid_info.get("balance", "0"))

        # 3) 실제 사용 가능한 최대 예산 계산
        raw_budget = min(available_krw, exchange_balance)
        if raw_budget <= 0:
            self.db.log(f"⚠️ [COIN잔액없음] {market} 주문가능 KRW=0")
            return

        safety_factor = 0.98
        budget = raw_budget * safety_factor

        effective_min = max(self.min_order_amount_krw, min_total)
        if budget < effective_min:
            self.db.log(
                f"⚠️ [COIN금액컷] {market} budget={budget:.0f}원 "
                f"(< effective_min={effective_min:.0f}원)"
            )
            return

        volume = budget / price
        amount = volume * price

        if volume <= 0 or amount < effective_min:
            self.db.log(
                f"⚠️ [COIN수량컷] {market} volume={volume:.6f}, amount={amount:.0f}원"
            )
            return

        # 4) 지정가 매수 시도
        order_id = self.fetcher.send_coin_order(
            market=market,
            side="bid",
            volume=volume,
            price=price,
            ord_type="limit",
        )

        if not order_id:
            self.db.log(
                f"❌ [COIN주문실패] {market} 지원 주문 방식/금액 조건 불만족, 매수 스킵"
            )
            return

        # ✅ 여기까지 왔으면 주문 성공
        available_krw -= amount

        self.trade_state[market] = {
            "entry_time": datetime.now(),
            "ml_score_entry": ml_proba,
            "ml_worst_entry": c.get("ml_worst"),
            "atr_ratio_entry": c.get("atr_ratio"),
            "strategy_name": strategy_name,
        }

        self.pending_orders[market] = {
            "order_id": order_id,
            "created_at": datetime.now(),
            "side": "bid",
        }

        trade_id = self.db.save_trade(
            region=region,
            symbol=market,
            trade_type="BUY",
            price=price,
            qty=volume,
            profit=0,
            signal_id=signal_id,
            ml_proba=ml_proba,
            entry_allowed=True,
        )

        self.db.log(
            f"✅🚀[COIN매수] {market} {volume:.6f} | ML:{ml_proba:.3f} "
            f"| 약 {amount:,.0f}원 (남은 KRW: {available_krw:,.0f})"
        )

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

    def cancel_stale_orders(self, max_wait_sec: int = 30):
        """
        1분 이상 체결 안 된 주문 취소
        - pending_orders 에 저장된 주문 기준
        - 실제로는 "진짜 체결됐는지"는 모르고, 시간 기준으로만 취소 시도
        - 이미 체결된 주문이면 거래소에서 취소 실패 리턴할 수 있지만,
          그건 로그만 찍고 넘어간다.
        """
        if not self.pending_orders:
            return

        now = datetime.now()
        to_remove = []

        for market, info in self.pending_orders.items():
            order_id = info.get("order_id")
            created_at = info.get("created_at")

            if not order_id or not created_at:
                to_remove.append(market)
                continue

            elapsed = (now - created_at).total_seconds()

            if elapsed >= max_wait_sec:
                self.db.log(
                    f"⏱️ [COIN주문취소시도] {market} order_id={order_id} "
                    f"대기시간={elapsed:.1f}초 (>{max_wait_sec}초)"
                )
                try:
                    ok = self.fetcher.cancel_order(order_id)
                except Exception as e:
                    self.db.log(f"❌ [COIN주문취소예외] {market} order_id={order_id} | {e}")
                    ok = False

                if ok:
                    self.db.log(
                        f"✅ [COIN주문취소완료] {market} order_id={order_id}"
                    )
                else:
                    self.db.log(
                        f"⚠️ [COIN주문취소실패] {market} order_id={order_id} "
                        f"(이미 체결/취소되었을 수 있음)"
                    )

                # 어쨌든 더 이상 이 주문은 관리하지 않음
                to_remove.append(market)

        for m in to_remove:
            self.pending_orders.pop(m, None)

    # ------------------------------------------------
    # 메인 체크 루프 (주식 trader.run_check 와 비슷한 구조)
    # ------------------------------------------------
    def run_check(self):
        self.cancel_stale_orders(max_wait_sec=30)
        # 1. 잔고 및 현금 조회 (코인 전용)
        try:
            coin_balance = self.fetcher.get_coin_balance()
            cash_krw = self.fetcher.get_coin_buyable_cash()
        except Exception as e:
            self.db.log(f"❌ [COIN] 잔고 불러오기 실패: {e}")
            return
        
        # ✅ 재실행 시 trades 테이블 기준으로 entry_time 복구
        self._restore_entry_state_from_db(coin_balance)
        
        holding_any = len(coin_balance) > 0  # ✅ 포지션 보유 여부

        self.db.log(
            f"💰 [COIN 잔고스냅샷] 보유코인:{len(coin_balance)} | "
            f"KRW:{cash_krw:,.0f}원"
        )

        # 엔트리/스캔 관련 변수
        entry_candidates = []
        entry_summary = []
        ml_scores = []

        count_checked = 0
        count_skipped = 0

        skip_market_closed = 0
        skip_no_price = 0
        skip_no_df = 0
        skip_short_df = 0

        # ✅ 멀티 전략 + 멀티 심볼 엔트리를 위해
        #    심볼별 df/가격을 모아두는 딕셔너리
        df_by_symbol: dict[str, pd.DataFrame] = {}
        price_by_symbol: dict[str, float] = {}
        region_by_symbol: dict[str, str] = {}

        # 2. 코인 유니버스 스캔
        for t in self.targets:
            region = t["region"]
            market = t["symbol"]   # "KRW-BTC" 형식

            if region != "CR":
                # 혹시 섞여있어도 안전하게 방어
                count_skipped += 1
                continue

            time.sleep(0.2)  # API 과부하 방지

            # ✅ 이 코인을 보유 중인지 먼저 확인
            has_coin = (market in coin_balance) or (market in self.trade_state)

            # ✅ 포지션 보유 중이면, 내가 들고 있는 코인만 감시하고 나머지는 스킵
            if holding_any and not has_coin:
                count_skipped += 1
                continue

            # (1) 시장 열렸는지 체크 (CR은 거의 항상 True)
            if not self.is_market_open(region):
                skip_market_closed += 1
                count_skipped += 1
                continue

            # (2) 현재가 조회
            price = self.fetcher.get_coin_current_price(market)
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

            from c_ml_features import SEQ_LEN  # 이미 상단 import 돼있긴 함
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

            # ✅ 포지션 보유 시: 이 코인은 '청산 감시 모드'만 수행 (기존 로직 그대로)
            if holding_any and has_coin and my_info:
                avg_price = my_info["avg_price"]
                qty = my_info["qty"]

                state = self.trade_state.get(market, {})
                entry_time = state.get("entry_time", datetime.now())
                ml_score_entry = state.get("ml_score_entry")
                ml_worst_entry = state.get("ml_worst_entry")
                atr_ratio_entry = state.get("atr_ratio_entry")
                strategy_name_entry = state.get("strategy_name")

                pos = CrPosition(
                    region=region,
                    symbol=market,
                    side="BUY",
                    qty=qty,
                    entry_price=avg_price,
                    entry_time=entry_time,
                    ml_score_entry=ml_score_entry,
                    ml_worst_entry=ml_worst_entry,
                    atr_ratio_entry=atr_ratio_entry,
                )

                now = datetime.now()
                exit_decision = decide_exit_cr(
                    pos=pos,
                    df_5m=df,
                    cur_price=price,
                    now_dt=now,
                    params=None,
                )

                if exit_decision.get("should_exit"):
                    sell_qty = qty
                    sell_type = exit_decision.get("reason", "EXIT")

                    success = self.fetcher.send_coin_order(
                        market=market,
                        side="ask",
                        volume=sell_qty,
                        price=None,
                        ord_type="market",
                    )

                    if success:
                        profit_rate = (price - avg_price) / avg_price
                        elapsed_min = (now - entry_time).total_seconds() / 60.0

                        trade_id = self.db.save_trade(
                            region=region,
                            symbol=market,
                            trade_type=sell_type,
                            price=price,
                            qty=sell_qty,
                            profit=profit_rate * 100,
                        )

                        self.db.log(
                            f"📉[COIN매도] {market}: {sell_type} {sell_qty:.6f} "
                            f"({profit_rate*100:.2f}%) | note={exit_decision.get('note','')}"
                        )

                        # ✅ 쿨다운 기록
                        self.last_exit_time[market] = now

                        if market in self.trade_state:
                            del self.trade_state[market]

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
                                "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                                "entry_strategy_name": strategy_name_entry,
                                "entry_ml_proba": float(ml_score_entry) if ml_score_entry is not None else None,
                                "entry_ml_worst": float(ml_worst_entry) if ml_worst_entry is not None else None,
                                "entry_atr_ratio": float(atr_ratio_entry) if atr_ratio_entry is not None else None,
                            }
                            comment = make_exit_comment(exit_ctx)
                            self.db.update_trade_exit_comment(trade_id, comment)
                        except Exception as e:
                            self.db.log(f"⚠️ [COIN AI청산코멘트 실패] {market} | {e}")

                # ✅ 포지션 보유 모드에서는 이 코인에 대해 엔트리 로직은 더 안 보고 다음으로
                continue

            # ✅ 여기부터는 '포지션이 없을 때'만 실행 (엔트리 스캔 모드)
            if not holding_any:
                # 재진입 쿨다운 걸린 코인은 후보에서 제외
                if self._is_in_cooldown(market):
                    count_skipped += 1
                    continue

                # 멀티 전략 평가를 위해 df/현재가/region만 모아둔다
                df_by_symbol[market] = df
                price_by_symbol[market] = price
                region_by_symbol[market] = region

        # ✅ 포지션이 없을 때만 새 엔트리 시도
        if not holding_any:
            if not df_by_symbol:
                self.db.log("⏭️ [COIN 엔트리] 유효한 스캔 대상 없음 (df_by_symbol 비어있음)")
            else:
                # 🔥 전체 코인 + 전략1/2/3 중에서 최종 1개 선택 (+ 최소 스코어 조건)
                result = pick_best_entry_across_universe(
                    df_by_symbol=df_by_symbol,
                    strategies=self.entry_strategies,
                    params_by_strategy={},             # 필요 시 전략별 파라미터 넣기
                    min_final_score=AI_PARAMS_COIN["ms_min_final_score"],
                )

                if not result.get("has_final_entry"):
                    self.db.log(f"🧊 [COIN 엔트리 없음] reason={result.get('reason')}")
                else:
                    symbol = result["symbol"]              # 예: "KRW-BTC"
                    strategy_key = result["strategy"]      # "MS" / "REV" / "MOMO"
                    entry = result["entry"]                # 해당 전략의 엔트리 결과
                    region = region_by_symbol.get(symbol, "CR")
                    price = price_by_symbol[symbol]

                    strategy_name = entry.get("strategy_name", f"CR_{strategy_key}")
                    score = float(entry.get("entry_score") or 0.0)
                    risk = entry.get("risk") or {}
                    ml_pred = entry.get("ml_pred") or {}

                    # 로그/요약용
                    ml_scores.append((symbol, score))
                    entry_summary.append(
                        {
                            "market": symbol,
                            "strategy": strategy_name,
                            "score": score,
                            "atr_ratio": risk.get("atr_ratio"),
                            "note": entry.get("note", ""),
                        }
                    )

                    # signals 테이블에 기록 (MS가 아니면 at_support/is_bullish/price_up는 False 처리)
                    at_support = bool(entry.get("at_support", False))
                    is_bullish = bool(entry.get("is_bullish", False))
                    price_up = bool(entry.get("price_up", False))

                    lookback_val = self.params.get("lookback", 20)
                    band_pct_val = self.params.get("band_pct", 0.005)       

                    signal_id = self.db.save_signal(
                        region=region,
                        symbol=symbol,
                        price=float(df_by_symbol[symbol]["close"].iloc[-1]),
                        at_support=at_support,
                        is_bullish=is_bullish,
                        price_up=price_up,
                        lookback=lookback_val,
                        band_pct=band_pct_val, 
                        has_stock=False,
                        entry_signal=True,
                        ml_proba=score,
                        entry_allowed=True,
                        note=strategy_name,
                    )

                    # ✅ execute_buys에서 요구하는 형태의 후보 1개 생성
                    entry_candidates.append(
                        {
                            "region": region,
                            "symbol": symbol,
                            "current_price": price,
                            "ml_proba": score,
                            "signal_id": signal_id,
                            "strategy_name": strategy_name,
                            "ml_worst": ml_pred.get("worst"),
                            "atr_ratio": risk.get("atr_ratio"),
                        }
                    )

            # 최종 후보(있으면 1개)를 가지고 전액 매수 로직 실행
            self.execute_buys(entry_candidates, coin_balance, cash_krw)
        else:
            self.db.log("🛡️ [COIN 전액모드] 포지션 보유 중 → 신규 매수 스킵 (청산만 감시)")

        # 4. 요약 로그 (주식 트레이더와 형식 통일)
        ml_scores.sort(key=lambda x: x[1], reverse=True)
        top_ml_str = ", ".join([f"{s}({p:.4f})" for s, p in ml_scores[:3]])

        summary_msg = (
            f"📊 [COIN 스캔완료] 대상:{count_checked} 스킵:{count_skipped} "
            f"(장마감:{skip_market_closed}, 가격없음:{skip_no_price}, "
            f"데이터없음:{skip_no_df}, 캔들부족:{skip_short_df}) "
            f"| 매수후보:{len(entry_candidates)} "
            f"| 🔥ML Top3: [{top_ml_str}]"
        )
        self.db.log(summary_msg)

        # 5. 엔트리 요약 (이번 스캔에서 '최종' 신호 뜬 코인만)
        if entry_summary:
            sorted_entries = sorted(
                entry_summary,
                key=lambda x: (x["score"] is not None, x["score"]),
                reverse=True,
            )

            N = 3
            lines = []
            for e in sorted_entries[:N]:
                m = e["market"]
                strat = e["strategy"]
                sc = e["score"]
                atr = e["atr_ratio"]
                sc_str = f"{sc*100:.2f}%" if sc is not None else "NA"
                atr_str = f"{atr*100:.2f}%" if atr is not None else "NA"
                lines.append(f"{m}:{strat} score={sc_str}, ATR={atr_str}")

            msg = "🔥 [COIN ENTRY SUMMARY] " + " | ".join(lines)
            self.db.log(msg)

