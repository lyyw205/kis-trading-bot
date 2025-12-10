# 역할 요약
#
# 바이낸스 전용 실시간 코인 트레이더 클래스
# Spot / Futures 모드 둘 다 지원 (market_type="spot" | "futures", leverage 지원)
# 엔트리/익절/손절 로직은 전부 TCN 엔트리/익절 허브 + ML 스코어에 위임

from datetime import datetime, timedelta
import math
from typing import Any, Dict, List, Optional

import pandas as pd
from bi_features import (            # ✅ 공통 Feature 정의에서 가져오게 변경
    FEATURE_COLS,
    SEQ_LENS,
    HORIZONS,
    build_multiscale_samples_cr,
    resample_from_5m,  
)
from ai_helpers import make_entry_comment, make_exit_comment
from bi_entry_hub import (
    pick_best_entry_across_universe,
    DEFAULT_ENTRY_PARAMS_MS,
)
from bi_exit_lib import CrPosition
from bi_exit_hub import decide_exit_cr
from bi_client import BinanceDataFetcher
from c_db_manager import BotDatabase


class BinanceCoinRealTimeTrader:
    """
    Binance Spot/Futures 겸용 실시간 트레이더
    """

    def __init__(
        self,
        fetcher: BinanceDataFetcher,
        targets,
        params,
        db: BotDatabase,
        model=None,
        ml_threshold: float = 0.55,
        dry_run: bool = False,
        # 🔽 추가된 설정
        market_type: str = "spot",  # "spot" or "futures"
        leverage: int = 1,          # 선물일 때 적용할 레버리지
    ):
        self.region = "BI"
        self.fetcher = fetcher
        self.targets = targets
        self.min_bars_5m = max(SEQ_LENS["5m"], max(HORIZONS) + 10)
        base_params = DEFAULT_ENTRY_PARAMS_MS.copy()
        if params:
            base_params.update(params)
        self.params = base_params
        self.exit_state_log_min_abs_pnl = 1.0  # 예: ±1% 이상일 때만 로그
        self._last_exit_state_pnl = {}
        self.db = db
        self.trade_state: dict[str, dict] = {}
        self.pending_orders: dict[str, dict] = {}
        self.model = model
        self.ml_threshold = ml_threshold
        self.dry_run = dry_run

        # 🔽 시장 구분 및 레버리지
        self.market_type = market_type.lower()
        self.leverage = leverage if self.market_type == "futures" else 1

        # 시장 레짐 관련
        self.market_regime = None
        self.market_regime_avg_ret_1d = 0.0
        self.market_regime_updated_at = None
        self._last_regime_log_time = None

        # Binance용 설정
        self.min_order_amount_usdt = 0.1
        self.max_pos = 1
        self.reentry_cooldown_min = 60
        self.last_exit_time: dict[str, datetime] = {}
        self.min_final_score = 0.08
        # 👉 포지션/잔고 로그용 직전 상태
        self._last_balance_log_state = {
            "pos": None,
            "usdt": None,
        }
        self._logged_target_scan = False
        self.db.log(
            f"🔄 [BI-COIN] Trader Initialized | Type={self.market_type.upper()} | Lev={self.leverage}x "
        )

    def _refresh_market_regime_if_needed(self):
        """30분에 한 번 시장 레짐 갱신"""
        now = datetime.now()
        if (
            self.market_regime is None or
            self.market_regime_updated_at is None or
            (now - self.market_regime_updated_at).total_seconds() > 1800
        ):
            regime = self.db.get_setting("market_regime_coin", default="NEUTRAL")
            avg_ret_str = self.db.get_setting("market_regime_coin_avg_return_1d", default="0.0")

            try:
                avg_ret = float(avg_ret_str)
            except Exception:
                avg_ret = 0.0

            self.market_regime = regime
            self.market_regime_avg_ret_1d = avg_ret
            self.market_regime_updated_at = now

    def _log_market_regime_if_needed(self):
        """5분 간격 레짐 로그"""
        now = datetime.now()
        if self._last_regime_log_time is not None:
            if (now - self._last_regime_log_time) < timedelta(minutes=5):
                return

        self._last_regime_log_time = now
        ts = now.strftime("%m-%d-%H-%M")
        regime = self.market_regime or "UNKNOWN"
        avg_pct = self.market_regime_avg_ret_1d * 100.0

        # self.db.log(
        #     f"[{ts}] 기준 시장 레짐 : \"{regime}\", 상위 150코인 평균 {avg_pct:+.2f}%"
        # )

    # ------------------------------------------------
    # 유틸
    # ------------------------------------------------
    def is_market_open(self) -> bool:
        return True

    def _get_trade_type(self, signal_side: str) -> str:
        """
        positions.trade_type 값 생성
        - spot  : 'SPOT'
        - futures LONG : 'FUTURES_LONG'
        - futures SHORT: 'FUTURES_SHORT'
        """
        if self.market_type == "spot":
            return "SPOT"
        # futures
        if signal_side.upper() == "SHORT":
            return "FUTURES_SHORT"
        return "FUTURES_LONG"
    
    def _is_in_cooldown(self, symbol: str) -> bool:
        last = self.last_exit_time.get(symbol)
        if not last:
            return False
        elapsed_min = (datetime.now() - last).total_seconds() / 60.0
        if elapsed_min < self.reentry_cooldown_min:
            return True
        return False
    
    def _get_quantity_precision(self, step_size: float) -> int:
        """
        step_size(예: 0.001)를 기반으로 소수점 자릿수(예: 3)를 반환
        """
        if step_size == 0:
            return 0  # 정보 없으면 정수로 처리 (가장 안전)
        
        s = f"{step_size:.8f}".rstrip("0")
        if "." not in s:
            return 0
        return len(s.split(".")[1])

    # ------------------------------------------------
    # 상태 복구 (positions 기준)
    # ------------------------------------------------
    def _restore_entry_state_from_db(self, coin_balance: dict):
        """
        Spot/Futures 모두 잔고 기반으로 trade_state 복원
        - positions 테이블에 status='OPEN' 인 포지션이 있으면 그 기준으로 복구
        - positions 기록 없으면:
            * Spot   : 현재가 기준 BUY 포지션 자동 복구
            * Futures: 잔고 side(LONG/SHORT) 기준으로 자동 복구
        """
        if not coin_balance:
            return
        if self.trade_state:
            return

        try:
            conn = self.db.get_connection()
            cur = conn.cursor()

            for symbol, info in coin_balance.items():

                # 이미 상태 복구된 심볼은 스킵
                if symbol in self.trade_state:
                    continue

                # --- 1) positions 에서 열린 포지션 가져오기 ---
                cur.execute(
                    """
                    SELECT 
                        id,
                        entry_time,
                        trade_type,
                        entry_price,
                        ml_proba
                    FROM positions
                    WHERE region = %s
                      AND symbol = %s
                      AND status = 'OPEN'
                    ORDER BY entry_time DESC
                    LIMIT 1
                    """,
                    (self.region, symbol),
                )
                row = cur.fetchone()

                if row:
                    (
                        pos_id,
                        raw_time,
                        trade_type,
                        db_entry_price,
                        ml_proba,
                    ) = row

                    try:
                        entry_time = pd.Timestamp(raw_time).tz_localize(None)
                    except Exception:
                        entry_time = datetime.now()

                    # side 복원
                    if self.market_type == "futures":
                        if trade_type == "FUTURES_SHORT":
                            side = "SHORT"
                        else:
                            side = "LONG"
                    else:
                        side = "BUY"

                    state = {
                        "position_id": pos_id,
                        "entry_time": entry_time,
                        "ml_score_entry": float(ml_proba) if ml_proba is not None else None,
                        "ml_worst_entry": None,
                        "atr_ratio_entry": None,
                        "side": side,
                        "strategy_name": "RESTORE_DB",
                    }

                    if self.market_type == "spot":
                        state["entry_price_override"] = float(db_entry_price or 0.0)

                    self.trade_state[symbol] = state
                    continue  # ✅ positions 기록 있으면 여기서 끝

                # --- 2) positions 기록이 없는 경우 → 기존 자동 복구 로직 ---
                if self.market_type == "spot":
                    current_price = self.fetcher.get_coin_current_price(
                        symbol, market_type="spot"
                    ) or 0.0

                    self.trade_state[symbol] = {
                        "entry_time": datetime.now(),
                        "ml_score_entry": None,
                        "ml_worst_entry": None,
                        "atr_ratio_entry": None,
                        "side": "BUY",
                        "strategy_name": "SPOT_AUTO",
                        "entry_price_override": current_price,
                    }

                    self.db.log(f"🟦 [Spot 자동 복구] {symbol} entry_price={current_price}")
                    continue

                if self.market_type == "futures":
                    side = info.get("side")
                    if not side:
                        qty = info.get("qty", 0)
                        side = "SHORT" if qty < 0 else "LONG"

                    self.trade_state[symbol] = {
                        "entry_time": datetime.now(),
                        "ml_score_entry": None,
                        "ml_worst_entry": None,
                        "atr_ratio_entry": None,
                        "side": side,
                        "strategy_name": "FUT_AUTO",
                    }

                    self.db.log(f"🟥 [Futures 자동 복구] {symbol} side={side}")
                    continue

            cur.close()
            conn.close()

        except Exception as e:
            self.db.log(f"⚠️ [BI 복구실패] {e}")

    # ------------------------------------------------
    # positions 테이블을 바이낸스 현재 상태와 동기화 (새 스키마 버전)
    # ------------------------------------------------
    def sync_positions_from_binance(self):
        """
        자동매매 시작 시점에 한 번 호출해서,
        - 현재 DB에 남아있는 OPEN 포지션을 모두 'CLOSED'로 정리하고
        - Binance 실제 열린 포지션을 그대로 새로운 OPEN 레코드로 INSERT한다.

        ⚠️ 주의:
        - 과거에 OPEN으로 남아 있던 레코드는 여기서 전부 CLOSED 처리된다.
        - 동기화 이후부터의 세션을 기준으로 positions를 맞추는 용도.
        """
        try:
            open_pos = self.fetcher.get_open_positions(market_type=self.market_type)

            if open_pos is None:
                self.db.log("⚠️ [BI] positions 동기화 실패: get_open_positions()가 None 반환")
                # 그래도 DB 쪽 OPEN 정리는 해준다.
                open_pos = {}

            # dict 형태 기대: { "BTCUSDT": {"qty":..., "entry_price":..., "side":..., "entry_time":...}, ... }
            open_pos = open_pos or {}

            conn = self.db.get_connection()
            cur = conn.cursor()

            # 1) 이 region 의 OPEN 포지션 전부 닫기
            #    (예전 is_open/closed_at 컬럼은 사용하지 않고 status 기반으로만 관리)
            cur.execute(
                """
                UPDATE positions
                SET status = 'CLOSED',
                    updated_at = now()
                WHERE region = %s
                  AND status = 'OPEN'
                """,
                (self.region,),
            )

            # 2) Binance 현재 열린 포지션들을 새로 INSERT
            inserted_cnt = 0
            source_label = f"BINANCE_{self.market_type.upper()}_SYNC"

            for symbol, p in open_pos.items():
                try:
                    qty = float(p.get("qty", 0) or 0.0)
                    entry_price = float(p.get("entry_price", 0) or 0.0)
                    if qty == 0 or entry_price == 0:
                        # 의미 없는 포지션은 건너뜀
                        continue

                    side = p.get("side")
                    # side가 없으면 기본값 추론
                    if not side:
                        # 선물: qty 부호로 LONG/SHORT 추론
                        if self.market_type == "futures":
                            side = "SHORT" if qty < 0 else "LONG"
                        else:
                            side = "BUY"

                    # trade_type 매핑
                    if self.market_type == "spot":
                        trade_type = "SPOT"
                    else:
                        trade_type = "FUTURES_SHORT" if side.upper() == "SHORT" else "FUTURES_LONG"

                    entry_time = p.get("entry_time")
                    if entry_time is None:
                        entry_time = datetime.now()

                    # numpy / timezone 붙은 것들 정리
                    try:
                        entry_time = pd.Timestamp(entry_time).tz_localize(None).to_pydatetime()
                    except Exception:
                        entry_time = datetime.now()

                    entry_notional = entry_price * abs(qty)

                    cur.execute(
                        """
                        INSERT INTO positions (
                            region,
                            symbol,
                            trade_type,
                            source,
                            entry_time,
                            entry_price,
                            entry_qty,
                            entry_notional,
                            signal_id,
                            ml_proba,
                            entry_allowed,
                            entry_comment,
                            status
                        ) VALUES (
                            %s,%s,%s,%s,
                            %s,%s,%s,%s,
                            %s,%s,%s,%s,%s
                        )
                        """,
                        (
                            self.region,
                            symbol,
                            trade_type,
                            source_label,
                            entry_time,
                            entry_price,
                            qty,
                            entry_notional,
                            None,      # signal_id 없음
                            None,      # ml_proba 없음
                            True,      # entry_allowed 기본 True
                            None,      # entry_comment 없음
                            "OPEN",
                        ),
                    )
                    inserted_cnt += 1
                except Exception as e_inner:
                    self.db.log(f"⚠️ [BI] positions 동기화 중 {symbol} INSERT 실패: {e_inner}")
                    continue

            conn.commit()
            cur.close()
            conn.close()

            self.db.log(
                f"✅ [BI] positions 동기화 완료 | Binance OPEN={len(open_pos)}개 → DB에 {inserted_cnt}개 INSERT"
            )

        except Exception as e:
            self.db.log(f"⚠️ [BI] positions 동기화 실패: {e}")

    # ------------------------------------------------
    # 심볼 필터
    # ------------------------------------------------
    def _get_binance_symbol_filters(self, market: str) -> Dict[str, float]:
        min_notional = 0.0
        min_qty = 0.0
        step_size = 0.0
        try:
            info = self.fetcher.get_order_chance(market, market_type=self.market_type) or {}
            symbols = info.get("symbols") or []
            if symbols:
                filters = symbols[0].get("filters", []) or []
                for f in filters:
                    ftype = f.get("filterType")
                    if ftype == "MIN_NOTIONAL":  # Futures
                        min_notional = float(f.get("minNotional", "0"))
                    elif ftype == "NOTIONAL":    # Spot
                        min_notional = float(f.get("minNotional", "0"))
                    elif ftype == "LOT_SIZE":
                        min_qty = float(f.get("minQty", "0"))
                        step_size = float(f.get("stepSize", "0"))
        except Exception as e:
            self.db.log(f"⚠️ [Filter Error] {market} 필터 조회 실패: {e}")
            pass
            
        return {"min_notional": min_notional, "min_qty": min_qty, "step_size": step_size}

    # ------------------------------------------------
    # 진입 집행 (Spot/Futures 통합 + Spot/Futures 모두 금액 기준 진입)
    # ------------------------------------------------
    def execute_entry(self, candidates, coin_balance, cash_usdt: float):
        has_position = bool(coin_balance)
        has_pending = bool(self.pending_orders)
        if has_position or has_pending:
            return
        if not candidates:
            return

        # 점수 높은 후보 선택
        candidates.sort(key=lambda x: (x["ml_proba"] or 0), reverse=True)
        c = candidates[0]

        signal_side = c.get("side", "BUY")
        if self.market_type == "spot" and signal_side == "SHORT":
            return

        market = c["symbol"]
        price = c["current_price"]
        ml_proba = c["ml_proba"]
        strategy_name = c.get("strategy_name", "UNKNOWN")

        if signal_side == "SHORT":
            order_side = "SELL"
            position_side = "SHORT"
        else:
            order_side = "BUY"
            position_side = "LONG"

        # Futures: 마진타입/레버리지 설정
        if self.market_type == "futures":
            try:
                self.fetcher.set_margin_type(market, "ISOLATED")
            except Exception:
                pass
            try:
                self.fetcher.set_leverage(market, self.leverage)
            except Exception:
                pass

        # 공통 필터 / 예산
        filters = self._get_binance_symbol_filters(market)
        min_notional = max(self.min_order_amount_usdt, filters.get("min_notional", 5.0))

        available_usdt = cash_usdt
        if available_usdt <= 0 or available_usdt < min_notional:
            self.db.log(f"⚠️ [BI 진입중단] {market} 가용 USDT 부족: {available_usdt:.2f}")
            return

        entry_budget = available_usdt
        leveraged_budget = entry_budget * self.leverage

        filled_qty = 0.0
        fill_price = price
        order_id = None

        # ======================
        #  A) SPOT 진입 (quoteOrderQty = 100%)
        # ======================
        if self.market_type == "spot":
            quote_amount = leveraged_budget  # 레버리지는 1이라 사실상 available_usdt
            if quote_amount < min_notional:
                self.db.log(
                    f"⚠️ [BI 진입중단] {market} Spot quote_amount({quote_amount:.4f}) < min_notional({min_notional:.4f})"
                )
                return

            try:
                order_id = self.fetcher.send_coin_order(
                    market=market,
                    side=order_side,
                    volume=None,
                    price=None,
                    ord_type="MARKET",
                    market_type="spot",
                    position_side=None,
                    reduce_only=None,
                    quote_order_qty=quote_amount,  # ✅ USDT 금액 기준
                )
            except Exception:
                order_id = None

            if not order_id:
                self.db.log(
                    f"❌ [BI 진입실패] {market} Spot {order_side} quoteOrderQty={quote_amount:.4f}"
                )
                return

            try:
                od = self.fetcher.get_order_details(
                    market=market,
                    order_id=str(order_id),
                    market_type="spot",
                )
                if od:
                    exec_qty = float(od.get("executedQty", 0) or 0)
                    avg_fill = float(od.get("avg_fill_price", 0) or 0)
                    if exec_qty > 0:
                        filled_qty = exec_qty
                    if avg_fill > 0:
                        fill_price = avg_fill
            except Exception as e:
                self.db.log(f"⚠️ [BI ENTRY] {market} 주문 상세 조회 실패, ticker 사용: {e}")

        # ======================
        #  B) FUTURES 진입 (금액 → 수량 계산 후 volume 기반 주문)
        # ======================
        else:
            # 수수료/여유 감안해서 90%만 사용
            risk_frac = 0.95
            quote_amount = leveraged_budget * risk_frac  # USDT 기준 베팅 금액
            if quote_amount < min_notional:
                self.db.log(
                    f"⚠️ [BI 진입중단] {market} Futures quote_amount({quote_amount:.4f}) < "
                    f"min_notional({min_notional:.4f})"
                )
                return

            # 현재 가격 기준으로 "코인 수량"으로 변환
            if not price or price <= 0:
                self.db.log(
                    f"⚠️ [BI 진입중단] {market} Futures price 비정상: {price}"
                )
                return

            raw_qty = quote_amount / price  # USDT → 코인 수량
            if raw_qty <= 0:
                self.db.log(
                    f"⚠️ [BI 진입중단] {market} Futures raw_qty<=0 (quote={quote_amount:.4f}, price={price})"
                )
                return

            try:
                # 🔹 선물은 항상 volume(수량) 기반 주문
                order_id = self.fetcher.send_coin_order(
                    market=market,
                    side=order_side,
                    volume=raw_qty,       # ✅ 수량으로 넘김
                    price=None,
                    ord_type="MARKET",
                    market_type="futures",
                    position_side=position_side,
                    reduce_only=None,
                    # quote_order_qty는 더 이상 사용하지 않음
                )
            except Exception:
                order_id = None

            if not order_id:
                self.db.log(
                    f"❌ [BI 최종실패] {market} Futures {position_side} "
                    f"qty={raw_qty:.6f} (quote≈{quote_amount:.2f} USDT)"
                )
                return

            try:
                od = self.fetcher.get_order_details(
                    market=market,
                    order_id=str(order_id),
                    market_type="futures",
                )
                if od:
                    exec_qty = float(od.get("executedQty", 0) or 0)
                    avg_fill = float(od.get("avg_fill_price", 0) or 0)
                    if exec_qty > 0:
                        filled_qty = exec_qty
                    if avg_fill > 0:
                        fill_price = avg_fill
            except Exception as e:
                self.db.log(f"⚠️ [BI ENTRY] {market} 선물 주문 상세 조회 실패, ticker 사용: {e}")
        # ======================
        #  C) 상태/DB 기록 (공통)
        # ======================
        now_ts = datetime.now()

        # trade_state 기본 정보
        self.trade_state[market] = {
            "entry_time": now_ts,
            "ml_score_entry": ml_proba,
            "ml_worst_entry": c.get("ml_worst"),
            "atr_ratio_entry": c.get("atr_ratio"),
            "strategy_name": strategy_name,
            "side": signal_side,
            "entry_price_override": fill_price,
        }
        self.pending_orders[market] = {
            "order_id": order_id,
            "created_at": now_ts,
            "side": order_side,
        }

        db_trade_type = "SELL" if signal_side == "SHORT" else "BUY"
        trade_id = self.db.save_trade(
            region="BI",
            symbol=market,
            trade_type=db_trade_type,
            price=fill_price,
            qty=filled_qty,
            profit=0,
            signal_id=c["signal_id"],
            ml_proba=ml_proba,
            entry_allowed=True,
        )

        # ---- positions 신규 기록 (새 스키마) ----
        try:
            conn = self.db.get_connection()
            cur = conn.cursor()

            trade_type = self._get_trade_type(signal_side)
            source = f"BI_{self.market_type.upper()}_BOT"
            entry_notional = float(fill_price) * float(filled_qty or 0.0)

            cur.execute(
                """
                INSERT INTO positions (
                    region,
                    symbol,
                    trade_type,
                    source,
                    entry_time,
                    entry_price,
                    entry_qty,
                    entry_notional,
                    signal_id,
                    ml_proba,
                    entry_allowed,
                    entry_comment,
                    status
                ) VALUES (
                    %s,%s,%s,%s,
                    %s,%s,%s,%s,
                    %s,%s,%s,%s,%s
                )
                RETURNING id
                """,
                (
                    self.region,
                    market,
                    trade_type,
                    source,
                    now_ts,
                    float(fill_price),
                    float(filled_qty or 0.0),
                    entry_notional,
                    c["signal_id"],
                    float(ml_proba) if ml_proba is not None else None,
                    True,
                    None,          # entry_comment: 필요하면 나중에 채워도 됨
                    "OPEN",
                ),
            )
            pos_id_row = cur.fetchone()
            conn.commit()
            cur.close()
            conn.close()

            if pos_id_row:
                position_id = pos_id_row[0]
                self.trade_state[market]["position_id"] = position_id

        except Exception as e:
            self.db.log(f"⚠️ [BI positions INSERT 실패] {market}: {e}")

        self.db.log(
            f"✅🚀[BI 진입] {market} {position_side} {filled_qty} (Lev {self.leverage}x)"
        )

        # 엔트리 코멘트는 trades 쪽에도, 원하면 positions.entry_comment에도 쓸 수 있음
        try:
            entry_ctx = {
                "time": now_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "region": "BI",
                "symbol": market,
                "exchange": "BINANCE",
                "market_type": self.market_type,
                "side": position_side,
                "qty": filled_qty,
                "price": float(fill_price),
                "ml_proba": ml_proba,
                "strategy": strategy_name,
            }
            self.db.update_trade_entry_comment(trade_id, make_entry_comment(entry_ctx))
        except Exception:
            pass
        
    # ------------------------------------------------
    # 주문 관리
    # ------------------------------------------------
    def cancel_stale_orders(self, max_wait_sec: int = 30):
        if not self.pending_orders:
            return
        now = datetime.now()
        to_remove = []
        for symbol, info in self.pending_orders.items():
            if (now - info["created_at"]).total_seconds() >= max_wait_sec:
                try:
                    self.fetcher.cancel_order(
                        symbol, 
                        info["order_id"], 
                        market_type=self.market_type
                    )
                    self.db.log(f"✅ [BI 주문취소] {symbol}")
                except Exception:
                    pass
                to_remove.append(symbol)
        for s in to_remove:
            self.pending_orders.pop(s, None)

    # ------------------------------------------------
    # 멀티스케일 입력 헬퍼 (실전에서도 학습과 동일 파이프라인 사용)
    # ------------------------------------------------
    def make_multiscale_inputs_for_symbol(self, symbol: str):
        """
        1) Binance에서 5m OHLCV를 가져오고
        2) 15m/30m/1h로 리샘플한 뒤
        3) build_multiscale_samples_cr()로 샘플 생성
        4) 마지막 샘플 1개만 (numpy)로 반환

        추론/엔트리 로직에서는 이 결과를 받아서
        torch 텐서로 바꾸고 모델에 넣으면 됨.
        """
        try:
            df_5m = self.fetcher.get_coin_ohlcv(
                symbol,
                "5m",
                limit=max(120, self.min_bars_5m),
                market_type=self.market_type,
            )
        except Exception as e:
            self.db.log(f"⚠️ [BI MS-INPUT] {symbol} 5m OHLCV 로드 실패: {e}")
            return None

        if df_5m is None or len(df_5m) < self.min_bars_5m:
            return None

        # 인덱스 정리
        if not isinstance(df_5m.index, pd.DatetimeIndex):
            if "dt" in df_5m.columns:
                df_5m = df_5m.copy()
                df_5m["dt"] = pd.to_datetime(df_5m["dt"])
                df_5m = df_5m.set_index("dt")
            else:
                return None

        df_5m = df_5m.sort_index()

        # 15m/30m/1h 리샘플 (공통 유틸)
        df_5m, df_15m, df_30m, df_1h = resample_from_5m(df_5m)

        try:
            X_5m, X_15m, X_30m, X_1h, Y, base_dt = build_multiscale_samples_cr(
                df_5m=df_5m,
                df_15m=df_15m,
                df_30m=df_30m,
                df_1h=df_1h,
                feature_cols=FEATURE_COLS,
                seq_lens=SEQ_LENS,
                horizons=HORIZONS,
                return_index=True,
            )
        except ValueError as e:
            self.db.log(f"⚠️ [BI MS-INPUT] {symbol} 샘플 생성 실패: {e}")
            return None

        if len(X_5m) == 0:
            return None

        # 마지막 샘플만 반환 (numpy, shape: (1, L, F))
        return {
            "x_5m": X_5m[-1:],      # (1, L5, F)
            "x_15m": X_15m[-1:],
            "x_30m": X_30m[-1:],
            "x_1h": X_1h[-1:],
            "base_dt": base_dt[-1], # 이 시점이 기준
        }

    # ------------------------------------------------
    # 메인 체크 루프
    # ------------------------------------------------
    def run_check(self):
        # 0. 매 루프마다 레짐 상태 갱신
        self._refresh_market_regime_if_needed()
        self._log_market_regime_if_needed()

        self.cancel_stale_orders(max_wait_sec=30)

        # 1. 잔고 조회
        try:
            coin_balance = self.fetcher.get_coin_balance(market_type=self.market_type)
            cash_usdt = self.fetcher.get_coin_buyable_cash(market_type=self.market_type)
        except Exception as e:
            self.db.log(f"❌ [BI] 잔고 불러오기 실패: {e}")
            return
        
        # 🔽 먼지 포지션 제거 (1 USDT 미만은 무시)
        cleaned_balance = {}
        for sym, info in coin_balance.items():
            if sym == "USDT":
                continue
            try:
                price = self.fetcher.get_coin_current_price(sym, market_type=self.market_type)
                notional = price * info["qty"]
            except Exception:
                notional = 0

            if notional >= 1.5:
                cleaned_balance[sym] = info

        coin_balance = cleaned_balance

        self._restore_entry_state_from_db(coin_balance)
        holding_any = len(coin_balance) > 0

        # ------ 포지션/잔고 로그 (변화가 있을 때만) ------
        pos_count = len(coin_balance)
        usdt_rounded = float(f"{cash_usdt:.1f}")  # 로그와 동일한 소수 1자리 기준

        prev = getattr(self, "_last_balance_log_state", None)
        should_log_balance = False

        if not prev or prev["pos"] is None:
            # 최초 1번은 무조건 찍기
            should_log_balance = True
        else:
            # 포지션 개수가 변했거나,
            # USDT가 0.1 이상 변했을 때만 로그
            if prev["pos"] != pos_count:
                should_log_balance = True
            elif abs(prev["usdt"] - usdt_rounded) >= 0.1:
                should_log_balance = True

        if should_log_balance:
            self._last_balance_log_state = {
                "pos": pos_count,
                "usdt": usdt_rounded,
            }
            self.db.log(
                f"💰 [BI {self.market_type.upper()}] 포지션:{pos_count} | USDT:{usdt_rounded:,.1f}"
            )

        entry_candidates = []
        
        df_by_symbol = {}
        price_by_symbol = {}
        region_by_symbol = {}

        # ===========================
        # 2-A. EXIT: 잔고 기준 처리
        # ===========================
        if holding_any:
            for symbol, my_info in coin_balance.items():
                try:
                    price = self.fetcher.get_coin_current_price(symbol, market_type=self.market_type)
                    if not price:
                        continue

                    df = self.fetcher.get_coin_ohlcv(
                        symbol,
                        "5m",
                        limit=max(120, self.min_bars_5m),  # ✅ limit도 공통 기준 이상으로
                        market_type=self.market_type,
                    )
                    if df is None or len(df) < self.min_bars_5m:
                        continue

                    pos_state = self.trade_state.get(symbol, {})
                    current_side = (
                        pos_state.get("side")
                        or (my_info.get("side") if my_info else None)
                        or "LONG"
                    )
                    strategy_name = pos_state.get("strategy_name")

                    # entry_price 우선순위
                    entry_pr = pos_state.get("entry_price_override")
                    if not entry_pr:
                        entry_pr = my_info.get("avg_price", 0)
                    if not entry_pr or entry_pr <= 0:
                        entry_pr = price

                    # float 강제 변환
                    try:
                        qty_val = float(my_info["qty"])
                    except Exception:
                        qty_val = float(my_info.get("qty", 0) or 0.0)

                    try:
                        entry_price_val = float(entry_pr)
                    except Exception:
                        entry_price_val = float(price)

                    ml_score_val = pos_state.get("ml_score_entry")
                    if ml_score_val is not None:
                        try:
                            ml_score_val = float(ml_score_val)
                        except Exception:
                            ml_score_val = None

                    ml_worst_val = pos_state.get("ml_worst_entry")
                    if ml_worst_val is not None:
                        try:
                            ml_worst_val = float(ml_worst_val)
                        except Exception:
                            ml_worst_val = None

                    atr_ratio_val = pos_state.get("atr_ratio_entry")
                    if atr_ratio_val is not None:
                        try:
                            atr_ratio_val = float(atr_ratio_val)
                        except Exception:
                            atr_ratio_val = None

                    pos = CrPosition(
                        region="BI",
                        symbol=symbol,
                        side=current_side,
                        qty=qty_val,
                        entry_price=entry_price_val,
                        entry_time=pos_state.get("entry_time", datetime.now()),
                        ml_score_entry=ml_score_val,
                        ml_worst_entry=ml_worst_val,
                        atr_ratio_entry=atr_ratio_val,
                    )

                    # 디버그용 현재 상태
                    try:
                        dbg_entry = float(pos.entry_price or 0.0)
                    except Exception:
                        dbg_entry = 0.0

                    try:
                        dbg_price = float(price or 0.0)
                    except Exception:
                        dbg_price = 0.0

                    try:
                        dbg_qty = float(my_info["qty"])
                    except Exception:
                        dbg_qty = 0.0

                    pnl_pct_dbg = 0.0
                    if dbg_entry > 0 and dbg_qty > 0:
                        if current_side == "SHORT":
                            pnl_pct_dbg = (dbg_entry - dbg_price) / dbg_entry * 100.0
                        else:
                            pnl_pct_dbg = (dbg_price - dbg_entry) / dbg_entry * 100.0

                    if not hasattr(self, "_last_exit_state_pnl"):
                        self._last_exit_state_pnl = {}

                    prev_pnl = self._last_exit_state_pnl.get(symbol)
                    threshold = getattr(self, "exit_state_log_min_abs_pnl", 1.0)

                    # prev_pnl이 없는 첫 실행은 저장만 하고 로그는 찍지 않음
                    if prev_pnl is None:
                        self._last_exit_state_pnl[symbol] = pnl_pct_dbg
                    else:
                        # 이전 대비 변동폭 계산
                        if abs(pnl_pct_dbg - prev_pnl) >= threshold:
                            self.db.log(
                                f"🔎 [BI EXIT-STATE] 수익률={pnl_pct_dbg:+.2f}% "
                            )
                            # 새로운 기준으로 업데이트
                            self._last_exit_state_pnl[symbol] = pnl_pct_dbg

                    res = decide_exit_cr(pos, df, price, datetime.now(), strategy_name)
                    
                    try:
                        timeout_left = res.get("timeout_left_min")
                        held_bars = res.get("held_bars")
                        max_hold_bars = res.get("max_hold_bars")
                        trailing_active = res.get("trailing_active")
                        trailing_level = res.get("trailing_level")

                        if timeout_left is not None:
                            self.db.log(
                                f"⏳ [BI EXIT-CHECK] {symbol} side={current_side} "
                                f"held={held_bars}/{max_hold_bars} bars, "
                                f"timeout까지 {timeout_left:.1f}분, "
                                f"trailing_active={trailing_active}, "
                                f"trailing_level={trailing_level}"
                            )
                    except Exception as e:
                        self.db.log(f"⚠️ [BI EXIT 디버그 로그 실패] {symbol}: {e}")

                    # 실제 청산 여부 판단
                    if not res.get("should_exit"):
                        continue

                    # -----------------------------------------
                    # 0) Binance 필터 조회 + 수량 정밀도 보정
                    # -----------------------------------------
                    raw_qty_obj = my_info.get("qty", 0) or 0.0
                    try:
                        pos_qty = float(raw_qty_obj)
                    except Exception as e:
                        self.db.log(f"⚠️ [BI EXIT 중단] {symbol} qty 변환 실패: {raw_qty_obj} err={e}")
                        continue

                    if pos_qty <= 0:
                        self.db.log(f"⚠️ [BI EXIT 중단] {symbol} pos_qty<=0 ({pos_qty})")
                        continue

                    filters = self._get_binance_symbol_filters(symbol)
                    step_size = filters.get("step_size", 0.0)
                    if step_size <= 0:
                        step_size = 1.0  # 정보 없으면 정수로만

                    step_prec = self._get_quantity_precision(step_size)

                    if self.market_type == "futures":
                        target_qty = abs(pos_qty)
                    else:
                        target_qty = abs(pos_qty)

                    max_prec = step_prec

                    if target_qty <= 0:
                        self.db.log(
                            f"⚠️ [BI EXIT 중단] {symbol} target_qty<=0 ({target_qty})"
                        )
                        continue

                    scale = 10 ** max_prec
                    qty_scaled = math.floor(target_qty * scale) / scale

                    qty_floored = math.floor(qty_scaled / step_size) * step_size

                    if max_prec == 0:
                        close_qty = int(qty_floored)
                    else:
                        close_qty = float(f"{qty_floored:.{max_prec}f}")

                    if close_qty <= 0:
                        self.db.log(
                            f"⚠️ [BI EXIT 중단] {symbol} target_qty={target_qty} → "
                            f"step_size={step_size}, max_prec={max_prec} 적용 후 0 이하여서 스킵"
                        )
                        continue

                    exit_reason = res.get("reason", "EXIT")

                    if current_side == "SHORT":
                        close_side = "BUY"
                        position_side = "SHORT"
                    else:
                        close_side = "SELL"
                        position_side = "LONG"

                    # -----------------------------------------
                    # 1) 실제 주문 전송 + precision 에러 대비 재시도
                    # -----------------------------------------
                    order_id = None

                    if self.dry_run:
                        self.db.log(
                            f"🔍 [BI DRY-RUN 청산] {symbol} {close_side}({position_side}) "
                            f"qty={close_qty} reason={exit_reason}"
                        )
                        success = True
                    else:
                        success = False
                        try:
                            order_id = self.fetcher.send_coin_order(
                                market=symbol,
                                side=close_side,
                                volume=close_qty,
                                price=None,
                                ord_type="MARKET",
                                market_type=self.market_type,
                                position_side=position_side if self.market_type == "futures" else None,
                                reduce_only=True if self.market_type == "futures" else None,
                            )
                            success = bool(order_id)
                        except Exception as e:
                            self.db.log(
                                f"❌ [BI 청산 주문 실패 1차] {symbol} qty={close_qty} err={e}"
                            )
                            success = False

                        if (not success) and isinstance(close_qty, float) and (close_qty != int(close_qty)):
                            retry_qty = int(close_qty)
                            if retry_qty > 0:
                                self.db.log(
                                    f"⚠️ [BI EXIT 재시도] {symbol} "
                                    f"1차 qty={close_qty} 실패 → 정수({retry_qty})로 재시도"
                                )
                                try:
                                    order_id = self.fetcher.send_coin_order(
                                        market=symbol,
                                        side=close_side,
                                        volume=retry_qty,
                                        price=None,
                                        ord_type="MARKET",
                                        market_type=self.market_type,
                                        position_side=position_side if self.market_type == "futures" else None,
                                        reduce_only=True if self.market_type == "futures" else None,
                                    )
                                    success = bool(order_id)
                                    if success:
                                        close_qty = retry_qty
                                except Exception as e:
                                    self.db.log(
                                        f"❌ [BI 청산 주문 실패 2차] {symbol} qty={retry_qty} err={e}"
                                    )
                                    success = False

                    # -----------------------------------------
                    # 2) 주문 성공 시 Binance 체결 정보로 PnL 계산
                    # -----------------------------------------
                    if success:
                        filled_qty = close_qty
                        exit_price = float(price)

                        if not self.dry_run and order_id:
                            try:
                                od = self.fetcher.get_order_details(
                                    market=symbol,
                                    order_id=str(order_id),
                                    market_type=self.market_type,
                                )
                                if od:
                                    exec_qty = float(od.get("executedQty", 0) or 0)
                                    avg_fill = float(od.get("avg_fill_price", 0) or 0)
                                    if exec_qty > 0:
                                        filled_qty = exec_qty
                                    if avg_fill > 0:
                                        exit_price = avg_fill
                            except Exception as e:
                                self.db.log(
                                    f"⚠️ [BI EXIT] {symbol} 주문 상세 조회 실패, ticker 가격 사용: {e}"
                                )

                        try:
                            entry_pr = float(pos.entry_price or 0.0)
                        except Exception:
                            entry_pr = 0.0

                        try:
                            qty = float(filled_qty)
                        except Exception:
                            qty = 0.0

                        if entry_pr <= 0 or qty <= 0:
                            profit_rate = 0.0
                            pnl_usdt = 0.0
                            self.db.log(
                                f"⚠️ [BI EXIT] {symbol} 수익 계산 불가 "
                                f"(entry_pr={entry_pr}, price={exit_price}, qty={qty}) → 0으로 처리"
                            )
                        else:
                            if current_side == "SHORT":
                                profit_rate = (entry_pr - exit_price) / entry_pr
                                pnl_usdt = (entry_pr - exit_price) * qty
                            else:
                                profit_rate = (exit_price - entry_pr) / entry_pr
                                pnl_usdt = (exit_price - entry_pr) * qty

                        trade_id = self.db.save_trade(
                            region="BI",
                            symbol=symbol,
                            trade_type=close_side,
                            price=exit_price,
                            qty=qty,
                            profit=profit_rate * 100,
                        )
                        self.db.log(
                            f"📉[BI 청산] {symbol} side={current_side} qty={qty} "
                            f"({profit_rate*100:.2f}%) {exit_reason}"
                        )

                        # positions 업데이트 (새 스키마)
                        try:
                            conn = self.db.get_connection()
                            cur = conn.cursor()

                            pos_state = self.trade_state.get(symbol, {})
                            position_id = pos_state.get("position_id")

                            # holding_seconds 계산
                            entry_time = pos_state.get("entry_time", datetime.now())
                            try:
                                holding_seconds = int(
                                    (datetime.now() - entry_time).total_seconds()
                                )
                            except Exception:
                                holding_seconds = None

                            # bars_held는 exit 로직 결과(res)에 있으면 사용
                            bars_held = res.get("held_bars")

                            exit_notional = float(exit_price) * float(qty)

                            if position_id:
                                # id 기준으로 안전하게 업데이트
                                cur.execute(
                                    """
                                    UPDATE positions
                                    SET 
                                        exit_time = %s,
                                        exit_price = %s,
                                        exit_qty = %s,
                                        exit_notional = %s,
                                        pnl_usdt = %s,
                                        pnl_pct = %s,
                                        holding_seconds = %s,
                                        bars_held = %s,
                                        status = 'CLOSED',
                                        updated_at = now()
                                    WHERE id = %s
                                    """,
                                    (
                                        datetime.now(),
                                        float(exit_price),
                                        float(qty),
                                        exit_notional,
                                        pnl_usdt,
                                        profit_rate * 100.0,   # 퍼센트 기준
                                        holding_seconds,
                                        bars_held,
                                        position_id,
                                    ),
                                )
                            else:
                                # position_id 없으면 심볼/OPEN 기준으로 백업 업데이트
                                cur.execute(
                                    """
                                    UPDATE positions
                                    SET 
                                        exit_time = %s,
                                        exit_price = %s,
                                        exit_qty = %s,
                                        exit_notional = %s,
                                        pnl_usdt = %s,
                                        pnl_pct = %s,
                                        holding_seconds = %s,
                                        bars_held = %s,
                                        status = 'CLOSED',
                                        updated_at = now()
                                    WHERE region = %s
                                      AND symbol = %s
                                      AND status = 'OPEN'
                                    """,
                                    (
                                        datetime.now(),
                                        float(exit_price),
                                        float(qty),
                                        exit_notional,
                                        pnl_usdt,
                                        profit_rate * 100.0,
                                        holding_seconds,
                                        bars_held,
                                        self.region,
                                        symbol,
                                    ),
                                )

                            conn.commit()
                            cur.close()
                            conn.close()
                        except Exception as e:
                            self.db.log(f"⚠️ [BI positions UPDATE 실패] {symbol}: {e}")

                        self.last_exit_time[symbol] = datetime.now()
                        self.trade_state.pop(symbol, None)

                        try:
                            exit_ctx = {
                                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "region": "BI",
                                "symbol": symbol,
                                "exchange": "BINANCE",
                                "market_type": self.market_type,
                                "side": position_side,
                                "qty": qty,
                                "avg_entry": float(entry_pr),
                                "exit_price": float(exit_price),
                                "pnl_pct": profit_rate * 100,
                                "reason": exit_reason,
                            }
                            self.db.update_trade_exit_comment(trade_id, make_exit_comment(exit_ctx))
                        except Exception:
                            pass

                except Exception as e:
                    self.db.log(f"⚠️ [BI EXIT 스킵] {symbol} 처리 중 에러: {e}")
                    continue

        scan_total = 0
        scan_holding_only = 0
        scan_for_entry = 0

        # 2-B. 유니버스 스캔 (엔트리 후보 스캔)
        for t in self.targets:
            try:
                region = t["region"]
                symbol = t["symbol"]
                market = t.get("market", "spot").lower()

                if region != "BI":
                    continue

                if market != self.market_type:
                    continue

                scan_total += 1

                has_coin = (symbol in coin_balance) or (symbol in self.trade_state)
                if holding_any and not has_coin:
                    scan_holding_only += 1
                    continue

                if not self.is_market_open():
                    continue

                price = self.fetcher.get_coin_current_price(symbol, market_type=self.market_type)
                if not price:
                    continue

                df = self.fetcher.get_coin_ohlcv(
                    symbol,
                    "5m",
                    limit=max(120, self.min_bars_5m),  # ✅ limit도 공통 기준 이상으로
                    market_type=self.market_type,
                )
                if df is None or len(df) < self.min_bars_5m:
                    continue

                if not holding_any and not self._is_in_cooldown(symbol):
                    scan_for_entry += 1
                    df_by_symbol[symbol] = df
                    price_by_symbol[symbol] = price
                    region_by_symbol[symbol] = region

            except Exception as e:
                err_symbol = locals().get("symbol", "UNKNOWN")
                self.db.log(f"⚠️ [BI 스캔 건너뜀] {err_symbol} 처리 중 에러: {e}")
                continue

        # 5. 엔트리 결정 (Hub)
        if not holding_any and df_by_symbol:

            if not self._logged_target_scan:
                self._logged_target_scan = True
                mt_label = "Spot" if self.market_type == "spot" else "Futures"
                self.db.log(
                    f"🎯 [BI 타겟 스캔] {mt_label} {scan_total} 종목"
                )

            result = pick_best_entry_across_universe(
                df_by_symbol=df_by_symbol,
                strategies=None,
                params_by_strategy={},
                min_final_score=self.min_final_score,
                market_regime=self.market_regime,
                per_strategy_min_score={
                    "MS": 0.015,
                    "MS_SHORT": 0.018,
                },
            )

            if result.get("has_final_entry"):
                sym = result["symbol"]
                entry = result["entry"]
                
                score = float(entry.get("final_score") or entry.get("entry_score") or 0.0)
                strategy_name = result["strategy"]
                side = entry.get("side", "BUY")

                entry_candidates.append({
                    "region": region_by_symbol.get(sym, "BI"),
                    "symbol": sym,
                    "current_price": price_by_symbol[sym],
                    "ml_proba": score,
                    "signal_id": 0,
                    "strategy_name": strategy_name,
                    "side": side,
                    "ml_worst": entry.get("ml_pred", {}).get("worst"),
                    "atr_ratio": entry.get("risk", {}).get("atr_ratio"),
                })
                
                self.db.log(f"🔭 [BI 발견] {sym} ({strategy_name}) Score:{score:.4f} Side:{side}")
                sel_reason = result.get("selection_reason")
                if sel_reason:
                    self.db.log(f"🧠 [BI 엔트리 선택 이유] {sel_reason}")

            self.execute_entry(entry_candidates, coin_balance, cash_usdt)
