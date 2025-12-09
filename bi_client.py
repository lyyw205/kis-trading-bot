import os
import time
import hmac
import hashlib
import math  # 🔹 precision 처리용
from typing import Dict, Any, Optional, List
from datetime import datetime

import requests
import pandas as pd


class BinanceDataFetcher:
    """
    Binance Spot 전용 데이터/주문 래퍼 Class
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        mode: str = "real",
        logger=None,
    ):
        """
        mode:
          - "real": api.binance.com (Spot 실제 서버)
          - "test": testnet (원하면 향후 분기 추가)
        """
        self.api_key = api_key or os.getenv("BINANCE_API_KEY", "")
        self.secret_key = (secret_key or os.getenv("BINANCE_SECRET_KEY", "")).encode()
        self.mode = mode
        self.logger = logger or print

        # base URL 세트
        if mode == "real":
            self.spot_base_url = "https://api.binance.com"
            self.fut_base_url = "https://fapi.binance.com"  # USDT-M Futures
        else:
            # 필요하면 나중에 testnet 주소로 수정
            self.spot_base_url = "https://testnet.binance.vision"
            self.fut_base_url = "https://testnet.binancefuture.com"

    # --------------------------------------------------
    # 공통 유틸
    # --------------------------------------------------
    def log(self, msg: str):
        try:
            self.logger(msg)
        except Exception:
            print(msg)

    def _get_base(self, market_type: str = "spot") -> str:
        """
        market_type에 따라 Base URL 반환
        """
        if market_type == "futures":
            return self.fut_base_url
        return self.spot_base_url

    def _sign_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Binance 인증: Query String + HMAC-SHA256 서명
        - params에 timestamp를 자동으로 넣고, signature를 추가해서 반환
        - ❗ 사인에 사용한 파라미터 순서와 실제 전송 순서를 동일하게 맞춘다.
        """
        if not self.api_key or not self.secret_key:
            raise ValueError("Binance API Key / Secret이 설정되어 있지 않습니다.")

        # 원본 복사 + timestamp 추가
        tmp = dict(params)
        tmp["timestamp"] = int(time.time() * 1000)

        # 1) 정렬된 순서로 쿼리스트링 생성
        items = sorted(tmp.items())  # [(key, value), ...] 정렬
        query = "&".join(f"{k}={v}" for k, v in items)

        # 2) 서명 생성
        signature = hmac.new(
            self.secret_key,
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        # 3) 서버에 전송할 파라미터도 "정렬된 순서 그대로" 만들기
        signed_params = {k: v for k, v in items}  # 정렬된 순서 그대로 dict 생성
        signed_params["signature"] = signature    # 마지막에 signature 추가

        return signed_params

    def _get_auth_headers(self) -> Dict[str, str]:
        return {
            "X-MBX-APIKEY": self.api_key,
        }

    # --------------------------------------------------
    # Precision / Filters 관련 헬퍼
    # --------------------------------------------------
    def _floor_to_decimals(self, value: float, decimals: int) -> float:
        """
        value를 지정된 소수 자리수까지 '내림'하는 함수
        decimals=3 → 소수 셋째 자리까지
        decimals=0 → 정수까지
        """
        if decimals < 0:
            return math.floor(value)
        factor = 10 ** decimals
        return math.floor(value * factor) / factor

    def _floor_to_step(self, value: float, step: float) -> float:
        """
        stepSize/tickSize에 맞게 value를 아래 방향으로 스냅
        """
        if step <= 0:
            return value
        return math.floor(value / step) * step

    def _get_symbol_filters(
        self,
        symbol: str,
        market_type: str = "spot",
    ) -> Dict[str, float]:
        """
        LOT_SIZE / PRICE_FILTER / MIN_NOTIONAL 등 파싱
        - stepSize_str / tickSize_str 도 같이 반환해서 소수 자릿수 계산에 사용
        - Futures에서는 MARKET_LOT_SIZE도 같이 파싱
        """
        info = self.get_order_chance(symbol, market_type=market_type)
        result: Dict[str, float] = {}

        try:
            symbols = info.get("symbols", [])
            if not symbols:
                return result

            sym = symbols[0]

            # quantityPrecision 있으면 같이 저장
            qty_prec = sym.get("quantityPrecision")
            if qty_prec is not None:
                result["quantityPrecision"] = int(qty_prec)

            f_list = sym.get("filters", [])
            for f in f_list:
                f_type = f.get("filterType")

                if f_type == "LOT_SIZE":
                    step_str = f.get("stepSize", "0")
                    step = float(step_str)
                    if step > 0:
                        result["stepSize"] = step
                        result["stepSize_str"] = step_str

                    min_qty = float(f.get("minQty", 0) or 0)
                    max_qty = float(f.get("maxQty", 0) or 0)
                    if min_qty > 0:
                        result["minQty"] = min_qty
                    if max_qty > 0:
                        result["maxQty"] = max_qty

                # 🔥 Futures MARKET 주문에 쓰이는 MARKET_LOT_SIZE 추가
                elif f_type == "MARKET_LOT_SIZE":
                    m_step_str = f.get("stepSize", "0")
                    m_step = float(m_step_str)
                    if m_step > 0:
                        result["marketStepSize"] = m_step
                        result["marketStepSize_str"] = m_step_str

                    m_min_qty = float(f.get("minQty", 0) or 0)
                    m_max_qty = float(f.get("maxQty", 0) or 0)
                    if m_min_qty > 0:
                        result["marketMinQty"] = m_min_qty
                    if m_max_qty > 0:
                        result["marketMaxQty"] = m_max_qty

                elif f_type == "PRICE_FILTER":
                    tick_str = f.get("tickSize", "0")
                    tick = float(tick_str)
                    if tick > 0:
                        result["tickSize"] = tick
                        result["tickSize_str"] = tick_str

                elif f_type in ("MIN_NOTIONAL", "NOTIONAL"):
                    min_notional = float(f.get("minNotional", 0) or 0)
                    if min_notional > 0:
                        result["minNotional"] = min_notional

        except Exception as e:
            self.log(f"⚠️ [Binance {market_type} 필터 파싱 실패] symbol={symbol} | {e}")

        return result
    
    def _normalize_volume(
        self,
        market: str,
        volume: float,
        market_type: str = "spot",
        use_market_filters: bool = False,
    ) -> float:
        """
        모든 주문 수량을 Binance 허용 범위에 강제 정규화하는 함수.
        - LOT_SIZE.stepSize / minQty / maxQty
        - (Futures MARKET일 경우) MARKET_LOT_SIZE.stepSize / minQty / maxQty
        - quantityPrecision
        - floor() 기반으로 항상 허용 범위 내로 깎음
        """
        try:
            filters = self._get_symbol_filters(market, market_type=market_type)
        except Exception:
            filters = {}

        # 기본 LOT_SIZE 값
        step_size      = float(filters.get("stepSize", 0) or 0)
        min_qty        = float(filters.get("minQty", 0) or 0)
        max_qty        = float(filters.get("maxQty", 0) or 0)
        step_size_str  = filters.get("stepSize_str", "")

        # 🔥 Futures MARKET 주문이면 MARKET_LOT_SIZE 기준 사용
        if use_market_filters:
            m_step = float(filters.get("marketStepSize", 0) or 0)
            if m_step > 0:
                step_size = m_step
                step_size_str = filters.get("marketStepSize_str", step_size_str)

            m_min = float(filters.get("marketMinQty", 0) or 0)
            if m_min > 0:
                min_qty = m_min

            m_max = float(filters.get("marketMaxQty", 0) or 0)
            if m_max > 0:
                max_qty = m_max

        qty_prec_raw = filters.get("quantityPrecision")
        qty_prec = int(qty_prec_raw) if qty_prec_raw is not None else 0

        def _calc_decimals(s: str) -> int:
            if not s or "." not in s:
                return 0
            return len(s.split(".")[1].rstrip("0"))

        step_decimals = _calc_decimals(step_size_str)

        # ✅ precision 규칙: 기본은 stepSize 기준, quantityPrecision이 있으면 그 이상은 안 쓰게 clamp
        if step_decimals > 0:
            decimals = step_decimals
            if qty_prec > 0:
                decimals = min(decimals, qty_prec)  # 너무 많이 쓰지 않게 최소값
        else:
            decimals = qty_prec

        v = float(volume)

        # 1) minQty 적용: 너무 작으면 minQty로 올림 (안전하게 진입)
        if min_qty > 0 and v < min_qty:
            v = min_qty

        # 2) stepSize 기준 floor
        if step_size > 0:
            v = math.floor(v / step_size) * step_size

        # 3) 소수 자릿수 제한 (floor 느낌으로 잘라내기)
        if decimals is not None and decimals >= 0:
            if decimals == 0:
                v = math.floor(v)
            else:
                factor = 10**decimals
                v = math.floor(v * factor) / factor

        # 4) maxQty 제한
        if max_qty > 0 and v > max_qty:
            v = max_qty

        # 음수/0 방지
        if v <= 0:
            return 0.0

        return v

    # ============================================================
    # 1. 자산 및 잔고 조회
    # ============================================================
    def _get_spot_trades(self, symbol: str, limit: int = 1000) -> list[dict]:
        """
        Spot 트레이드 히스토리 조회 (BUY/SELL fills)
        GET /api/v3/myTrades (SIGNED)
        """
        base = self._get_base("spot")
        url = f"{base}/api/v3/myTrades"
        headers = self._get_auth_headers()

        params = self._sign_params({
            "symbol": symbol,
            "limit": limit,
        })

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=5)
            data = resp.json()
            if resp.status_code == 200 and isinstance(data, list):
                return data
            else:
                self.log(f"⚠️ [Binance spot myTrades 실패] symbol={symbol} | {resp.status_code} | {data}")
                return []
        except Exception as e:
            self.log(f"❌ [Binance spot myTrades 예외] symbol={symbol} | {e}")
            return []
        
    def _build_spot_position_from_trades(self, symbol: str, trades: list[dict]) -> Optional[dict]:
        """
        myTrades 리스트를 가지고 '현재 열려있는 포지션'만 복원한다.
        - 여러 번 매수/매도/청산이 섞여 있어도
        마지막으로 포지션이 0이 된 이후 구간만 현재 포지션으로 본다.
        - LONG(매수) 포지션만 고려 (Spot 기준)
        """
        if not trades:
            return None

        # 시간순 정렬
        trades_sorted = sorted(trades, key=lambda t: t.get("time", 0))

        pos_qty = 0.0
        segment_trades: list[dict] = []

        for t in trades_sorted:
            qty = float(t.get("qty", 0))
            is_buyer = bool(t.get("isBuyer"))
            signed_qty = qty if is_buyer else -qty

            pos_qty += signed_qty
            segment_trades.append({**t, "signed_qty": signed_qty})

            # 포지션이 0으로 돌아왔으면, 이전 segment는 닫힌 포지션으로 보고 리셋
            if abs(pos_qty) < 1e-12:
                segment_trades = []

        # 루프 끝났는데 segment_trades가 비어있으면 → 현재 열린 포지션 없음
        if not segment_trades:
            return None

        # 현재 열려 있는 segment 에서 BUY 체결만 모아서 평균 매수가 계산
        buy_qty = 0.0
        buy_quote = 0.0
        first_buy_time = None

        for t in segment_trades:
            if not bool(t.get("isBuyer")):
                continue
            qty = float(t.get("qty", 0))
            price = float(t.get("price", 0))
            buy_qty += qty
            buy_quote += qty * price
            t_time = t.get("time")
            if t_time is not None:
                if first_buy_time is None or t_time < first_buy_time:
                    first_buy_time = t_time

        if buy_qty <= 0:
            return None

        entry_price = buy_quote / buy_qty
        entry_dt = (
            datetime.fromtimestamp(first_buy_time / 1000.0)
            if first_buy_time is not None
            else datetime.now()
        )

        # qty 는 실제 잔고와 미세하게 다를 수 있지만,
        # 여기서는 "트레이드 기반 평균 매수가/참고 수량"을 priority로 둔다.
        return {
            "symbol": symbol,
            "qty": buy_qty,
            "entry_price": entry_price,
            "side": "BUY",
            "leverage": 1,
            "pnl": None,
            "roi": None,
            "entry_time": entry_dt,
        }
    
    def _get_futures_trades(self, symbol: str, limit: int = 1000) -> list[dict]:
        """
        Futures 트레이드 히스토리 조회
        GET /fapi/v1/userTrades (SIGNED)
        """
        base = self._get_base("futures")
        url = f"{base}/fapi/v1/userTrades"
        headers = self._get_auth_headers()

        params = self._sign_params({
            "symbol": symbol,
            "limit": limit,
        })

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=5)
            data = resp.json()
            if resp.status_code == 200 and isinstance(data, list):
                return data
            else:
                self.log(f"⚠️ [Binance futures userTrades 실패] symbol={symbol} | {resp.status_code} | {data}")
                return []
        except Exception as e:
            self.log(f"❌ [Binance futures userTrades 예외] symbol={symbol} | {e}")
            return []
        
    def _build_futures_position_from_trades(
        self,
        symbol: str,
        trades: list[dict],
        target_pos_amt: float,
    ) -> Optional[dict]:
        """
        futures userTrades 리스트를 가지고 '현재 열려있는 포지션' 구간만 복원한다.
        - target_pos_amt: /fapi/v2/account 의 positionAmt (롱=양수, 숏=음수)
        - 마지막으로 net position 이 0 이 된 시점 이후 segment 만 현재 포지션으로 간주
        """
        if not trades or abs(target_pos_amt) < 1e-12:
            return None

        trades_sorted = sorted(trades, key=lambda t: t.get("time", 0))

        net = 0.0
        segment_trades: list[dict] = []

        for t in trades_sorted:
            qty = float(t.get("qty", 0))
            side = t.get("side", "").upper()  # "BUY" / "SELL"
            if side == "BUY":
                signed_qty = qty
            elif side == "SELL":
                signed_qty = -qty
            else:
                continue

            net += signed_qty
            segment_trades.append({**t, "signed_qty": signed_qty})

            # 포지션이 0 으로 돌아오면 이전 구간은 닫힌 포지션 → 리셋
            if abs(net) < 1e-12:
                segment_trades = []

        # segment_trades 가 비어 있으면 현재 열린 포지션 없음
        if not segment_trades:
            return None

        # net 과 target_pos_amt 가 크게 다르면 이상치로 보고 무시
        if abs(net - target_pos_amt) > max(1e-8, abs(target_pos_amt) * 0.001):
            # (정확히 맞추기 힘든 경우가 있어도, 너무 다르면 사용 안 함)
            return None

        # 롱/숏 기준으로 진입 레그만 모아 평균 매수가 계산
        side = "LONG" if target_pos_amt > 0 else "SHORT"
        signed_target = target_pos_amt

        entry_qty = 0.0
        entry_quote = 0.0
        first_time = None

        for t in segment_trades:
            s = t.get("side", "").upper()
            qty = float(t.get("qty", 0))
            price = float(t.get("price", 0))
            t_time = t.get("time")

            if side == "LONG" and s == "BUY":
                entry_qty += qty
                entry_quote += qty * price
            elif side == "SHORT" and s == "SELL":
                entry_qty += qty
                entry_quote += qty * price

            if t_time is not None:
                if first_time is None or t_time < first_time:
                    first_time = t_time

        if entry_qty <= 0:
            return None

        entry_price = entry_quote / entry_qty
        entry_dt = (
            datetime.fromtimestamp(first_time / 1000.0)
            if first_time is not None
            else datetime.now()
        )

        return {
            "symbol": symbol,
            "side": side,
            "entry_price": entry_price,
            "entry_time": entry_dt,
        }
    
    def get_coin_balance(self, market_type: str = "spot") -> Dict[str, Dict[str, float]]:
        base = self._get_base(market_type)
        headers = self._get_auth_headers()

        if market_type == "futures":
            url = f"{base}/fapi/v2/account"
        else:
            url = f"{base}/api/v3/account"

        try:
            params = self._sign_params({})
            resp = requests.get(url, headers=headers, params=params, timeout=5)

            # 🔍 여기에서 HTTP 에러 바디까지 같이 로깅
            if resp.status_code != 200:
                try:
                    self.log(
                        f"❌ [Binance {market_type} 잔고 조회 실패] "
                        f"status={resp.status_code} | body={resp.text}"
                    )
                finally:
                    resp.raise_for_status()

            data = resp.json()

        except requests.HTTPError as e:
            # 위에서 이미 body 찍었으니 메시지만 한 번 더 남기고 종료
            self.log(f"❌ [Binance {market_type} 잔고 HTTPError] {e}")
            return {}

        except Exception as e:
            self.log(f"❌ [Binance {market_type} 잔고 조회 예외] {e}")
            return {}

        balances: Dict[str, Dict[str, float]] = {}

        if market_type == "spot":
            # Binance Spot에 없는 자산(거래쌍이 없는 코인) 리스트
            unsupported_assets = {
                "ETHW", "LUNC", "USTC", "LUNA", "LUNA2", "BTTOLD", "BCHSV"
            }

            for b in data.get("balances", []):
                asset = b.get("asset")
                free = float(b.get("free", 0))
                locked = float(b.get("locked", 0))
                qty = free + locked

                if qty <= 0:
                    continue
                
                # USDT 등 quote 자산 제외
                if asset in ("USDT", "BUSD", "USDC"):
                    continue

                # ❌ Binance Spot에 거래쌍이 없는 자산 제외
                if asset in unsupported_assets:
                    continue

                # 정상 자산만 심볼 생성
                symbol = f"{asset}USDT"
                balances[symbol] = {"qty": qty, "avg_price": 0.0}

        else:
            # 🔹 Futures: account.positions 에서 실제 포지션 가져오기
            for p in data.get("positions", []):
                symbol = p.get("symbol")
                if not symbol:
                    continue

                # positionAmt: 롱은 양수, 숏은 음수 (one-way 모드 기준)
                amt = float(p.get("positionAmt", 0))
                if amt == 0.0:
                    continue  # 포지션 없으면 스킵

                entry_price = float(p.get("entryPrice", 0) or 0.0)

                # side 정보까지 같이 내려줌 (롱/숏 자동 판별)
                side = "SHORT" if amt < 0 else "LONG"
                qty = abs(amt)

                balances[symbol] = {
                    "qty": qty,
                    "avg_price": entry_price,
                    "side": side,
                }

            # (선물 지갑 USDT 잔고는 get_coin_buyable_cash()에서 따로 보고 있으니
            # 여기서는 굳이 balances에 USDT 안 넣어도 됨)

        return balances

    def get_coin_buyable_cash(
        self,
        quote_asset: str = "USDT",
        market_type: str = "spot",
    ) -> float:
        """
        매수 가능 자산 조회
        - Spot: /api/v3/account 의 balances.free
        - Futures: /fapi/v2/account 의 assets[].availableBalance
        """
        base = self._get_base(market_type)
        headers = self._get_auth_headers()

        if market_type == "futures":
            url = f"{base}/fapi/v2/account"
        else:
            url = f"{base}/api/v3/account"

        try:
            params = self._sign_params({})
            resp = requests.get(url, headers=headers, params=params, timeout=5)

            # 🔍 여기서도 에러 바디 같이 찍기
            if resp.status_code != 200:
                try:
                    self.log(
                        f"❌ [Binance {market_type} {quote_asset} 잔고 조회 실패] "
                        f"status={resp.status_code} | body={resp.text}"
                    )
                finally:
                    resp.raise_for_status()

            data = resp.json()

        except requests.HTTPError as e:
            self.log(f"❌ [Binance {market_type} {quote_asset} 잔고 HTTPError] {e}")
            return 0.0

        except Exception as e:
            self.log(f"❌ [Binance {market_type} {quote_asset} 잔고 조회 예외] {e}")
            return 0.0

        if market_type == "spot":
            for b in data.get("balances", []):
                if b.get("asset") == quote_asset:
                    return float(b.get("free", 0))
        else:
            for a in data.get("assets", []):
                if a.get("asset") == quote_asset:
                    return float(a.get("availableBalance", 0))

        return 0.0

    def get_order_chance(self, symbol: str, market_type: str = "spot") -> Dict[str, Any]:
        """
        심볼의 최소 주문 수량/가격 정보 등 (exchangeInfo)
        - Spot:    GET /api/v3/exchangeInfo?symbol=...
        - Futures: GET /fapi/v1/exchangeInfo?symbol=...
        """
        base = self._get_base(market_type)

        if market_type == "futures":
            url = f"{base}/fapi/v1/exchangeInfo"
        else:
            url = f"{base}/api/v3/exchangeInfo"

        params = {"symbol": symbol}

        try:
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()
            if resp.status_code == 200:
                return data
            else:
                self.log(
                    f"⚠️ [Binance {market_type} 주문 가능 정보 조회 실패] "
                    f"symbol={symbol} | {resp.status_code} | {data}"
                )
                return {}
        except Exception as e:
            self.log(
                f"❌ [Binance {market_type} 주문 가능 정보 조회 예외] "
                f"symbol={symbol} | {e}"
            )
            return {}

    # ============================================================
    # 2. 주문 및 트래킹 (Spot / Futures)
    # ============================================================


    def send_coin_order(
        self,
        market: str,
        side: str,
        volume: float | None = None,
        price: float | None = None,
        ord_type: str = "LIMIT",
        time_in_force: str = "GTC",
        market_type: str = "spot",
        position_side: Optional[str] = None,  # futures: LONG/SHORT
        reduce_only: Optional[bool] = None,   # futures: reduceOnly
        quote_order_qty: float | None = None, # spot MARKET BUY 전용
        skip_normalize: bool = False,         # 🔹 추가: True면 _normalize_volume() 건너뜀
    ) -> str | None:

        base = self._get_base(market_type)
        headers = self._get_auth_headers()

        # API endpoint
        url = f"{base}/fapi/v1/order" if market_type == "futures" else f"{base}/api/v3/order"

        # 필터 조회
        f = self._get_symbol_filters(market, market_type=market_type) or {}
        tick_size      = float(f.get("tickSize", 0) or 0)
        tick_size_str  = f.get("tickSize_str", "")
        min_notional   = float(f.get("minNotional", 0) or 0)

        def _calc_decimals_from_str(s: str) -> int:
            if not s or "." not in s:
                return 0
            return len(s.split(".")[1].rstrip("0"))

        price_decimals = _calc_decimals_from_str(tick_size_str) if tick_size_str else 0

        ord_type_u = ord_type.upper()
        side_u     = side.upper()

        # =========================
        # ① Spot은 quoteOrderQty 우선적으로 사용
        # =========================
        use_quote_amount = (
            market_type == "spot"
            and ord_type_u == "MARKET"
            and quote_order_qty is not None
            and quote_order_qty > 0
        )

        # =========================
        # ② Futures / Spot 공통 수량 정규화
        # =========================
        adj_volume = volume
        adj_price  = price

        # --- 수량 정규화 (Spot은 volume 기반일 때만, Futures 항상 적용) ---
        if not use_quote_amount:
            if adj_volume is not None:
                use_market_filters_flag = (
                    market_type == "futures" and ord_type_u == "MARKET"
                )
                adj_volume = self._normalize_volume(
                    market,
                    adj_volume,
                    market_type=market_type,
                    use_market_filters=use_market_filters_flag,  # 🔥 여기
                )
                if adj_volume <= 0:
                    self.log(
                        f"⚠️ [Binance {market_type} 주문 스킵] "
                        f"volume<=0 (orig={volume}, normalized={adj_volume})"
                    )
                    return None

        # --- 가격 정규화 ---
        if adj_price is not None and tick_size > 0:
            try:
                adj_price = (int(adj_price / tick_size)) * tick_size
            except Exception:
                adj_price = self._floor_to_step(adj_price, tick_size)

            if price_decimals > 0:
                adj_price = float(f"{adj_price:.{price_decimals}f}")

        # =========================
        # ③ Body 구성
        # =========================
        body: Dict[str, Any] = {
            "symbol": market,
            "side": side_u,
            "type": ord_type_u,
        }

        # ---------- LIMIT ----------
        if ord_type_u == "LIMIT":
            if adj_volume is None or adj_price is None:
                raise ValueError("LIMIT 주문에는 volume, price가 필요합니다.")
            body["timeInForce"] = time_in_force
            body["quantity"] = adj_volume
            body["price"] = adj_price

        # ---------- MARKET ----------
        elif ord_type_u == "MARKET":
            # Spot BUY → quoteOrderQty 사용
            if use_quote_amount:
                if min_notional > 0 and quote_order_qty < min_notional:
                    self.log(
                        f"⚠️ [Spot 주문 스킵] quoteOrderQty<{min_notional} (amount={quote_order_qty}) symbol={market}"
                    )
                    return None
                body["quoteOrderQty"] = float(f"{quote_order_qty:.8f}")
            else:
                if adj_volume is None or adj_volume <= 0:
                    raise ValueError("MARKET 주문에는 volume(수량)이 필요합니다.")
                body["quantity"] = adj_volume

        else:
            raise ValueError(f"지원하지 않는 주문 타입: {ord_type}")

        # ---------- Futures 옵션 적용 ----------
        if market_type == "futures":
            if position_side:
                body["positionSide"] = position_side

        # =========================
        # ④ 사인 + 요청 (Futures도 단일 시도)
        # =========================
        params = self._sign_params(body)

        try:
            resp = requests.post(url, headers=headers, params=params, timeout=5)
            data = resp.json()

            if resp.status_code == 200:
                order_id = data.get("orderId")
                msg_qty = (
                    f"quoteOrderQty={quote_order_qty}"
                    if use_quote_amount else
                    f"qty={adj_volume}"
                )

                self.log(f"✅ [Binance {market_type} 주문 성공] {market} {side_u} | {msg_qty}")
                return str(order_id)

            # 실패 로그
            self.log(
                f"❌ [Binance {market_type} 주문 실패] "
                f"HTTP {resp.status_code} | {data} | "
                f"symbol={market}, side={side}, qty={adj_volume}"
            )
            return None

        except Exception as e:
            self.log(f"❌ [Binance {market_type} 주문 예외] {e}")
            return None

    def get_order_details(
        self,
        market: str,
        order_id: str,
        market_type: str = "spot",
    ) -> Dict[str, Any]:
        """
        주문 상세 조회
        Spot:   /api/v3/order + /api/v3/myTrades
        Futures:/fapi/v1/order + /fapi/v1/userTrades
        """
        base = self._get_base(market_type)
        headers = self._get_auth_headers()

        # 1) 주문 정보
        if market_type == "futures":
            url_order = f"{base}/fapi/v1/order"
        else:
            url_order = f"{base}/api/v3/order"

        params_order = self._sign_params({"symbol": market, "orderId": order_id})

        try:
            resp_o = requests.get(url_order, headers=headers, params=params_order, timeout=5)
            data_o = resp_o.json()
            if resp_o.status_code != 200:
                self.log(f"❌ [Binance {market_type} 주문 조회 실패] {resp_o.status_code} | {data_o}")
                return {}
        except Exception as e:
            self.log(f"❌ [Binance {market_type} 주문 조회 예외] {e}")
            return {}

        info = {
            "orderId": data_o.get("orderId"),
            "clientOrderId": data_o.get("clientOrderId"),
            "symbol": data_o.get("symbol"),
            "side": data_o.get("side"),
            "status": data_o.get("status"),
            "origQty": float(data_o.get("origQty", 0)),
            "executedQty": float(data_o.get("executedQty", 0)),
            "price": float(data_o.get("price", 0)),
            "avg_fill_price": 0.0,
            "trades": [],
        }

        # 2) 체결 내역
        if market_type == "futures":
            url_trades = f"{base}/fapi/v1/userTrades"
        else:
            url_trades = f"{base}/api/v3/myTrades"

        params_trades = self._sign_params({"symbol": market})

        try:
            resp_t = requests.get(url_trades, headers=headers, params=params_trades, timeout=5)
            data_t = resp_t.json()
            if resp_t.status_code == 200 and isinstance(data_t, list):
                trades = [t for t in data_t if str(t.get("orderId")) == str(order_id)]
            else:
                trades = []
        except Exception as e:
            self.log(f"❌ [Binance {market_type} 체결 조회 예외] {e}")
            trades = []

        info["trades"] = trades

        # avg_fill_price 계산
        if trades:
            total_qty = 0.0
            total_quote = 0.0
            for t in trades:
                qty = float(t.get("qty", 0))
                price = float(t.get("price", 0))
                total_qty += qty
                total_quote += qty * price
            if total_qty > 0:
                info["avg_fill_price"] = total_quote / total_qty

        return info

    def get_open_orders(
        self,
        market: Optional[str] = None,
        market_type: str = "spot",
    ) -> List[Dict[str, Any]]:
        """
        미체결 주문 조회
        Spot:   /api/v3/openOrders
        Futures:/fapi/v1/openOrders
        """
        base = self._get_base(market_type)
        headers = self._get_auth_headers()

        if market_type == "futures":
            url = f"{base}/fapi/v1/openOrders"
        else:
            url = f"{base}/api/v3/openOrders"

        params: Dict[str, Any] = {}
        if market:
            params["symbol"] = market
        params = self._sign_params(params)

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=5)
            data = resp.json()
            if resp.status_code == 200 and isinstance(data, list):
                return data
            else:
                self.log(f"⚠️ [Binance {market_type} 미체결 조회] {resp.status_code} | {data}")
                return []
        except Exception as e:
            self.log(f"❌ [Binance {market_type} 미체결 조회 예외] {e}")
            return []

    def cancel_order(
        self,
        market: str,
        order_id: str,
        market_type: str = "spot",
    ) -> bool:
        """
        주문 취소
        Spot:   DELETE /api/v3/order
        Futures:DELETE /fapi/v1/order
        """
        base = self._get_base(market_type)
        headers = self._get_auth_headers()

        if market_type == "futures":
            url = f"{base}/fapi/v1/order"
        else:
            url = f"{base}/api/v3/order"

        params = self._sign_params({"symbol": market, "orderId": order_id})

        try:
            resp = requests.delete(url, headers=headers, params=params, timeout=5)
            data = resp.json()

            if resp.status_code == 200:
                self.log(f"✅ [Binance {market_type} 취소 성공] {market} | {order_id}")
                return True

            # 🔹 -2011: Unknown order sent → 이미 체결/취소된 주문인 경우가 많으니 성공 취급
            if isinstance(data, dict) and data.get("code") == -2011:
                return True

            self.log(f"⚠️ [Binance {market_type} 취소 실패] status={resp.status_code} | {data}")
            return False
        except Exception as e:
            self.log(f"❌ [Binance {market_type} 취소 예외] {e}")
            return False

    # ============================================================
    # 3. 시세 데이터 (Public)
    # ============================================================

    def get_coin_current_price(
        self,
        market: str,
        market_type: str = "spot",
    ) -> Optional[float]:
        """
        현재가 조회:
          Spot:    /api/v3/ticker/price
          Futures: /fapi/v1/ticker/price
        """
        base = self._get_base(market_type)

        if market_type == "futures":
            url = f"{base}/fapi/v1/ticker/price"
        else:
            url = f"{base}/api/v3/ticker/price"

        params = {"symbol": market}

        try:
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()

            if resp.status_code == 200 and "price" in data:
                return float(data["price"])

            # 🔹 Invalid symbol(-1121) 처리
            if isinstance(data, dict) and data.get("code") == -1121:
                self.log(
                    f"⚠️ [Binance {market_type} 현재가 조회 실패 - Invalid symbol] "
                    f"symbol={market} | {data}"
                )
                return None

            self.log(
                f"⚠️ [Binance {market_type} 현재가 조회 실패] "
                f"symbol={market} | status={resp.status_code} | data={data}"
            )
            return None
        except Exception as e:
            self.log(f"❌ [Binance {market_type} 현재가 예외] symbol={market} | {e}")
            return None
        
    def get_open_positions(self, market_type: str = "futures") -> dict:
        """
        바이낸스에서 현재 열려 있는 포지션들을 조회해서
        심볼별 메타데이터를 dict 로 반환한다.

        반환 예시:
        {
            "BTCUSDT": {
                "qty": 0.01,
                "entry_price": 42000.0,
                "side": "LONG" | "SHORT" | "BUY",
                "leverage": 10,
                "pnl": 12.5,        # USDT
                "roi": 3.2,         # %
                "entry_time": datetime(...)
            },
            ...
        }
        """
        result: dict = {}

        try:
            if market_type == "futures":
                # 1) 계정 기준으로 현재 포지션/레버리지/미실현손익 파악
                base = self._get_base("futures")
                headers = self._get_auth_headers()
                url = f"{base}/fapi/v2/account"

                params = self._sign_params({})
                resp = requests.get(url, headers=headers, params=params, timeout=5)
                resp.raise_for_status()
                data = resp.json()

                for p in data.get("positions", []):
                    symbol = p.get("symbol")
                    if not symbol:
                        continue

                    pos_amt = float(p.get("positionAmt", 0) or 0.0)
                    if pos_amt == 0.0:
                        continue  # 열린 포지션 없음

                    entry_price_acc = float(p.get("entryPrice", 0) or 0.0)
                    leverage = int(p.get("leverage", 1) or 1)
                    un_pnl = float(p.get("unRealizedProfit", 0) or 0.0)

                    side = "SHORT" if pos_amt < 0 else "LONG"
                    qty = abs(pos_amt)

                    roi = None
                    if entry_price_acc > 0 and qty > 0:
                        roi = (un_pnl / (qty * entry_price_acc)) * 100

                    # 2) userTrades 기반으로 entry_price / entry_time 복원
                    trades = self._get_futures_trades(symbol, limit=1000)
                    pos_meta = self._build_futures_position_from_trades(
                        symbol, trades, target_pos_amt=pos_amt
                    )

                    if pos_meta:
                        entry_price = pos_meta["entry_price"]
                        entry_time = pos_meta["entry_time"]
                    else:
                        # 복원이 안 되면 account 의 entryPrice 사용
                        entry_price = entry_price_acc
                        entry_time = None

                    result[symbol] = {
                        "qty": qty,
                        "entry_price": entry_price,
                        "side": side,
                        "leverage": leverage,
                        "pnl": un_pnl,
                        "roi": roi,
                        "entry_time": entry_time,
                    }

            else:
                # ---------- Spot: balances + myTrades 기반 ----------
                balances = self.get_coin_balance(market_type="spot") or {}

                for symbol, info in balances.items():
                    if symbol == "USDT":
                        continue

                    qty_balance = float(info.get("qty", 0) or 0.0)
                    if qty_balance <= 0:
                        continue

                    trades = self._get_spot_trades(symbol, limit=1000)
                    if trades:
                        pos_info = self._build_spot_position_from_trades(symbol, trades)
                    else:
                        pos_info = None

                    if pos_info:
                        # qty 는 실제 잔고 기준으로 맞추고, entry 정보는 trades 기준 사용
                        pos_info["qty"] = qty_balance
                        result[symbol] = pos_info
                    else:
                        # 트레이드 히스토리 없으면 현재가 기준 fallback
                        price_now = self.get_coin_current_price(symbol, market_type="spot") or 0.0
                        result[symbol] = {
                            "qty": qty_balance,
                            "entry_price": price_now,
                            "side": "BUY",
                            "leverage": 1,
                            "pnl": None,
                            "roi": None,
                            "entry_time": None,
                        }

        except Exception as e:
            self.log(f"[BinanceDataFetcher] get_open_positions error: {e}")

        return result

    def get_futures_position_qty_and_side(self, symbol: str) -> tuple[float, Optional[str]]:
        """
        특정 선물 심볼의 현재 포지션 수량과 방향(LONG/SHORT)을 반환.
        포지션 없으면 (0.0, None)
        """
        balances = self.get_coin_balance(market_type="futures") or {}
        info = balances.get(symbol)
        if not info:
            return 0.0, None

        qty = float(info.get("qty", 0) or 0.0)
        side = info.get("side")
        if qty <= 0:
            return 0.0, None
        return qty, side

    def close_futures_position_full(
        self,
        symbol: str,
        position_side: Optional[str] = None,  # one-way면 None, hedge이면 LONG/SHORT 지정
    ) -> str | None:
        """
        선물 포지션을 100% 청산 (MARKET, reduceOnly).
        - 현재 포지션이 LONG이면 SELL
        - SHORT이면 BUY
        """
        qty, side = self.get_futures_position_qty_and_side(symbol)
        if qty <= 0 or not side:
            self.log(f"ℹ️ [Futures 청산 스킵] {symbol} 포지션 없음")
            return None

        if side.upper() == "LONG":
            close_side = "SELL"
        else:
            close_side = "BUY"

        # 🔹 여기서만 skip_normalize=True → Binance 포지션 수량 그대로 사용
        return self.send_coin_order(
            market=symbol,
            side=close_side,
            volume=qty,
            ord_type="MARKET",
            market_type="futures",
            position_side=position_side,
            skip_normalize=True,   # ✅ 핵심
        )


    def get_coin_ohlcv(
        self,
        market: str,
        interval: str = "5m",
        limit: int = 120,
        market_type: str = "spot",
    ) -> Optional[pd.DataFrame]:
        """
        캔들(OHLCV) 조회
          Spot:    /api/v3/klines
          Futures: /fapi/v1/klines
        """
        base = self._get_base(market_type)

        if market_type == "futures":
            url = f"{base}/fapi/v1/klines"
        else:
            url = f"{base}/api/v3/klines"

        params = {"symbol": market, "interval": interval, "limit": limit}

        try:
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()

            if resp.status_code != 200 or not isinstance(data, list):
                # 🔹 Invalid symbol(-1121)도 여기로 떨어질 수 있음
                if isinstance(data, dict) and data.get("code") == -1121:
                    self.log(
                        f"⚠️ [Binance {market_type} 캔들 조회 실패 - Invalid symbol] "
                        f"symbol={market} | {data}"
                    )
                else:
                    self.log(
                        f"⚠️ [Binance {market_type} 캔들 조회 실패] "
                        f"symbol={market} | status={resp.status_code} | data={data}"
                    )
                return None

            records = []
            for k in data:
                open_time = int(k[0])
                open_ts = pd.to_datetime(open_time, unit="ms")
                records.append(
                    {
                        "datetime": open_ts,
                        "open": float(k[1]),
                        "high": float(k[2]),
                        "low": float(k[3]),
                        "close": float(k[4]),
                        "volume": float(k[5]),
                    }
                )

            df = pd.DataFrame(records)
            df = df.sort_values("datetime").set_index("datetime")
            return df

        except Exception as e:
            self.log(f"❌ [Binance {market_type} 캔들 조회 예외] symbol={market} | {e}")
            return None
        
    # ============================================================
    # 4. Futures 전용 설정: 마진 타입 / 레버리지
    # ============================================================
    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED") -> bool:
        """
        Futures 마진 타입 설정
        POST /fapi/v1/marginType
        margin_type: "ISOLATED" or "CROSSED"
        """
        base = self._get_base("futures")
        headers = self._get_auth_headers()
        url = f"{base}/fapi/v1/marginType"

        body = {
            "symbol": symbol,
            "marginType": margin_type.upper(),
        }

        params = self._sign_params(body)

        try:
            resp = requests.post(url, headers=headers, params=params, timeout=5)
            data = resp.json()

            if resp.status_code == 200:
                self.log(f"✅ [Futures 마진타입 설정 성공] {symbol} -> {margin_type}")
                return True

            # 이미 ISOLATED 인데 또 ISOLATED로 바꾸면 -4046 에러 나는데, 이건 무시 가능
            if isinstance(data, dict) and data.get("code") == -4046:
                self.log(f"ℹ️ [Futures 마진타입 이미 {margin_type}] {symbol} | {data}")
                return True

            self.log(f"⚠️ [Futures 마진타입 설정 실패] {symbol} | {resp.status_code} | {data}")
            return False
        except Exception as e:
            self.log(f"❌ [Futures 마진타입 설정 예외] {symbol} | {e}")
            return False

    def set_leverage(self, symbol: str, leverage: int) -> bool:
        """
        Futures 심볼 레버리지 설정
        POST /fapi/v1/leverage
        """
        base = self._get_base("futures")
        headers = self._get_auth_headers()
        url = f"{base}/fapi/v1/leverage"

        body = {
            "symbol": symbol,
            "leverage": leverage,
        }

        params = self._sign_params(body)

        try:
            resp = requests.post(url, headers=headers, params=params, timeout=5)
            data = resp.json()

            if resp.status_code == 200:
                self.log(f"✅ [Futures 레버리지 설정 성공] {symbol} -> {leverage}x")
                return True

            self.log(f"⚠️ [Futures 레버리지 설정 실패] {symbol} | {resp.status_code} | {data}")
            return False
        except Exception as e:
            self.log(f"❌ [Futures 레버리지 설정 예외] {symbol} | {e}")
            return False
        