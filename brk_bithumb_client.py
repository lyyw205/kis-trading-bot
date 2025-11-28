"""
빗썸(Bithumb) 전용 브로커 클라이언트

- 공식 문서: https://apidocs.bithumb.com
- 사용 API (v2.0 PUBLIC / PRIVATE):
    * [PUBLIC] 분(Minute) 캔들:   GET /v1/candles/minutes/{unit}
    * [PUBLIC] 일(Day) 캔들:      GET /v1/candles/days
    * [PUBLIC] 현재가 정보:       GET /v1/ticker
    * [PRIVATE] 전체 계좌 조회:   GET /v1/accounts
    * [PRIVATE] 주문 가능 정보:   GET /v1/orders/chance
    * [PRIVATE] 주문하기:        POST /v1/orders
"""

import os
import time
import uuid
import hashlib
from typing import Dict, Any, Optional
from urllib.parse import urlencode

import requests
import pandas as pd
import jwt  # pip install pyjwt


class BithumbDataFetcher:
    """
    빗썸용 데이터/주문 래퍼

    - access_key / secret_key는 환경변수:
        BITHUMB_ACCESS_KEY
        BITHUMB_SECRET_KEY
      또는 __init__ 인자로 직접 전달해도 됨.
    """

    def __init__(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        mode=None,
        logger=None,
    ):
        self.base_url = "https://api.bithumb.com"
        self.access_key = access_key or os.getenv("BITHUMB_ACCESS_KEY", "")
        self.secret_key = secret_key or os.getenv("BITHUMB_SECRET_KEY", "")
        self.mode = mode
        self.logger = logger or print

    # --------------------------------------------------
    # 공통 로그
    # --------------------------------------------------
    def log(self, msg: str):
        try:
            self.logger(msg)
        except Exception:
            print(msg)

    # --------------------------------------------------
    # PRIVATE API 인증 헤더 (JWT)
    #   - 공식 예제 기준:
    #       access_key, nonce(UUID), timestamp(ms)
    #       (+ query_hash, query_hash_alg=SHA512)
    # --------------------------------------------------
    def _make_auth_headers(self, params_or_body: dict | None = None) -> dict:
        """
        빗썸 API 2.0 JWT 인증 헤더 생성

        - params_or_body 가 있으면:
          - dict를 query string 형태로 인코딩
          - SHA512(query) → query_hash, query_hash_alg 포함
        - 없으면: access_key, nonce, timestamp만 포함
        """
        if not self.access_key or not self.secret_key:
            raise ValueError("BITHUMB_ACCESS_KEY / BITHUMB_SECRET_KEY 가 설정되어 있지 않습니다.")

        payload: dict = {
            "access_key": self.access_key,
            "nonce": str(uuid.uuid4()),
            "timestamp": round(time.time() * 1000),
        }

        if params_or_body:
            # key1=val1&key2=val2...
            query: bytes = urlencode(params_or_body, doseq=True).encode()
            h = hashlib.sha512()
            h.update(query)
            query_hash = h.hexdigest()

            payload["query_hash"] = query_hash
            payload["query_hash_alg"] = "SHA512"

        jwt_token = jwt.encode(
            payload,
            self.secret_key,
            algorithm="HS256",
        )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jwt_token}",
        }
        return headers

    # --------------------------------------------------
    # 주문 가능 정보 (GET /v1/orders/chance)
    # --------------------------------------------------
    def get_order_chance(self, market: str) -> Dict[str, Any]:
        """
        빗썸 '주문 가능 정보' 조회
        - endpoint: GET /v1/orders/chance
        - param: market (예: "KRW-BTC")
        """
        url = f"{self.base_url}/v1/orders/chance"
        params = {"market": market}

        # 쿼리 파라미터를 JWT query_hash에 포함해야 하므로 params로 서명
        headers = self._make_auth_headers(params)

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=5)
            data = resp.json()
        except Exception as e:
            self.log(f"❌ [BITHUMB chance 조회 예외] {market} | {e}")
            raise

        if resp.status_code != 200:
            # v2.0 에러는 body 안에 에러 정보가 들어옴
            self.log(f"❌ [BITHUMB chance 조회 실패] status={resp.status_code}, body={data}")
            raise RuntimeError(f"chance 조회 실패: {data}")

        return data

    # --------------------------------------------------
    # 계좌 / 잔고
    #   - [API 2.0] PRIVATE: 전체 계좌 조회 GET /v1/accounts
    # --------------------------------------------------
    def get_coin_balance(self) -> Dict[str, Dict[str, float]]:
        """
        코인 보유 잔고 조회 (빗썸)

        반환 예시:
        {
          "KRW-BTC": {"qty": 0.01, "avg_price": 95000000.0},
          "KRW-ETH": {"qty": 0.3,  "avg_price": 3800000.0},
        }
        """
        url = f"{self.base_url}/v1/accounts"
        headers = self._make_auth_headers()

        try:
            resp = requests.get(url, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            self.log(f"❌ [BITHUMB 잔고 조회 실패] {e}")
            return {}

        balances: Dict[str, Dict[str, float]] = {}

        for item in data:
            currency = item["currency"]          # "BTC", "ETH", "KRW", "POINT" 등
            balance = float(item["balance"])

            # KRW / 포인트 계정은 코인 보유 개수에서 제외
            if currency in ("KRW", "POINT"):
                continue

            # 먼지 수준 제외 (필요시 임계값 조정 가능)
            if balance <= 300:
                continue

            avg_price = float(item.get("avg_buy_price") or 0.0)
            market = f"KRW-{currency}"

            balances[market] = {
                "qty": balance,
                "avg_price": avg_price,
            }

        return balances

    def get_coin_buyable_cash(self) -> float:
        """
        코인 매수 가능 KRW (빗썸)
        - 전체 계좌 조회 결과에서 currency == "KRW" 항목의 balance 사용
        """
        url = f"{self.base_url}/v1/accounts"
        headers = self._make_auth_headers()

        try:
            resp = requests.get(url, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            self.log(f"❌ [BITHUMB KRW 잔고 조회 실패] {e}")
            return 0.0

        krw = 0.0
        for item in data:
            if item.get("currency") == "KRW":
                krw = float(item.get("balance", "0") or "0")
                break

        self.log(f"💰 [BITHUMB 코인 매수 가능 KRW] {krw:,.0f}원")
        return krw

    # --------------------------------------------------
    # 주문
    #   - [API 2.0] PRIVATE: 주문하기 POST /v1/orders
    #   - docs: market, side, order_type, price, volume
    # --------------------------------------------------
    def send_coin_order(
        self,
        market: str,
        side: str,
        volume: float | None = None,
        price: float | None = None,
        ord_type: str = "limit",  # "limit" / "price" / "market"
    ) -> str | None:
        """
        빗썸 코인 주문

        - market: "KRW-BTC"
        - side: "bid"(매수) / "ask"(매도)
        - ord_type:
            * "limit" : 지정가 (volume + price 필수)
            * "price" : 시장가 매수 (price = 사용할 KRW 금액, volume 생략)
            * "market": 시장가 매도 (volume 필수, price 생략)

        ✅ 성공 시 order_id (str) 반환, 실패 시 None 반환
        """
        url = f"{self.base_url}/v1/orders"

        body: dict = {
            "market": market,
            "side": side,
            "ord_type": ord_type,
        }

        if volume is not None:
            body["volume"] = str(volume)

        if price is not None:
            body["price"] = str(price)

        headers = self._make_auth_headers(body)

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=5)
            data = resp.json()
        except Exception as e:
            self.log(f"❌ [BITHUMB 주문 예외] {market} {side} | {e}")
            return None

        if resp.status_code in (200, 201):
            order_id = data.get("uuid") # v2는 uuid로 줍니다
            if order_id:
                self.log(f"✅ [BITHUMB 주문 성공] {market} {side} | ID={order_id}")
                return str(order_id)
            
            # 가끔 성공했지만 uuid가 없는 경우 대비
            self.log(f"⚠️ [BITHUMB 주문 응답] {data}")
            return str(data.get("order_id", "")) # 구버전 호환용
            
        else:
            # 에러 메시지 상세 확인
            error_info = data.get("error", {})
            msg = error_info.get("message", str(data))
            self.log(f"❌ [BITHUMB 주문 실패] status={resp.status_code}, msg={msg}")
            return None

    def cancel_order(self, order_id: str) -> bool:
        """
        빗썸 주문 취소
        - endpoint 예시: DELETE /v1/orders/{orderId}
        - 실제 경로/필드는 공식 문서에 맞게 필요하면 미세 조정
        """
        url = f"{self.base_url}/v1/orders/{order_id}"

        body = {"order_id": order_id}
        headers = self._make_auth_headers(body)

        try:
            resp = requests.delete(url, headers=headers, json=body, timeout=5)
            data = resp.json()
        except Exception as e:
            self.log(f"❌ [BITHUMB 주문 취소 예외] order_id={order_id} | {e}")
            return False

        if resp.status_code == 200:
            # 이미 체결/취소된 주문은 에러 코드로 내려줄 수도 있음 → 로그만 보고 스킵
            if data.get("status") in ("0000", None) or "order_id" in data:
                self.log(f"✅ [BITHUMB 주문 취소 성공] order_id={order_id} | resp={data}")
                return True

            self.log(f"⚠️ [BITHUMB 주문 취소 응답] order_id={order_id} | body={data}")
            return False
        else:
            self.log(f"❌ [BITHUMB 주문 취소 실패] order_id={order_id} status={resp.status_code}, body={data}")
            return False

    # --------------------------------------------------
    # 시세 / 현재가
    #   - [API 2.0] PUBLIC: 현재가 정보 GET /v1/ticker
    # --------------------------------------------------
    def get_coin_current_price(self, market: str) -> Optional[float]:
        """
        빗썸 현재가 조회 (REST 1.0)
        - 내부 심볼: "KRW-BTC"
        - 빗썸 REST: /public/ticker/BTC_KRW (order_currency_payment_currency)
        """
        try:
            payment_currency, order_currency = market.split("-")
        except ValueError:
            self.log(f"⚠️ [BITHUMB 현재가] 잘못된 마켓 포맷: {market}")
            return None

        rest_symbol = f"{order_currency}_{payment_currency}"  # ex) BTC_KRW
        url = f"{self.base_url}/public/ticker/{rest_symbol}"

        try:
            resp = requests.get(url, timeout=5)
            resp.raise_for_status()
            data = resp.json()

            status = data.get("status")
            if status != "0000":
                self.log(f"⚠️ [BITHUMB 현재가] status={status} | market={market} | raw={data}")
                return None

            ticker = data.get("data", {})
            price_str = ticker.get("closing_price")

            if price_str is None:
                self.log(f"⚠️ [BITHUMB 현재가] closing_price 없음: {market} | raw={ticker}")
                return None

            return float(price_str)

        except Exception as e:
            self.log(f"❌ [BITHUMB 현재가 조회 실패] {market} | {e}")
            return None

    # --------------------------------------------------
    # 시세 / OHLCV
    #   - [API 2.0] PUBLIC: 분(Minute) 캔들 GET /v1/candles/minutes/{unit}
    # --------------------------------------------------
    def get_coin_ohlcv(
        self,
        market: str,
        interval: str = "minute5",
        count: int = 120,
    ) -> Optional[pd.DataFrame]:
        """
        빗썸 2.0 분(Minute) 캔들 조회
        - interval: "minute5" → 5분봉
        - market: "KRW-BTC"
        - count: 최대 200개까지
        """
        if interval.startswith("minute"):
            unit_str = interval.replace("minute", "")
        else:
            raise ValueError(f"지원하지 않는 interval: {interval}")

        try:
            unit = int(unit_str)
        except ValueError:
            raise ValueError(f"지원하지 않는 interval: {interval}")

        url = f"{self.base_url}/v1/candles/minutes/{unit}"
        params = {
            "market": market,
            "count": count,
        }

        try:
            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return None

            records = []
            for c in data:
                ts = pd.to_datetime(c["candle_date_time_kst"])
                records.append(
                    {
                        "datetime": ts,
                        "open": float(c["opening_price"]),
                        "high": float(c["high_price"]),
                        "low": float(c["low_price"]),
                        "close": float(c["trade_price"]),
                        "volume": float(c["candle_acc_trade_volume"]),
                    }
                )

            df = pd.DataFrame(records)
            df = df.sort_values("datetime").set_index("datetime")
            return df

        except Exception as e:
            self.log(f"❌ [BITHUMB OHLCV 조회 실패] {market} | {e}")
            return None
