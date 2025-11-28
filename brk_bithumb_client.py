# brk_bithumb_client.py
"""
빗썸(Bithumb) 전용 브로커 클라이언트

- 공식 문서: https://apidocs.bithumb.com
- 사용 API (v2.0 PUBLIC / PRIVATE):
    * [PUBLIC] 분(Minute) 캔들:   GET /v1/candles/minutes/{unit}
    * [PUBLIC] 일(Day) 캔들:      GET /v1/candles/days
    * [PUBLIC] 현재가 정보:       GET /v1/ticker
    * [PRIVATE] 전체 계좌 조회:   GET /v1/accounts
    * [PRIVATE] 주문하기:        POST /v1/orders

- 이 클래스는 UpbitDataFetcher와 비슷한 느낌으로 설계되어 있고,
  코인 자산군(region="CR") 전용으로 사용할 수 있게 구성함.
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
    def _make_auth_headers(self, body: dict | None = None) -> dict:
        """
        빗썸 API 2.0 JWT 인증 헤더 생성
        - body가 있으면: body를 query string으로 인코딩 → SHA512 해시 → query_hash / query_hash_alg 포함
        - body가 없으면: access_key, nonce, timestamp만 포함
        """
        if not self.access_key or not self.secret_key:
            raise ValueError("BITHUMB_ACCESS_KEY / BITHUMB_SECRET_KEY 가 설정되어 있지 않습니다.")

        # 1) 기본 페이로드
        payload: dict = {
            "access_key": self.access_key,
            "nonce": str(uuid.uuid4()),                 # 무작위 UUID
            "timestamp": round(time.time() * 1000),     # ms 단위 타임스탬프
        }

        # 2) body가 있을 경우 → query string 해시(SHA512)
        if body:
            # 예: {"order_currency": "BTC", "payment_currency": "KRW", ...}
            query: bytes = urlencode(body).encode()     # key1=val1&key2=val2...

            h = hashlib.sha512()
            h.update(query)
            query_hash = h.hexdigest()

            payload["query_hash"] = query_hash
            payload["query_hash_alg"] = "SHA512"

        # 3) JWT 생성 (HS256, secret_key로 서명)
        jwt_token = jwt.encode(
            payload,
            self.secret_key,
            algorithm="HS256",        # 문서에서 권장하는 알고리즘
        )

        # 4) Authorization 헤더로 포함
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jwt_token}",
        }
        return headers

    # --------------------------------------------------
    # 계좌 / 잔고
    #   - [API 2.0] PRIVATE: 전체 계좌 조회 GET /v1/accounts
    #   - 응답: [{currency, balance, locked, avg_buy_price, unit_currency, ...}, ...]
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

            # 1) KRW / 포인트 계정은 코인 보유 개수에서 제외
            if currency in ("KRW", "POINT"):
                continue

            # 2) 잔고 0 이하(혹은 먼지 수준)는 제외
            if balance <= 300:
                continue

            # 빗썸 평균 매수가 (없으면 0.0)
            avg_price = float(item.get("avg_buy_price") or 0.0)

            # 우리 쪽에서 쓰는 마켓 형식으로 변환
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
    #   - body 필드 (문서 기준): market, side, volume, price, ord_type ...
    # --------------------------------------------------
    def send_coin_order(
        self,
        market: str,
        side: str,
        volume: float | None = None,
        price: float | None = None,
        ord_type: str = "limit",  # "limit" / "market" 등 실제 빗썸 스펙에 맞게
    ) -> bool:
        """
        빗썸 코인 주문
        - market: "KRW-BTC" 같은 마켓 코드
        - side: "bid"(매수) / "ask"(매도)
        - volume: 수량
        - price: 가격 (지정가인 경우)
        """

        # 🔹 빗썸 주문 엔드포인트 (공식 레퍼런스 보고 정확한 path로 바꿔줘)
        url = f"{self.base_url}/v1/orders"

        # 빗썸이 요구하는 파라미터 이름대로 맞추기 (예시는 구조만)
        body: dict = {
            "market": market,
            "side": side,           # "bid" / "ask"
            "ord_type": ord_type,   # "limit" / "price" / "market"
        }
        if volume is not None:
            body["volume"] = str(volume)     # 문자열로 보내는 게 안전
        if price is not None and ord_type == "limit":
            body["price"] = str(price)

        # ✅ JWT 인증 헤더 생성 (요청 body 기반으로 query_hash까지 포함)
        headers = self._make_auth_headers(body)

        try:
            resp = requests.post(url, headers=headers, json=body, timeout=5)
            data = resp.json()
        except Exception as e:
            self.log(f"❌ [BITHUMB 주문 예외] {market} {side} | {e}")
            return False

        # 빗썸 응답 스펙에 맞게 성공/실패 체크
        if resp.status_code in (200, 201) and not data.get("status") in ("5100", "5000"):
            self.log(f"✅ [BITHUMB 주문 성공] {market} {side} {volume} @ {price} | resp={data}")
            return True
        else:
            self.log(
                f"❌ [BITHUMB 주문 실패] status={resp.status_code}, body={data}"
            )
            return False

    # --------------------------------------------------
    # 시세 / 현재가
    #   - [API 2.0] PUBLIC: 현재가 정보 GET /v1/ticker
    #   - 응답 필드: trade_price (현재가), opening_price, high_price, low_price, ...
    # --------------------------------------------------
    def get_coin_current_price(self, market: str) -> Optional[float]:
        """
        빗썸 현재가 조회 (REST 1.0)
        - 내부 심볼: "KRW-BTC"
        - 빗썸 REST: /public/ticker/BTC_KRW (order_currency_payment_currency)
        """
        try:
            # "KRW-BTC" -> ["KRW", "BTC"]
            payment_currency, order_currency = market.split("-")
        except ValueError:
            self.log(f"⚠️ [BITHUMB 현재가] 잘못된 마켓 포맷: {market}")
            return None

        # 빗썸 REST 포맷: BTC_KRW
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
            # 문서상 closing_price가 종가(현재가 역할)
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
    #   - [API 2.0] PUBLIC:
    #       * 분(Minute) 캔들: GET /v1/candles/minutes/{unit}
    #       * 일(Day) 캔들:    GET /v1/candles/days
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
        # "minute5" -> 5
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
            "market": market,   # ex) KRW-BTC
            "count": count,
            # "to": 생략하면 최신 캔들 기준
        }

        try:
            resp = requests.get(url, params=params, timeout=5)
            resp.raise_for_status()
            data = resp.json()
            if not data:
                return None

            records = []
            for c in data:
                # candle_date_time_kst: 'yyyy-MM-ddTHH:mm:ss'
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


