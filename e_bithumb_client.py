# Bithumb 전용 브로커 클라이언트

# - Bithumb API 2.0 JWT 인증 기반
# - 계좌 잔고(KRW 및 코인) 조회, 매수/매도 주문, 주문 상세/미체결/취소 처리
# - 현재가 및 분봉(OHLCV) 시세 조회까지 포함된 통합형 데이터+주문 래퍼

# 주요 기능:
# 1) get_coin_balance()        : 보유 코인 잔고 조회 (소액 먼지 제외)
# 2) get_coin_buyable_cash()   : 매수 가능 KRW 조회
# 3) get_order_chance()        : 특정 마켓 주문 가능 정보 조회
# 4) send_coin_order()         : 주문 전송 (limit/market) 및 결과 UUID 반환
# 5) get_order_details()       : 주문 상세 조회 + 실제 체결 평단가(avg_fill_price) 계산
# 6) get_open_orders()         : 특정 마켓 미체결(wait) 주문 목록 조회
# 7) cancel_order()            : 주문 취소 (DELETE /v1/order)
# 8) get_coin_current_price()  : 현재가 조회 (Legacy Public API, KRW-BTC → BTC_KRW 변환)
# 9) get_coin_ohlcv()          : 분봉 시세(OHLCV) 조회 및 pandas DataFrame 변환

import os
import time
import uuid
import hashlib
from typing import Dict, Any, Optional, List
from urllib.parse import urlencode

import requests
import pandas as pd
import jwt  # pip install pyjwt


class BithumbDataFetcher:
    """
    빗썸용 데이터/주문 래퍼 Class
    """

    def __init__(
        self,
        access_key: Optional[str] = None,
        secret_key: Optional[str] = None,
        mode: str = None,
        logger=None,
    ):
        self.base_url = "https://api.bithumb.com"
        self.access_key = access_key or os.getenv("BITHUMB_ACCESS_KEY", "")
        self.secret_key = secret_key or os.getenv("BITHUMB_SECRET_KEY", "")
        self.mode = mode
        self.logger = logger or print

    def log(self, msg: str):
        """로그 출력 래퍼"""
        try:
            self.logger(msg)
        except Exception:
            print(msg)

    def _make_auth_headers(self, params_or_body: dict | None = None) -> dict:
        """
        빗썸 API 2.0 JWT 인증 헤더 생성
        - query_hash 사용 (SHA512)
        """
        if not self.access_key or not self.secret_key:
            raise ValueError("API Key가 설정되지 않았습니다.")

        payload: dict = {
            "access_key": self.access_key,
            "nonce": str(uuid.uuid4()),
            "timestamp": round(time.time() * 1000),
        }

        if params_or_body:
            query: bytes = urlencode(params_or_body, doseq=True).encode()
            h = hashlib.sha512()
            h.update(query)
            query_hash = h.hexdigest()

            payload["query_hash"] = query_hash
            payload["query_hash_alg"] = "SHA512"

        jwt_token = jwt.encode(payload, self.secret_key, algorithm="HS256")
        
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {jwt_token}",
        }

    # ============================================================
    # 1. 자산 및 잔고 조회
    # ============================================================

    def get_coin_balance(self) -> Dict[str, Dict[str, float]]:
        """보유 코인 잔고 조회 (먼지 제외)"""
        url = f"{self.base_url}/v1/accounts"
        headers = self._make_auth_headers()

        try:
            resp = requests.get(url, headers=headers, timeout=5)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            self.log(f"❌ [잔고 조회 실패] {e}")
            return {}

        balances = {}
        for item in data:
            currency = item.get("currency")
            if currency in ("KRW", "POINT",'P'):
                continue

            balance = float(item.get("balance", 0))
            avg_price = float(item.get("avg_buy_price", 0))

            total_value = balance * avg_price
            # 잔고가 너무 적으면(예: 0.0001 등) 제외하려면 기준 상향 조정 필요
            if total_value <= 999:
                continue

            market = f"KRW-{currency}"
            balances[market] = {
                "qty": balance,
                "avg_price": avg_price,
            }
        return balances

    def get_coin_buyable_cash(self) -> float:
        """매수 가능 KRW 조회"""
        url = f"{self.base_url}/v1/accounts"
        headers = self._make_auth_headers()

        try:
            resp = requests.get(url, headers=headers, timeout=5)
            data = resp.json()
            for item in data:
                if item.get("currency") == "KRW":
                    return float(item.get("balance", 0))
        except Exception as e:
            self.log(f"❌ [KRW 잔고 조회 실패] {e}")
        return 0.0

    def get_order_chance(self, market: str) -> Dict[str, Any]:
        """주문 가능 상세 정보"""
        url = f"{self.base_url}/v1/orders/chance"
        params = {"market": market}
        headers = self._make_auth_headers(params)
        try:
            resp = requests.get(url, headers=headers, params=params, timeout=5)
            return resp.json()
        except Exception:
            return {}

    # ============================================================
    # 2. 주문 및 트래킹 (핵심 수정 사항)
    # ============================================================

    def send_coin_order(
        self,
        market: str,
        side: str,
        volume: float | None = None,
        price: float | None = None,
        ord_type: str = "limit",
    ) -> str | None:
        """
        주문 전송
        - 성공 시: 주문 UUID 반환
        - 실패 시: None 반환
        """
        url = f"{self.base_url}/v1/orders"
        
        # 파라미터 구성
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
            
            if resp.status_code in (200, 201):
                order_id = data.get("uuid")
                self.log(f"✅ [주문 성공] {market} {side} | ID={order_id}")
                return str(order_id)
            else:
                error_msg = data.get("error", {}).get("message", str(data))
                self.log(f"❌ [주문 실패] {resp.status_code} | {error_msg}")
                return None

        except Exception as e:
            self.log(f"❌ [주문 예외] {e}")
            return None

    def get_order_details(self, order_id: str) -> Dict[str, Any]:
        """
        [NEW] 주문 상세 조회 & 실제 체결 가격 계산
        - 반환값에 'avg_fill_price'(실제 평단가)를 포함합니다.
        """
        url = f"{self.base_url}/v1/orders/{order_id}"
        headers = self._make_auth_headers()  # Path param은 hash 제외

        try:
            resp = requests.get(url, headers=headers, timeout=5)
            data = resp.json()
        except Exception as e:
            self.log(f"❌ [주문 상세 조회 예외] {e}")
            return {}

        if resp.status_code != 200:
            return {}

        # 기본 정보 파싱
        info = {
            "uuid": data.get("uuid"),
            "side": data.get("side"),
            "state": data.get("state"),  # wait, done, cancel
            "market": data.get("market"),
            "ord_price": float(data.get("price") or 0),     # 주문 낸 가격
            "ord_volume": float(data.get("volume") or 0),   # 주문 낸 수량
            "executed_volume": float(data.get("executed_volume") or 0), # 체결된 수량
            "paid_fee": float(data.get("paid_fee") or 0),
            "trades": data.get("trades", []),
            "avg_fill_price": 0.0,
        }

        # 실제 체결 평단가(avg_fill_price) 계산
        trades = info["trades"]
        if trades:
            total_amt = 0.0
            total_qty = 0.0
            for t in trades:
                p = float(t.get("price", 0))
                v = float(t.get("volume", 0))
                total_amt += p * v
                total_qty += v
            
            if total_qty > 0:
                info["avg_fill_price"] = total_amt / total_qty
        
        return info

    def get_open_orders(self, market: str) -> List[Dict[str, Any]]:
        """
        [NEW] 특정 마켓의 미체결(wait) 주문 목록 조회
        """
        url = f"{self.base_url}/v1/orders"
        params = {"market": market, "state": "wait"}
        headers = self._make_auth_headers(params)

        try:
            resp = requests.get(url, headers=headers, params=params, timeout=5)
            data = resp.json()
            
            if isinstance(data, list):
                return data
            return []
        except Exception as e:
            self.log(f"❌ [미체결 조회 예외] {e}")
            return []

    def cancel_order(self, order_id: str) -> bool:
        """
        주문 취소

        - 빗썸 API 2.0 기준
        DELETE https://api.bithumb.com/v1/order
        query: uuid=주문UUID
        """
        url = f"{self.base_url}/v1/order"
        params = {"uuid": order_id}

        # jwt payload에 query_hash 포함 (uuid 기반)
        headers = self._make_auth_headers(params)

        try:
            resp = requests.delete(url, headers=headers, params=params, timeout=5)
            try:
                data = resp.json()
            except Exception:
                data = {}

            if resp.status_code == 200 and data.get("uuid"):
                self.log(f"✅ [취소 성공] {order_id}")
                return True

            # 여기까지 왔으면 실패
            self.log(
                f"⚠️ [취소 실패] status={resp.status_code} "
                f"body={data}"
            )
            return False

        except Exception as e:
            self.log(f"❌ [취소 예외] {order_id} | {e}")
            return False

    # ============================================================
    # 3. 시세 데이터 (Public)
    # ============================================================

    def get_coin_current_price(self, market: str) -> Optional[float]:
        """현재가 조회 (Legacy Public API)"""
        try:
            # KRW-BTC -> BTC_KRW 변환
            parts = market.split("-")
            rest_symbol = f"{parts[1]}_{parts[0]}"
            url = f"{self.base_url}/public/ticker/{rest_symbol}"
            
            resp = requests.get(url, timeout=5)
            data = resp.json()
            
            # 빗썸 Public API 구조: status, data: { closing_price, ... }
            if data.get("status") == "0000":
                return float(data["data"]["closing_price"])
            return None
        except Exception:
            return None

    def get_coin_ohlcv(
        self, market: str, interval: str = "minute5", count: int = 120
    ) -> Optional[pd.DataFrame]:
        """분봉 조회"""
        # interval: minute1, minute3, minute5 ...
        unit = interval.replace("minute", "")
        url = f"{self.base_url}/v1/candles/minutes/{unit}"
        params = {"market": market, "count": count}

        try:
            resp = requests.get(url, params=params, timeout=5)
            data = resp.json()
            
            if not data or not isinstance(data, list):
                return None

            records = []
            for c in data:
                records.append({
                    "datetime": pd.to_datetime(c["candle_date_time_kst"]),
                    "open": float(c["opening_price"]),
                    "high": float(c["high_price"]),
                    "low": float(c["low_price"]),
                    "close": float(c["trade_price"]),
                    "volume": float(c["candle_acc_trade_volume"]),
                })
            
            df = pd.DataFrame(records)
            df = df.sort_values("datetime").set_index("datetime")
            return df

        except Exception as e:
            self.log(f"❌ [캔들 조회 실패] {e}")
            return None