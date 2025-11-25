# kis_api.py
import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

class KisDataFetcher:
    def __init__(self, app_key, app_secret, acc_no, mode="real"):
        self.app_key = app_key
        self.app_secret = app_secret
        
        # 계좌번호 하이픈 유무 자동 처리
        if '-' in acc_no:
            self.acc_no_prefix = acc_no.split('-')[0]
            self.acc_no_suffix = acc_no.split('-')[1]
        else:
            self.acc_no_prefix = acc_no[:8]
            self.acc_no_suffix = acc_no[8:]
            
        self.mode = mode
        
        if mode == "real":
            self.base_url = "https://openapi.koreainvestment.com:9443"
            self.tr_id_kr_buy = "TTTC0802U"
            self.tr_id_kr_sell = "TTTC0801U"
            self.tr_id_us_buy = "TTTT1002U"  # 미국 매수
            self.tr_id_us_sell = "TTTT1006U" # 미국 매도
        else:
            self.base_url = "https://openapivts.koreainvestment.com:29443"
            self.tr_id_kr_buy = "VTTC0802U"
            self.tr_id_kr_sell = "VTTC0801U"
            self.tr_id_us_buy = "VTTT1002U"
            self.tr_id_us_sell = "VTTT1006U"
            
        self.access_token = None
        self.token_file = "kis_token_cache.json"
    
    def _normalize_overseas_exchange(self, exchange: str) -> str:
        """
        yfinance/기타 데이터의 EXCD 값을
        한투 주문 API용 코드로 변환
        """
        mapping = {
            # 나스닥 계열
            "NAS": "NASD",
            "NMS": "NASD",
            "NGM": "NASD",
            "NCM": "NASD",

            # 뉴욕
            "NYS": "NYSE",

            # 아멕스
            "AMS": "AMEX",
            "ASE": "AMEX",
        }
        return mapping.get(exchange, exchange)

    # ------------------------
    # 인증 / 공통 요청
    # ------------------------
    def auth(self, force=False):
        # force=True 이면 무조건 새로 발급
        if not force and os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'r') as f:
                    saved_data = json.load(f)
                saved_time = datetime.strptime(
                    saved_data['timestamp'],
                    "%Y-%m-%d %H:%M:%S"
                )
                # 토큰 유효시간: 1시간 → 안전하게 50분 기준
                if datetime.now() - saved_time < timedelta(minutes=50):
                    self.access_token = saved_data['access_token']
                    return
            except:
                pass

        print("🔑 토큰 재발급...")
        url = f"{self.base_url}/oauth2/tokenP"
        headers = {"content-type": "application/json"}
        body = {
            "grant_type": "client_credentials",
            "appkey": self.app_key,
            "appsecret": self.app_secret
        }
        res = requests.post(url, headers=headers, data=json.dumps(body))
        if res.status_code == 200:
            self.access_token = res.json()["access_token"]
            with open(self.token_file, 'w') as f:
                json.dump({
                    "access_token": self.access_token,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }, f)
        else:
            raise Exception(f"토큰 발급 실패: {res.text}")
        
    def get_headers(self, tr_id):
        if not self.access_token:
            self.auth()
        return {
            "content-type": "application/json",
            "authorization": f"Bearer {self.access_token}",
            "appkey": self.app_key,
            "appsecret": self.app_secret,
            "tr_id": tr_id,
            "custtype": "P",   # 개인 계정
        }

    def _request(self, method, url, tr_id, **kwargs):
        headers = self.get_headers(tr_id)
        res = requests.request(method, url, headers=headers, **kwargs)

        # 토큰 만료 체크
        try:
            data = res.json()
        except:
            data = {}

        if res.status_code == 500 and data.get("msg_cd") == "EGW00123":
            # 기간이 만료된 token 입니다.
            print("♻️ 토큰 만료 감지 → 재발급 후 재시도")
            self.auth(force=True)
            headers = self.get_headers(tr_id)
            res = requests.request(method, url, headers=headers, **kwargs)

        return res

    # ------------------------
    # 예수금 / 잔고 관련
    # ------------------------
    def get_kr_buyable_cash(self):
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-psbl-order"
        tr_id = "VTTC8908R" if self.mode == "virtual" else "TTTC8908R"
        headers = self.get_headers(tr_id)
        params = {
            "CANO": self.acc_no_prefix,
            "ACNT_PRDT_CD": self.acc_no_suffix,
            "PDNO": "005930",
            "ORD_UNPR": "0",
            "ORD_DVSN": "01",
            "CMA_EVLU_AMT_ICLD_YN": "Y",
            "OVRS_ICLD_YN": "N"
        }
        res = requests.get(url, headers=headers, params=params)

        try:
            data = res.json()
        except Exception as e:
            print("❌ get_kr_buyable_cash JSON 파싱 실패:", e, res.text)
            return 0

        if res.status_code != 200 or data.get('rt_cd') != '0':
            print("❌ get_kr_buyable_cash 실패:",
                  res.status_code, data.get('rt_cd'), data.get('msg1'))
            return 0

        return int(data['output']['ord_psbl_cash'])
    
    def get_us_buyable_cash(self):
        """
        해외주식 주문가능금액 (원화 기준 계산)
        - 현재 계좌가 전부 원화라면 원화 주문가능금액만 환산해서 해외 매수 가능 금액 계산
        """
        # 1. USD 잔고 조회 (없으면 자동으로 0)
        usd_cash = 0.0
        try:
            tr_id = "VTTS3018R" if self.mode == "virtual" else "TTTS3018R"
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-present-balance"
            headers = self.get_headers(tr_id)
            params = {
                "CANO": self.acc_no_prefix,
                "ACNT_PRDT_CD": self.acc_no_suffix,
                "WCRC_FRCR_DVSN_CD": "02",
                "NATN_CD": "840",
                "TR_MKET_CD": "00",
                "INQR_DVSN_CD": "00",
                "OVRS_EXCG_CD": "NASD",
                "SORT_SQN": "",
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK200": ""
            }
            res = requests.get(url, headers=headers, params=params)
            data = res.json()
            if res.status_code == 200 and data.get('rt_cd') == '0':
                output2 = data.get('output2', [])
                for item in output2:
                    if item.get('crcy_cd') == 'USD':
                        usd_cash = float(item['frcr_dncl_amt_2'])
                        break
        except Exception as e:
            print(f"⚠️ USD 조회 실패(무시): {e}")

        # 2. 원화 주문가능금액 (→ ord_psbl_cash 기반)
        krw_cash = self.get_kr_buyable_cash()  # 여기서 147225원이 들어옴

        # 3. 원화를 달러로 환산
        # 안전하게 높은 환율 사용(1450원)
        converted_usd = krw_cash / 1450

        total_buying_power = usd_cash + converted_usd

        print(f"💰 [해외 매수 가능] USD(${usd_cash}) + KRW({krw_cash}원→${converted_usd:.2f}) = 총 ${total_buying_power:.2f}")

        return total_buying_power

    def get_kr_balance(self):
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        tr_id = "VTTC8434R" if self.mode == "virtual" else "TTTC8434R"

        res = self._request("GET", url, tr_id, params={
            "CANO": self.acc_no_prefix,
            "ACNT_PRDT_CD": self.acc_no_suffix,
            "AFHR_FLPR_YN": "N",
            "OFL_YN": "N",
            "INQR_DVSN": "02",
            "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N",
            "FNCG_AMT_AUTO_RDPT_YN": "N",
            "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "",
            "CTX_AREA_NK100": ""
        })

        stock_dict = {}
        try:
            data = res.json()
        except Exception as e:
            print("❌ get_kr_balance JSON 파싱 실패:", e, res.text)
            return stock_dict

        if res.status_code != 200:
            print("❌ get_kr_balance HTTP 에러:", res.status_code, res.text)
            return stock_dict

        if data.get('rt_cd') != '0':
            print("❌ get_kr_balance rt_cd 실패:", data.get('rt_cd'), data.get('msg1'))
            return stock_dict

        for item in data.get('output1', []):
            try:
                if int(item['hldg_qty']) > 0:
                    stock_dict[item['pdno']] = {
                        'qty': int(item['hldg_qty']),
                        'avg_price': float(item['pchs_avg_pric'])
                    }
            except Exception as e:
                print("⚠️ get_kr_balance 개별 항목 파싱 오류:", e, item)

        return stock_dict

    def get_us_balance(self):
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-balance"
        tr_id = "VTTS3012R" if self.mode == "virtual" else "TTTS3012R"
        headers = self.get_headers(tr_id)
        params = {
            "CANO": self.acc_no_prefix,
            "ACNT_PRDT_CD": self.acc_no_suffix,
            "OVRS_EXCG_CD": "NASD",
            "TR_CRCY_CD": "USD",
            "CTX_AREA_FK200": "",
            "CTX_AREA_NK200": ""
        }
        res = requests.get(url, headers=headers, params=params)

        stock_dict = {}
        try:
            data = res.json()
        except Exception as e:
            print("❌ get_us_balance JSON 파싱 실패:", e, res.text)
            return stock_dict

        if res.status_code != 200:
            print("❌ get_us_balance HTTP 에러:", res.status_code, res.text)
            return stock_dict

        if data.get('rt_cd') != '0':
            print("❌ get_us_balance rt_cd 실패:", data.get('rt_cd'), data.get('msg1'))
            return stock_dict

        for item in data.get('output1', []):
            try:
                qty = float(item['ovrs_cblc_qty'])
                if qty > 0:
                    stock_dict[item['ovrs_pdno']] = {
                        'qty': qty,
                        'avg_price': float(item['pchs_avg_pric'])
                    }
            except Exception as e:
                print("⚠️ get_us_balance 개별 항목 파싱 오류:", e, item)

        return stock_dict

    # ------------------------
    # 주문 관련
    # ------------------------
    def send_kr_order(self, symbol, order_type, qty):
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
        tr_id = self.tr_id_kr_buy if order_type == "buy" else self.tr_id_kr_sell
        headers = self.get_headers(tr_id)
        data = {
            "CANO": self.acc_no_prefix,
            "ACNT_PRDT_CD": self.acc_no_suffix,
            "PDNO": symbol,
            "ORD_DVSN": "01",   # 시장가
            "ORD_QTY": str(qty),
            "ORD_UNPR": "0"
        }
        res = requests.post(url, headers=headers, data=json.dumps(data))

        try:
            j = res.json()
        except Exception as e:
            print("❌ send_kr_order JSON 파싱 실패:", e, res.text)
            return False

        if res.status_code == 200 and j.get('rt_cd') == '0':
            print(f"✅ send_kr_order 성공: {order_type} {symbol} {qty}주")
            return True

        print(
            "❌ send_kr_order 실패:",
            f"HTTP={res.status_code}",
            f"rt_cd={j.get('rt_cd')}",
            f"msg1={j.get('msg1')}",
            f"symbol={symbol}",
            f"qty={qty}",
            f"order_type={order_type}",
        )
        return False

    def send_us_order(self, exchange, symbol, order_type, qty, price):
        """
        해외주식 주문 (현금)
        - exchange: NASD / NYSE / AMEX ...
        - order_type: "buy" / "sell" (TR ID 선택용)
        - 가격/주문 타입은 KIS 샘플과 동일하게: OVRS_ORD_UNPR + ORD_DVSN="00"(지정가)
        """
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
        tr_id = self.tr_id_us_buy if order_type == "buy" else self.tr_id_us_sell

        # 1) 거래소 코드 KIS 형식으로 정규화
        def _normalize_overseas_exchange(excd: str) -> str:
            mapping = {
                "NAS": "NASD",
                "NMS": "NASD",
                "NGM": "NASD",
                "NCM": "NASD",
                "NASD": "NASD",

                "NYS": "NYSE",
                "NYSE": "NYSE",

                "AMS": "AMEX",
                "AMEX": "AMEX",
            }
            return mapping.get(excd, excd)

        ovrs_excg_cd = _normalize_overseas_exchange(exchange)

        # 2) 가격은 문자열로, 소수 4자리 정도만
        order_price = round(float(price), 4)
        order_price_str = f"{order_price:.4f}"

        # 로깅용 실제 주문 금액
        total_amount = round(order_price * qty, 2)

        headers = self.get_headers(tr_id)
        body = {
            "CANO": self.acc_no_prefix,
            "ACNT_PRDT_CD": self.acc_no_suffix,
            "OVRS_EXCG_CD": ovrs_excg_cd,   # ✅ 해외 거래소 코드
            "PDNO": symbol,                 # ✅ 티커
            "ORD_QTY": str(qty),
            "OVRS_ORD_UNPR": order_price_str,  # ✅ 해외 주문단가
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00",                  # ✅ 지정가 (샘플과 동일)
        }

        res = requests.post(url, headers=headers, data=json.dumps(body))

        try:
            j = res.json()
        except Exception as e:
            print("❌ send_us_order JSON 파싱 실패:", e, res.text)
            return False

        if res.status_code == 200 and j.get("rt_cd") == "0":
            print(
                f"✅ send_us_order 성공: {order_type.upper()} {symbol} {qty}주 "
                f"@ {order_price_str} (EXCD={ovrs_excg_cd}, amount=${total_amount:.2f})"
            )
            return True

        print(
            "❌ send_us_order 실패:",
            f"HTTP={res.status_code}",
            f"rt_cd={j.get('rt_cd')}",
            f"msg_cd={j.get('msg_cd')}",
            f"msg1={j.get('msg1')}",
            f"EXCD={ovrs_excg_cd}",
            f"symbol={symbol}",
            f"qty={qty}",
            f"price={order_price_str}",
            f"amount={total_amount}",
            f"order_type={order_type}",
        )
        return False

    # ------------------------
    # 시세 / OHLCV
    # ------------------------
    def get_kr_current_price(self, symbol):
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-price"
        headers = self.get_headers("FHKST01010100")
        params = {"FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol}
        res = requests.get(url, headers=headers, params=params)

        try:
            data = res.json()
        except Exception as e:
            print("❌ get_kr_current_price JSON 실패:", e, res.text)
            return None

        if res.status_code != 200 or data.get('rt_cd') != '0':
            print("❌ get_kr_current_price 실패:",
                  res.status_code, data.get('rt_cd'), data.get('msg1'), symbol)
            return None

        # 🔥 여기부터 방어
        price_str = data.get('output', {}).get('stck_prpr', '')
        if price_str is None or price_str.strip() == '':
            print("⚠️ get_kr_current_price 빈 가격:", symbol, data.get('output'))
            return None

        try:
            return float(price_str.replace(',', ''))
        except Exception as e:
            print("⚠️ get_kr_current_price 변환 실패:", e, symbol, price_str)
            return None

    def get_us_current_price(self, exchange, symbol):
        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/price"
        tr_id = "HHDFS76200200"
        res = self._request("GET", url, tr_id, params={
            "AUTH": "",
            "EXCD": exchange,
            "SYMB": symbol
        })

        try:
            data = res.json()
        except Exception as e:
            print("❌ get_us_current_price JSON 실패:", e, res.text)
            return None

        if res.status_code != 200 or data.get('rt_cd') != '0':
            print("❌ get_us_current_price 실패:",
                  res.status_code, data.get('rt_cd'), data.get('msg1'), exchange, symbol)
            return None

        price_str = data.get('output', {}).get('last', '')
        if price_str is None or price_str.strip() == '':
            print("⚠️ get_us_current_price 빈 가격:", exchange, symbol, data.get('output'))
            return None

        try:
            return float(price_str.replace(',', ''))
        except Exception as e:
            print("⚠️ get_us_current_price 변환 실패:", e, exchange, symbol, price_str)
            return None

    def get_minute_ohlcv_5m(self, symbol, count=200):
        """
        국내 5분봉 OHLCV 조회
        """
        if not self.access_token:
            self.auth()

        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        tr_id = "FHKST03010200"   # 국내 분봉 차트 TR

        headers = self.get_headers(tr_id)
        params = {
            "FID_ETC_CLS_CODE": "",
            "FID_COND_MRKT_DIV_CODE": "J",   # 주식
            "FID_INPUT_ISCD": symbol,        # 종목코드
            "FID_INPUT_HOUR_1": "",          # 기준시간 ("" = 최신부터)
            "FID_PW_DATA_INCU_YN": "N",
            "FID_TIME_INTERVAL": "5"         # 5분봉
        }

        res = requests.get(url, headers=headers, params=params)

        try:
            data = res.json()
        except Exception as e:
            print("❌ get_minute_ohlcv_5m JSON 실패:", e, res.text)
            return pd.DataFrame()

        if res.status_code != 200 or data.get("rt_cd") != "0":
            print("❌ get_minute_ohlcv_5m 실패:",
                  res.status_code, data.get("rt_cd"), data.get("msg1"), symbol)
            return pd.DataFrame()

        if "output2" not in data or not data["output2"]:
            print("❌ get_minute_ohlcv_5m output2 비어있음:", symbol)
            return pd.DataFrame()

        df = pd.DataFrame(data["output2"])

        try:
            df = df[[
                "stck_bsop_date",
                "stck_cntg_hour",
                "stck_oprc",
                "stck_hgpr",
                "stck_lwpr",
                "stck_prpr",
                "cntg_vol"
            ]]
        except KeyError:
            print("⚠️ get_minute_ohlcv_5m 컬럼 매핑 실패, 실제 컬럼:", df.columns.tolist())
            return pd.DataFrame()

        # 날짜 + 시간 → datetime index
        df["datetime"] = pd.to_datetime(df["stck_bsop_date"] + df["stck_cntg_hour"])
        df.set_index("datetime", inplace=True)

        df = df.rename(columns={
            "stck_oprc": "open",
            "stck_hgpr": "high",
            "stck_lwpr": "low",
            "stck_prpr": "close",
            "cntg_vol": "volume"
        })

                # 🔥 여기서 문자열 → 숫자 변환 (에러는 NaN으로)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 숫자 변환 안 된 행 제거
        df = df.dropna(subset=["open", "high", "low", "close"])

        if df.empty:
            print("⚠️ get_minute_ohlcv_5m 변환 후 데이터 없음:", symbol)
            return pd.DataFrame()

        numeric_cols = ["open", "high", "low", "close", "volume"]
        for col in numeric_cols:
            # errors='coerce': 숫자로 못 바꾸는 값('', 문자 등)은 NaN으로 변환
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # NaN이 포함된 행 제거 (빈 데이터 제거)
        df = df.dropna(subset=numeric_cols)

        # 정렬 및 개수 자르기
        df = df.sort_index()
        if len(df) > count:
            df = df.iloc[-count:]

        return df
    
    # ========================
    # ② 해외 5분봉 OHLCV
    # ========================
    def get_us_minute_ohlcv_5m(self, exchange, symbol, count=200):
        """
        해외 5분봉 OHLCV 조회 (연속 조회 기능 추가)
        """
        if self.mode == "virtual":
            print("⚠️ 해외 분봉 조회는 모의투자 미지원 → 빈 DataFrame 반환")
            return pd.DataFrame()

        if not self.access_token:
            self.auth()

        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/inquire-time-itemchartprice"
        tr_id = "HHDFS76950200"

        headers = self.get_headers(tr_id)
        
        df_list = []
        collected_count = 0
        next_key = ""  # 다음 페이지 키

        # 목표 개수(count)를 채울 때까지 반복
        while collected_count < count:
            params = {
                "AUTH": "",
                "EXCD": exchange,
                "SYMB": symbol,
                "NMIN": "5",
                "PINC": "Y",
                "NEXT": "1" if next_key else "0",  # 첫 요청은 0, 그 뒤론 1
                "NREC": "120",                     # 한 번에 요청할 개수 (최대 120)
                "FILL": "0",
                "KEYB": next_key                   # 다음 페이지 키
            }

            time.sleep(0.2) # API 호출 제한 방지 (필수)
            
            res = requests.get(url, headers=headers, params=params)

            try:
                data = res.json()
            except Exception as e:
                print("❌ get_us_minute_ohlcv_5m JSON 실패:", e)
                break

            if res.status_code != 200 or data.get("rt_cd") != "0":
                print("❌ 실패:", data.get("msg1"), symbol)
                break

            if "output2" not in data or not data["output2"]:
                break

            # 1. 이번 페이지 데이터 변환
            chunk = pd.DataFrame(data["output2"])
            
            try:
                chunk = chunk[[
                    "kymd", "khms", "open", "high", "low", "last", "evol"
                ]]
            except KeyError:
                break

            chunk["datetime"] = pd.to_datetime(chunk["kymd"] + chunk["khms"])
            chunk.set_index("datetime", inplace=True)
            chunk = chunk.rename(columns={
                "last": "close",
                "evol": "volume"
            })
            chunk = chunk[["open", "high", "low", "close", "volume"]].astype(float)
            
            # 2. 리스트에 저장 (최신 데이터가 앞쪽에 오므로 뒤집거나 나중에 정렬)
            df_list.append(chunk)
            collected_count += len(chunk)

            # 3. 다음 페이지 키 확인 (없으면 종료)
            # 해외주식 분봉은 output1에 next 키가 들어오는 경우가 많음
            output1 = data.get("output1", {})
            next_key = output1.get("next")  # 혹은 "next_key"

            # API에 따라 next키가 없거나, 더 이상 데이터가 없으면 종료
            if not next_key or len(chunk) == 0:
                break
                
            # 너무 많이 조회했으면 강제 종료 (무한루프 방지)
            if collected_count >= count:
                break

        if not df_list:
            return pd.DataFrame()

        # 여러 페이지 합치기
        df_all = pd.concat(df_list)
        
        # 중복 제거 및 시간순 정렬
        df_all = df_all[~df_all.index.duplicated(keep='first')]
        df_all = df_all.sort_index()

        # 요청한 개수만큼 자르기 (최신순으로 뒤에서부터)
        if len(df_all) > count:
            df_all = df_all.iloc[-count:]

        return df_all


    def get_ohlcv(self, region, symbol, exchange=None, interval="5m", count=200):
        """
        공통 OHLCV 조회 헬퍼
        interval:
          - '5m' : 5분봉
          - '1d' : 일봉
        """
        if region == "KR":
            if interval == "5m":
                return self.get_minute_ohlcv_5m(symbol, count=count)
            elif interval == "1d":
                return self.get_daily_ohlcv(symbol)
            else:
                raise ValueError(f"지원하지 않는 interval: {interval}")
        else:
            if interval == "5m":
                return self.get_us_minute_ohlcv_5m(exchange, symbol, count=count)
            elif interval == "1d":
                return self.get_overseas_daily_ohlcv(exchange, symbol)
            else:
                raise ValueError(f"지원하지 않는 interval: {interval}")

    def get_daily_ohlcv(self, symbol):
        if not self.access_token:
            self.auth()
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        headers = self.get_headers("FHKST03010100")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J",
            "FID_INPUT_ISCD": symbol,
            "FID_INPUT_DATE_1": (datetime.now() - timedelta(days=200)).strftime("%Y%m%d"),
            "FID_INPUT_DATE_2": datetime.now().strftime("%Y%m%d"),
            "FID_PERIOD_DIV_CODE": "D",
            "FID_ORG_ADJ_PRC": "1"
        }
        res = requests.get(url, headers=headers, params=params)

        try:
            data = res.json()
        except Exception as e:
            print("❌ get_daily_ohlcv JSON 실패:", e, res.text)
            return pd.DataFrame()

        if res.status_code != 200 or data.get('rt_cd') != '0':
            print("❌ get_daily_ohlcv 실패:",
                  res.status_code, data.get('rt_cd'), data.get('msg1'), symbol)
            return pd.DataFrame()

        if 'output2' not in data or not data['output2']:
            print("❌ get_daily_ohlcv output2 비어있음:", symbol)
            return pd.DataFrame()

        df = pd.DataFrame(data['output2'])
        df = df[['stck_bsop_date', 'stck_oprc', 'stck_hgpr',
                 'stck_lwpr', 'stck_clpr', 'acml_vol']]
        df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df.astype(float).sort_index()

    def get_overseas_daily_ohlcv(self, exchange, symbol):
            """
            해외 주식 일봉 차트 조회 (기간 지정 가능 TR 사용)
            TR ID: HHDFS76500200 (해외주식 종목/지수/환율 기간별 시세)
            """
            if not self.access_token:
                self.auth()
                
            url = f"{self.base_url}/uapi/overseas-price/v1/quotations/dailyprice"
            
            headers = self.get_headers("HHDFS76240000")
            params = {
                "AUTH": "",
                "EXCD": exchange,
                "SYMB": symbol,
                "GUBN": "0", # 0:일, 1:주, 2:월
                "BYMD": datetime.now().strftime("%Y%m%d"), # 기준일
                "MODP": "1"  # 0:미수정, 1:수정주가
            }
            
            res = requests.get(url, headers=headers, params=params)
            
            if res.status_code == 200 and res.json().get('rt_cd') == '0':
                output2 = res.json().get('output2', [])
                if not output2:
                    return pd.DataFrame()
                    
                df = pd.DataFrame(output2)
                # API가 최신순으로 줌. 100개 들어오는지 확인
                
                df = df[['xymd', 'open', 'high', 'low', 'clos', 'tvol']]
                df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # 오름차순 정렬 (과거 -> 현재)
                df = df.astype(float).sort_index()
                
                return df
                
            return pd.DataFrame()