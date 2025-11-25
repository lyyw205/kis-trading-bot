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
            self.tr_id_us_buy = "JTTT1002U" 
            self.tr_id_us_sell = "JTTT1006U" 
        else:
            self.base_url = "https://openapivts.koreainvestment.com:29443"
            self.tr_id_kr_buy = "VTTC0802U"
            self.tr_id_kr_sell = "VTTC0801U"
            self.tr_id_us_buy = "VTTT1002U"
            self.tr_id_us_sell = "VTTT1006U"
            
        self.access_token = None
        self.token_file = "kis_token_cache.json"

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
            해외주식 잔고 조회 (TTTS3018R) - 디버깅용
            - 국가코드 000(전체)으로 조회하여 모든 통화내역을 출력합니다.
            """
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-present-balance"
            tr_id = "VTTS3018R" if self.mode == "virtual" else "TTTS3018R"
            headers = self.get_headers(tr_id)
            
            params = {
                "CANO": self.acc_no_prefix,
                "ACNT_PRDT_CD": self.acc_no_suffix,
                "WCRC_FRCR_DVSN_CD": "02",
                "NATN_CD": "000",          # ✅ 000: 전체 국가 (미국 한정 X)
                "TR_MKET_CD": "00",
                "INQR_DVSN_CD": "00",
                "OVRS_EXCG_CD": "NASD",
                "SORT_SQN": "",
                "CTX_AREA_FK200": "",
                "CTX_AREA_NK200": ""
            }
            
            res = requests.get(url, headers=headers, params=params)

            try:
                data = res.json()
                output2 = data.get('output2', [])
                
                # 🔍 계좌에 있는 모든 돈 출력 (여기 로그를 꼭 확인해주세요!)
                print(f"🔍 [계좌 통화 내역] --------------------")
                found_usd = 0.0
                
                for item in output2:
                    currency = item['crcy_cd']     # 통화코드 (USD, KRW, CNY 등)
                    amount = item['frcr_dncl_amt_2'] # 잔고 수량
                    print(f"   👉 통화: {currency} | 잔고: {amount}")
                    
                    if currency == 'USD':
                        found_usd = float(amount)

                print(f"----------------------------------------")
                
                return found_usd

            except Exception as e:
                print(f"⚠️ 파싱 에러: {e}")
                return 0.0
            
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
            "ORD_DVSN": "01",
            "ORD_QTY": str(qty),
            "ORD_UNPR": "0"
        }
        res = requests.post(url, headers=headers, data=json.dumps(data))
        if res.status_code == 200 and res.json().get('rt_cd') == '0':
            return True
        return False

    def send_us_order(self, exchange, symbol, order_type, qty, price):
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
        tr_id = self.tr_id_us_buy if order_type == "buy" else self.tr_id_us_sell
        headers = self.get_headers(tr_id)
        data = {
            "CANO": self.acc_no_prefix,
            "ACNT_PRDT_CD": self.acc_no_suffix,
            "OVRS_EXCG_CD": exchange,
            "PDNO": symbol,
            "ORD_QTY": str(qty),
            "ORD_UNPR": str(price),
            "ORD_DVSN": "00",
            "ORD_SVR_DVSN_CD": "0"
        }
        res = requests.post(url, headers=headers, data=json.dumps(data))
        if res.status_code == 200 and res.json().get('rt_cd') == '0':
            return True
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
        해외 5분봉 OHLCV 조회
        - KIS 해외 분봉 TR: v1_해외주식-030 (해외주식분봉조회)
        - 실전 TR ID: HHDFS76950200
        - 모의투자 미지원이라 mode='virtual' 이면 빈 DF 반환
        """
        if self.mode == "virtual":
            print("⚠️ 해외 분봉 조회는 모의투자 미지원 → 빈 DataFrame 반환")
            return pd.DataFrame()

        if not self.access_token:
            self.auth()

        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/inquire-time-itemchartprice"
        tr_id = "HHDFS76950200"  # 해외분봉조회 TR ID (실전)

        headers = self.get_headers(tr_id)
        params = {
            "AUTH": "",          # 사용자 권한정보 (일반적으로 빈 문자열)
            "EXCD": exchange,    # 거래소 코드 (예: 'NASD', 'NYSE')
            "SYMB": symbol,      # 종목코드 (예: 'AAPL')
            "NMIN": "5",         # 분갭: 5분봉
            "PINC": "Y",         # 전일 포함 여부 (Y = 포함)
            "NEXT": "0",         # 첫 페이지
            "NREC": str(count),  # 요청 갯수 (최대치는 KIS 문서 참고)
            "FILL": "0",         # 미체결 채움구분
            "KEYB": ""           # NEXT KEY BUFF (첫 호출은 빈값)
        }

        res = requests.get(url, headers=headers, params=params)

        try:
            data = res.json()
        except Exception as e:
            print("❌ get_us_minute_ohlcv_5m JSON 실패:", e, res.text)
            return pd.DataFrame()

        if res.status_code != 200 or data.get("rt_cd") != "0":
            print(
                "❌ get_us_minute_ohlcv_5m 실패:",
                res.status_code, data.get("rt_cd"), data.get("msg1"), exchange, symbol
            )
            return pd.DataFrame()

        # output2 안에 캔들 리스트
        if "output2" not in data or not data["output2"]:
            print("❌ get_us_minute_ohlcv_5m output2 비어있음:", exchange, symbol)
            return pd.DataFrame()

        df = pd.DataFrame(data["output2"])

        # 문서 기준 컬럼:
        # tymd, xymd, xhms, kymd, khms, open, high, low, last, evol, eamt
        try:
            df = df[[
                "kymd",   # 한국기준일자
                "khms",   # 한국기준시간
                "open",
                "high",
                "low",
                "last",
                "evol",
            ]]
        except KeyError:
            print("⚠️ get_us_minute_ohlcv_5m 컬럼 매핑 실패, 실제 컬럼:", df.columns.tolist())
            return pd.DataFrame()

        # 한국 기준 일자/시간을 하나의 datetime 으로 묶어서 index 로 사용
        df["datetime"] = pd.to_datetime(df["kymd"] + df["khms"])
        df.set_index("datetime", inplace=True)

        df = df.rename(columns={
            "open": "open",
            "high": "high",
            "low": "low",
            "last": "close",
            "evol": "volume",
        })

        df = df[["open", "high", "low", "close", "volume"]].astype(float)

        # 시간 순으로 정렬 + 필요 개수만 사용
        df = df.sort_index()
        if len(df) > count:
            df = df.iloc[-count:]

        return df


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
            "FID_INPUT_DATE_1": (datetime.now() - timedelta(days=100)).strftime("%Y%m%d"),
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
        if not self.access_token:
            self.auth()
        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/dailyprice"
        headers = self.get_headers("HHDFS76240000")
        params = {
            "AUTH": "",
            "EXCD": exchange,
            "SYMB": symbol,
            "GUBN": "0",
            "BYMD": datetime.now().strftime("%Y%m%d"),
            "MODP": "1"
        }
        res = requests.get(url, headers=headers, params=params)
        if res.status_code == 200 and res.json().get('rt_cd') == '0':
            df = pd.DataFrame(res.json()['output2'])
            df = df[['xymd', 'open', 'high', 'low', 'clos', 'tvol']]
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df.astype(float).sort_index()
        return pd.DataFrame()
