import os
import json
import time
import requests
import pandas as pd
from datetime import datetime, timedelta

class KisDataFetcher:
    def __init__(self, app_key, app_secret, acc_no, mode="real", logger=None):
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
        self.logger = logger or print
        
        if mode == "real":
            self.base_url = "https://openapi.koreainvestment.com:9443"
            self.tr_id_kr_buy = "TTTC0802U"
            self.tr_id_kr_sell = "TTTC0801U"
            self.tr_id_us_buy = "TTTT1002U"
            self.tr_id_us_sell = "TTTT1006U"
        else:
            self.base_url = "https://openapivts.koreainvestment.com:29443"
            self.tr_id_kr_buy = "VTTC0802U"
            self.tr_id_kr_sell = "VTTC0801U"
            self.tr_id_us_buy = "VTTT1002U"
            self.tr_id_us_sell = "VTTT1006U"
            
        self.access_token = None
        self.token_file = "kis_token_cache.json"

    def log(self, msg):
        try:
            self.logger(msg)
        except Exception:
            print(msg) 

    # ------------------------
    # 인증 / 공통 요청
    # ------------------------
    def auth(self, force=False):
        if not force and os.path.exists(self.token_file):
            try:
                with open(self.token_file, 'r') as f:
                    saved_data = json.load(f)
                saved_time = datetime.strptime(
                    saved_data['timestamp'],
                    "%Y-%m-%d %H:%M:%S"
                )
                if datetime.now() - saved_time < timedelta(minutes=50):
                    self.access_token = saved_data['access_token']
                    return
            except:
                pass

        self.log("🔑 토큰 재발급...")
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
            "custtype": "P",
        }

    def _request(self, method, url, tr_id, **kwargs):
        headers = self.get_headers(tr_id)
        res = requests.request(method, url, headers=headers, **kwargs)

        try:
            data = res.json()
        except:
            data = {}

        if res.status_code == 500 and data.get("msg_cd") == "EGW00123":
            self.log("♻️ 토큰 만료 감지 → 재발급 후 재시도")
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
            if res.status_code == 200 and data.get('rt_cd') == '0':
                return int(data['output']['ord_psbl_cash'])
        except Exception:
            pass
        return 0
    
    def get_us_buyable_cash(self):
        usd_cash = 0.0
        try:
            tr_id = "VTTS3018R" if self.mode == "virtual" else "TTTS3018R"
            url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-present-balance"
            headers = self.get_headers(tr_id)
            # 잔고 조회시에는 NASD가 보통 맞음
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
                for item in data.get('output2', []):
                    if item.get('crcy_cd') == 'USD':
                        usd_cash = float(item['frcr_dncl_amt_2'])
                        break
        except Exception:
            pass

        krw_cash = self.get_kr_buyable_cash()
        converted_usd = krw_cash / 1450
        total_buying_power = usd_cash + converted_usd

        self.log(f"💰 [해외 매수 가능] USD(${usd_cash}) + KRW환산(${converted_usd:.2f}) = 총 ${total_buying_power:.2f}")
        return total_buying_power

    def get_kr_balance(self):
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
        tr_id = "VTTC8434R" if self.mode == "virtual" else "TTTC8434R"

        res = self._request("GET", url, tr_id, params={
            "CANO": self.acc_no_prefix,
            "ACNT_PRDT_CD": self.acc_no_suffix,
            "AFHR_FLPR_YN": "N", "OFL_YN": "N", "INQR_DVSN": "02", "UNPR_DVSN": "01",
            "FUND_STTL_ICLD_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "01",
            "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""
        })

        stock_dict = {}
        try:
            data = res.json()
            if res.status_code == 200 and data.get('rt_cd') == '0':
                for item in data.get('output1', []):
                    if int(item['hldg_qty']) > 0:
                        stock_dict[item['pdno']] = {
                            'qty': int(item['hldg_qty']),
                            'avg_price': float(item['pchs_avg_pric'])
                        }
        except Exception:
            pass
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
            if res.status_code == 200 and data.get('rt_cd') == '0':
                for item in data.get('output1', []):
                    qty = float(item['ovrs_cblc_qty'])
                    if qty > 0:
                        stock_dict[item['ovrs_pdno']] = {
                            'qty': qty,
                            'avg_price': float(item['pchs_avg_pric'])
                        }
        except Exception:
            pass
        return stock_dict

    # ------------------------
    # 주문 관련
    # ------------------------
    def send_kr_order(self, symbol, order_type, qty):
        url = f"{self.base_url}/uapi/domestic-stock/v1/trading/order-cash"
        tr_id = self.tr_id_kr_buy if order_type == "buy" else self.tr_id_kr_sell
        headers = self.get_headers(tr_id)
        data = {
            "CANO": self.acc_no_prefix, "ACNT_PRDT_CD": self.acc_no_suffix,
            "PDNO": symbol, "ORD_DVSN": "01", "ORD_QTY": str(qty), "ORD_UNPR": "0"
        }
        res = requests.post(url, headers=headers, data=json.dumps(data))
        try:
            if res.status_code == 200 and res.json().get('rt_cd') == '0':
                self.log(f"✅ send_kr_order 성공: {order_type} {symbol} {qty}주")
                return True
        except:
            pass
        return False

    def send_us_order(self, exchange, symbol, order_type, qty, price):
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/order"
        tr_id = self.tr_id_us_buy if order_type == "buy" else self.tr_id_us_sell

        # [수정] 정규화 제거 -> 받은 exchange 그대로 사용
        ovrs_excg_cd = exchange

        order_price_str = f"{round(float(price), 4):.4f}"
        headers = self.get_headers(tr_id)
        body = {
            "CANO": self.acc_no_prefix,
            "ACNT_PRDT_CD": self.acc_no_suffix,
            "OVRS_EXCG_CD": ovrs_excg_cd,
            "PDNO": symbol,
            "ORD_QTY": str(qty),
            "OVRS_ORD_UNPR": order_price_str,
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00",
        }
        res = requests.post(url, headers=headers, data=json.dumps(body))
        try:
            if res.status_code == 200 and res.json().get("rt_cd") == "0":
                self.log(f"✅ send_us_order 성공: {order_type} {symbol}")
                return True
        except:
            pass
        self.log(f"❌ send_us_order 실패: {symbol}")
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
            price = res.json().get('output', {}).get('stck_prpr', '')
            return float(price.replace(',', '')) if price else None
        except:
            return None

    def get_us_current_price(self, exchange, symbol):
        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/price"
        
        # [수정] 정규화 제거 -> 받은 exchange 그대로 사용
        excd = exchange
        
        res = self._request("GET", url, "HHDFS76200200", params={
            "AUTH": "", "EXCD": excd, "SYMB": symbol
        })
        try:
            price = res.json().get('output', {}).get('last', '')
            return float(price.replace(',', '')) if price else None
        except:
            return None

    def get_minute_ohlcv_5m(self, symbol, count=200):
        if not self.access_token: self.auth()
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        headers = self.get_headers("FHKST03010200")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol,
            "FID_INPUT_HOUR_1": "", "FID_PW_DATA_INCU_YN": "N", "FID_TIME_INTERVAL": "5"
        }
        res = requests.get(url, headers=headers, params=params)
        try:
            data = res.json()
            if data.get("rt_cd") != "0": return pd.DataFrame()
            df = pd.DataFrame(data["output2"])
            df["datetime"] = pd.to_datetime(df["stck_bsop_date"] + df["stck_cntg_hour"])
            df = df.rename(columns={"stck_oprc":"open", "stck_hgpr":"high", "stck_lwpr":"low", "stck_prpr":"close", "cntg_vol":"volume"})
            df.set_index("datetime", inplace=True)
            df = df[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors='coerce').dropna()
            df = df.sort_index()
            return df.iloc[-count:] if len(df) > count else df
        except:
            return pd.DataFrame()
    
    # ========================
    # ② 해외 5분봉 OHLCV (페이지네이션 적용)
    # ========================
    def get_us_minute_ohlcv_5m(self, exchange, symbol, count=200):
        if self.mode == "virtual":
            return pd.DataFrame()

        if not self.access_token:
            self.auth()

        # [수정] 정규화 제거 (유저가 직접 "NASD", "NYSE" 사용)
        excd = exchange
        
        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/inquire-time-itemchartprice"
        tr_id = "HHDFS76950200"
        headers = self.get_headers(tr_id)
        
        df_list = []
        collected_count = 0
        next_key = ""

        max_loops = 30
        loop_cnt = 0

        while collected_count < count and loop_cnt < max_loops:
            loop_cnt += 1
            
            params = {
                "AUTH": "",
                "EXCD": excd,
                "SYMB": symbol,
                "NMIN": "5",
                "PINC": "Y",      # 전일 포함 (필수)
                "NEXT": "1" if next_key else "0",
                "NREC": "120",
                "FILL": "0",
                "KEYB": next_key  # ⭐ 여기가 핵심 (이전 데이터의 마지막 시간)
            }
            
            time.sleep(0.1)
            res = requests.get(url, headers=headers, params=params)

            try:
                data = res.json()
            except Exception:
                break

            if res.status_code != 200 or data.get("rt_cd") != "0":
                # 에러 로그는 남기지만, 이미 받은 데이터가 있다면 반환 시도
                break

            output2 = data.get("output2", [])
            if not output2:
                break

            # 데이터 변환
            chunk = pd.DataFrame(output2)
            chunk["datetime"] = pd.to_datetime(chunk["kymd"] + chunk["khms"])
            chunk.set_index("datetime", inplace=True)
            chunk = chunk.rename(columns={"last": "close", "evol": "volume"})
            chunk = chunk[["open", "high", "low", "close", "volume"]].astype(float)
            
            df_list.append(chunk)
            collected_count += len(chunk)

            # -----------------------------------------------------------
            # ⭐ [중요] 다음 페이지 키 생성 로직 변경
            # API가 주는 output1['next']는 단순히 "1"(있음) 또는 "0"(없음)일 수 있음.
            # 다음 조회를 위해서는 '마지막 데이터의 날짜+시간'을 KEYB로 줘야 함.
            # -----------------------------------------------------------
            output1 = data.get("output1", {})
            has_next = output1.get("next")  # "1" or "0"
            
            if has_next == "1" and len(output2) > 0:
                # API 응답(output2)은 최신 -> 과거 순서로 옴.
                # 따라서 리스트의 마지막 요소(-1)가 가장 과거 데이터.
                last_row = output2[-1]
                # KEYB 포맷: YYYYMMDDHHMMSS (초 단위까지)
                next_key = last_row['kymd'] + last_row['khms']
            else:
                # 더 이상 데이터 없음
                break

        if not df_list:
            return pd.DataFrame()

        # 병합 및 정렬
        df_all = pd.concat(df_list)
        df_all = df_all[~df_all.index.duplicated(keep='first')]
        df_all = df_all.sort_index()

        if len(df_all) > count:
            df_all = df_all.iloc[-count:]

        return df_all


    def get_ohlcv(self, region, symbol, exchange=None, interval="5m", count=200):
        if region == "KR":
            if interval == "5m":
                return self.get_minute_ohlcv_5m(symbol, count=count)
            elif interval == "1d":
                return self.get_daily_ohlcv(symbol)
        else:
            if interval == "5m":
                return self.get_us_minute_ohlcv_5m(exchange, symbol, count=count)
            elif interval == "1d":
                return self.get_overseas_daily_ohlcv(exchange, symbol)
        
        return pd.DataFrame()

    def get_daily_ohlcv(self, symbol):
        if not self.access_token: self.auth()
        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-daily-itemchartprice"
        headers = self.get_headers("FHKST03010100")
        params = {
            "FID_COND_MRKT_DIV_CODE": "J", "FID_INPUT_ISCD": symbol,
            "FID_INPUT_DATE_1": (datetime.now() - timedelta(days=200)).strftime("%Y%m%d"),
            "FID_INPUT_DATE_2": datetime.now().strftime("%Y%m%d"),
            "FID_PERIOD_DIV_CODE": "D", "FID_ORG_ADJ_PRC": "1"
        }
        res = requests.get(url, headers=headers, params=params)
        try:
            df = pd.DataFrame(res.json()['output2'])
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df.astype(float).sort_index()
        except:
            return pd.DataFrame()

    def get_overseas_daily_ohlcv(self, exchange, symbol):
        if not self.access_token: self.auth()
        
        # [수정] 정규화 제거
        excd = exchange
        
        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/dailyprice"
        headers = self.get_headers("HHDFS76240000")
        params = {
            "AUTH": "", "EXCD": excd, "SYMB": symbol,
            "GUBN": "0", "BYMD": datetime.now().strftime("%Y%m%d"), "MODP": "1"
        }
        res = requests.get(url, headers=headers, params=params)
        try:
            df = pd.DataFrame(res.json()['output2'])
            df = df[['xymd', 'open', 'high', 'low', 'clos', 'tvol']]
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            return df.astype(float).sort_index()
        except:
            return pd.DataFrame()