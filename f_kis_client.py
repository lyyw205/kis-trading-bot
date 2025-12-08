# KIS(한국투자증권) 전용 데이터/주문 클라이언트

# - 국내/해외 주식 공용 KIS REST API 래퍼
# - OAuth2 토큰 캐시(파일 저장) + 자동 재발급 로직 내장
# - 예수금/잔고/체결 내역 조회 + 주문 실행 + 시세/분봉/일봉 OHLCV까지 포함된 통합 클래스

# 주요 기능:
# 1) auth(), get_headers(), _request()                  : KIS 인증 토큰 발급/갱신, 공통 요청 래퍼
# 2) get_us_fills(), get_us_fills_normalized()          : 미국 주식 체결 내역(RAW/정규화 포맷) 조회
# 3) get_kr_buyable_cash(), get_us_buyable_cash()       : 국내/해외 매수 가능 금액 조회 (KRW→USD 환산 포함)
# 4) get_kr_balance(), get_us_balance()                 : 국내/해외 주식 보유 잔고 조회
# 5) send_kr_order(), send_us_order()                   : 국내/해외 주식 주문(매수/매도) 전송
# 6) get_kr_current_price(), get_us_current_price()     : 국내/해외 현재가 조회
# 7) get_minute_ohlcv_5m(), get_us_minute_ohlcv_5m()    : 국내/해외 5분봉 OHLCV 조회 (해외는 페이지네이션 포함)
# 8) get_daily_ohlcv(), get_overseas_daily_ohlcv()      : 국내/해외 일봉 OHLCV 조회 
# 9) get_ohlcv()                                        : region/interval에 따라 위 OHLCV 함수 통합 호출

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
    def get_us_fills(self, start_date, end_date, excd="NASD",
                     sll_buy_dvsn="00", ccld_nccs_dvsn="00",
                     ctx_area_nk200="", ctx_area_fk200=""):
        """
        방금 테스트해서 STATUS: 200 찍혔던 그 함수 (원본 JSON 반환)
        """
        url = f"{self.base_url}/uapi/overseas-stock/v1/trading/inquire-ccnl"
        tr_id = "VTTS3035R" if self.mode == "virtual" else "TTTS3035R"
        headers = self.get_headers(tr_id)

        params = {
            "CANO": self.acc_no_prefix,
            "ACNT_PRDT_CD": self.acc_no_suffix,
            "PDNO": "",
            "ORD_STRT_DT": start_date,
            "ORD_END_DT": end_date,
            "SLL_BUY_DVSN": sll_buy_dvsn,       # 00=전체, 01=매도, 02=매수
            "CCLD_NCCS_DVSN": ccld_nccs_dvsn,   # 00=전체, 01=체결, 02=미체결
            "OVRS_EXCG_CD": excd,               # NASD/NYSE/AMEX 등
            "SORT_SQN": "DS",                   # 내림차순
            "ORD_DT": "",
            "ORD_GNO_BRNO": "",
            "ODNO": "",
            "CTX_AREA_NK200": ctx_area_nk200,
            "CTX_AREA_FK200": ctx_area_fk200,
        }

        resp = requests.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()

    # 🔹 새로 추가: 우리 트레이드 포맷으로 변환
    def get_us_fills_normalized(self, start_date, end_date, excd="NASD"):
        raw = self.get_us_fills(
            start_date=start_date,
            end_date=end_date,
            excd=excd,
            sll_buy_dvsn="00",
            ccld_nccs_dvsn="00",
        )

        fills = []
        for row in raw.get("output", []):
            # 체결 수량 0이면 스킵
            ccld_qty = float(row.get("ft_ccld_qty", "0") or "0")
            if ccld_qty == 0:
                continue

            side_code = row.get("sll_buy_dvsn_cd", "")
            side = "BUY" if side_code == "02" else "SELL"

            # ord_dt(YYYYMMDD) + ord_tmd(HHMMSS) → datetime
            dt_str = (row.get("ord_dt", "") or "") + (row.get("ord_tmd", "") or "")
            try:
                dt = datetime.strptime(dt_str, "%Y%m%d%H%M%S")
            except ValueError:
                # 혹시 포맷이 이상하면 그냥 문자열 그대로 저장해도 됨
                dt = None

            fills.append(
                {
                    "time": dt,
                    "time_str": dt.strftime("%Y-%m-%d %H:%M:%S") if dt else dt_str,
                    "symbol": row.get("pdno"),
                    "type": side,
                    "price": float(row.get("ft_ccld_unpr3", "0") or "0"),
                    "qty": ccld_qty,
                    "excd": row.get("ovrs_excg_cd"),
                    "market_name": row.get("tr_mket_name"),
                    "order_no": row.get("odno"),
                    "orgn_order_no": row.get("orgn_odno"),
                    "status": row.get("prcs_stat_name"),
                    "currency": row.get("tr_crcy_cd"),
                    "source": row.get("mdia_dvsn_name"),  # 모바일 / OpenAPI 등
                }
            )

        return fills

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

        exchange_map = {
            "NAS": "NASD", 
            "NYS": "NYSE", 
            "AMS": "AMEX",
            "NASD": "NASD",
            "NYSE": "NYSE",
            "AMEX": "AMEX"
        }
        # 매핑에 없으면 입력값 그대로 사용 (안전장치)
        ovrs_excg_cd = exchange_map.get(exchange.upper(), exchange)

        # 3. 가격 포맷팅 (소수점 2자리 권장)
        order_price_str = f"{float(price):.2f}"
        
        headers = self.get_headers(tr_id)
        
        body = {
            "CANO": self.acc_no_prefix,
            "ACNT_PRDT_CD": self.acc_no_suffix,
            "OVRS_EXCG_CD": ovrs_excg_cd,  # 변환된 코드 사용 (NASD 등)
            "PDNO": symbol,
            "ORD_QTY": str(int(qty)),      # 수량은 정수 문자열
            "OVRS_ORD_UNPR": order_price_str,
            "ORD_SVR_DVSN_CD": "0",
            "ORD_DVSN": "00",              # 00: 지정가 (미국주식 필수)
        }
        
        try:
            res = requests.post(url, headers=headers, data=json.dumps(body))
            res_json = res.json()
            
            # 성공 여부 체크
            if res.status_code == 200 and res_json.get("rt_cd") == "0":
                self.log(f"✅ send_us_order 성공: {symbol} {order_type} {qty}주 (${order_price_str})")
                return True
            else:
                # 실패 시 구체적인 에러 메시지(msg1) 로그 출력
                error_msg = res_json.get("msg1", "알 수 없는 에러")
                self.log(f"❌ send_us_order 실패: {symbol} | 원인: {error_msg}")
                return False
                
        except Exception as e:
            self.log(f"❌ send_us_order 예외 발생: {symbol} | {e}")
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
        """
        국내 5분봉 OHLCV 조회 (REST주식당일분봉조회)
        - 실패 원인 디버깅 로그 강화 버전
        """
        if not self.access_token:
            self.auth()

        url = f"{self.base_url}/uapi/domestic-stock/v1/quotations/inquire-time-itemchartprice"
        tr_id = "FHKST03010200"   # 당일 분봉 조회 TR
        headers = self.get_headers(tr_id)

        # 기준 시간: 지금 시각(HHMMSS) -> 공식 예제들은 보통 '100000' 이런 식으로 고정값도 많이 씀
        now_hms = datetime.now().strftime("%H%M%S")

        params = {
            # 공식 예제 기준 세팅
            "FID_ETC_CLS_CODE": "00",        # 기타 분류 코드(주식/ETF/ETN: 00)
            "FID_COND_MRKT_DIV_CODE": "J",   # J: 주식/ETF/ETN
            "FID_INPUT_ISCD": symbol,        # 종목코드 (6자리)
            "FID_INPUT_HOUR_1": now_hms,     # 기준 시간 (HHMMSS)
            "FID_PW_DATA_INCU_YN": "N",      # 과거 데이터 포함 여부 (당일만이면 N)
            "FID_TIME_INTERVAL": "5",        # 5분봉
        }

        try:
            res = requests.get(url, headers=headers, params=params)

            # ✅ 1차: HTTP 레벨 에러 체크
            if res.status_code != 200:
                self.log(
                    f"❌ [KR 5분봉 HTTP에러] {symbol} "
                    f"status={res.status_code} body={res.text[:200]}"
                )
                return pd.DataFrame()

            # ✅ 2차: JSON 파싱
            try:
                data = res.json()
            except Exception as e:
                self.log(
                    f"❌ [KR 5분봉 JSON파싱 실패] {symbol} "
                    f"status={res.status_code} err={e} body={res.text[:200]}"
                )
                return pd.DataFrame()

            rt_cd = data.get("rt_cd", "")
            msg1 = data.get("msg1", "")

            # ✅ 3차: KIS 결과 코드 확인
            if rt_cd != "0":
                self.log(
                    f"⚠️ [KR 5분봉 실패] {symbol} rt_cd={rt_cd} msg1={msg1} "
                    f"raw={str(data)[:200]}"
                )
                return pd.DataFrame()

            candles = data.get("output2", [])
            if not candles:
                self.log(f"⚠️ [KR 5분봉 데이터없음] {symbol} output2 비어있음")
                return pd.DataFrame()

            # ✅ 4차: DataFrame 변환
            df = pd.DataFrame(candles)
            # stck_bsop_date: YYYYMMDD, stck_cntg_hour: HHMMSS
            df["datetime"] = pd.to_datetime(df["stck_bsop_date"] + df["stck_cntg_hour"])
            df = df.rename(
                columns={
                    "stck_oprc": "open",
                    "stck_hgpr": "high",
                    "stck_lwpr": "low",
                    "stck_prpr": "close",
                    "cntg_vol": "volume",
                }
            )
            df.set_index("datetime", inplace=True)

            df = df[["open", "high", "low", "close", "volume"]].apply(
                pd.to_numeric, errors="coerce"
            ).dropna()

            df = df.sort_index()

            if len(df) > count:
                df = df.iloc[-count:]

            return df

        except Exception as e:
            self.log(f"❌ [KR 5분봉 예외] {symbol} | {e}")
            return pd.DataFrame()
    
    # ========================
    # ② 해외 5분봉 OHLCV (페이지네이션 적용)
    # ========================
    def get_us_minute_ohlcv_5m(self, exchange, symbol, max_count=2000):
        if self.mode == "virtual":
            return pd.DataFrame()

        if not self.access_token:
            self.auth()

        excd = exchange
        url = f"{self.base_url}/uapi/overseas-price/v1/quotations/inquire-time-itemchartprice"
        tr_id = "HHDFS76950200"
        headers = self.get_headers(tr_id)

        df_list = []
        next_key = ""
        loop_cnt = 0
        max_loops = 50   # 너무 무한루프 안 가게

        while loop_cnt < max_loops:
            loop_cnt += 1

            params = {
                "AUTH": "",
                "EXCD": excd,
                "SYMB": symbol,
                "NMIN": "5",
                "PINC": "Y",   # 전일까지 포함
                "NEXT": "1" if next_key else "0",
                "NREC": "120",
                "FILL": "0",
                "KEYB": next_key,
            }

            time.sleep(0.1)
            res = requests.get(url, headers=headers, params=params)

            try:
                data = res.json()
            except Exception as e:
                self.log(f"❌ [US 5분봉 JSON파싱 실패] {exchange} {symbol} | {e}")
                break

            if res.status_code != 200 or data.get("rt_cd") != "0":
                self.log(
                    f"⚠️ [US 5분봉 실패] {exchange} {symbol} "
                    f"status={res.status_code} rt_cd={data.get('rt_cd')} msg1={data.get('msg1')}"
                )
                break

            output2 = data.get("output2", [])
            if not output2:
                self.log(f"⚠️ [US 5분봉 없음] {exchange} {symbol} output2 비어있음")
                break

            chunk = pd.DataFrame(output2)
            chunk["datetime"] = pd.to_datetime(chunk["kymd"] + chunk["khms"])
            chunk.set_index("datetime", inplace=True)
            chunk = chunk.rename(columns={"open": "open", "high": "high", "low": "low", "last": "close", "evol": "volume"})
            chunk = chunk[["open", "high", "low", "close", "volume"]].astype(float)

            df_list.append(chunk)

            output1_list = data.get("output1", [])
            if isinstance(output1_list, list) and len(output1_list) > 0:
                has_next = output1_list[0].get("next")
            else:
                has_next = None

            if has_next == "1":
                last_row = output2[-1]
                next_key = last_row["kymd"] + last_row["khms"]
            else:
                break

        if not df_list:
            return pd.DataFrame()

        df_all = pd.concat(df_list)
        df_all = df_all[~df_all.index.duplicated(keep="first")]
        df_all = df_all.sort_index()

        # 너무 많아지면 뒷부분만
        if len(df_all) > max_count:
            df_all = df_all.iloc[-max_count:]

        return df_all


    def get_ohlcv(self, region, symbol, exchange=None, interval="5m", count=200):
        if region == "KR":
            if interval == "5m":
                return self.get_minute_ohlcv_5m(symbol, count=count)
            elif interval == "1d":
                return self.get_daily_ohlcv(symbol)
        else:
            if interval == "5m":
                return self.get_us_minute_ohlcv_5m(exchange, symbol, max_count=count)
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