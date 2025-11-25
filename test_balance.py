# test_balance.py
import requests
import json
from kis_api import KisDataFetcher
from config import APP_KEY, APP_SECRET, ACCOUNT_NO, MODE

def debug_balance():
    print("🔍 예수금/잔고 정밀 분석 시작...\n")
    
    # API 객체 생성
    fetcher = KisDataFetcher(APP_KEY, APP_SECRET, ACCOUNT_NO, mode=MODE)
    fetcher.auth() # 토큰 발급
    
    # ---------------------------------------------------
    # 1. 원화 주문 가능 금액 (TTTC8908R) - 현재 사용 중인 API
    # ---------------------------------------------------
    print("📋 [1. 원화 주문가능금액 API (TTTC8908R)]")
    url = f"{fetcher.base_url}/uapi/domestic-stock/v1/trading/inquire-psbl-order"
    headers = fetcher.get_headers("VTTC8908R" if MODE == "virtual" else "TTTC8908R")
    params = {
        "CANO": fetcher.acc_no_prefix,
        "ACNT_PRDT_CD": fetcher.acc_no_suffix,
        "PDNO": "005930", "ORD_UNPR": "0", "ORD_DVSN": "01",
        "CMA_EVLU_AMT_ICLD_YN": "Y", "OVRS_ICLD_YN": "N"
    }
    res = requests.get(url, headers=headers, params=params)
    data = res.json()
    
    if res.status_code == 200 and data.get('rt_cd') == '0':
        out = data['output']
        print(f"   👉 ord_psbl_cash (주문가능현금): {out.get('ord_psbl_cash')} 원")
        print(f"   👉 nrcvb_buy_amt (미수없는매수): {out.get('nrcvb_buy_amt')} 원")
        print(f"   👉 max_buy_amt (최대매수금액): {out.get('max_buy_amt')} 원")
    else:
        print(f"   ❌ 조회 실패: {data.get('msg1')}")

    print("-" * 40)

    # ---------------------------------------------------
    # 2. 계좌 잔고 상세 (TTTC8434R) - 예수금 총액 확인용
    # ---------------------------------------------------
    print("📋 [2. 계좌 잔고 상세 API (TTTC8434R)]")
    url = f"{fetcher.base_url}/uapi/domestic-stock/v1/trading/inquire-balance"
    headers = fetcher.get_headers("VTTC8434R" if MODE == "virtual" else "TTTC8434R")
    params = {
        "CANO": fetcher.acc_no_prefix,
        "ACNT_PRDT_CD": fetcher.acc_no_suffix,
        "AFHR_FLPR_YN": "N", "OFL_YN": "N", "INQR_DVSN": "02", "UNPR_DVSN": "01",
        "FUND_STTL_ICLD_YN": "N", "FNCG_AMT_AUTO_RDPT_YN": "N", "PRCS_DVSN": "01",
        "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""
    }
    res = requests.get(url, headers=headers, params=params)
    data = res.json()

    if res.status_code == 200 and data.get('rt_cd') == '0':
        out2 = data['output2'][0] # 계좌 자산 정보
        print(f"   👉 dnca_tot_amt (예수금총액): {out2.get('dnca_tot_amt')} 원")
        print(f"   👉 nxdy_exkz_amt (익일정산금): {out2.get('nxdy_exkz_amt')} 원")
        print(f"   👉 prvs_rcdl_excc_amt (가수금): {out2.get('prvs_rcdl_excc_amt')} 원")
        print(f"   👉 thdt_buy_amt (금일매수금액): {out2.get('thdt_buy_amt')} 원")
        print(f"   👉 tot_evlu_amt (총평가금액): {out2.get('tot_evlu_amt')} 원")
    else:
        print(f"   ❌ 조회 실패: {data.get('msg1')}")

    print("-" * 40)
    print("💡 팁: 위 항목 중 149,000원(앱에 보이는 금액)과 일치하는 필드 이름을 찾으세요.")

if __name__ == "__main__":
    debug_balance()