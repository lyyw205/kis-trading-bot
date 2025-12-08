import requests
import json

# ==========================================
# 🛑 계좌번호 "앞 8자리"만 입력하세요 (하이픈 제외)
# ==========================================
APP_KEY = "PSYIRWHM6bWGIbflRXkOocumwNDcG0zdKxub"
APP_SECRET = "HJZ1+Fqz5pV84Clc05c4LD+YrdfviMQU90XpgUj2cVYAsGobMJnn29VSsuLDqJQb+RvPUn4iOy61rSP6AnBGXrqito2g/ZkgSBUHWXFbjG55osDQ5WiesUbfZ9ROcNuhi74M5GpwxPpXEK3J+lfF/pCj0itHCB+zBTPEjEvy3b0Z7GBo3Bk="
ACC_PREFIX = "43522038"  # 예: "12345678"
MODE = "real"

def find_my_dollars():
    print(f"🕵️‍♂️ 계좌 앞자리 [{ACC_PREFIX}]에 연결된 모든 상세계좌를 뒤집니다...\n")
    
    base_url = "https://openapi.koreainvestment.com:9443"
    
    # 1. 토큰 발급
    res = requests.post(f"{base_url}/oauth2/tokenP", json={
        "grant_type": "client_credentials", "appkey": APP_KEY, "appsecret": APP_SECRET
    })
    token = res.json()["access_token"]

    # 2. 흔한 뒷자리 후보군 모두 조회
    suffixes = ["01", "21", "22", "02", "19", "51"]
    
    found = False
    
    for suffix in suffixes:
        headers = {
            "content-type": "application/json", "authorization": f"Bearer {token}",
            "appkey": APP_KEY, "appsecret": APP_SECRET, "tr_id": "TTTS3012R"
        }
        params = {
            "CANO": ACC_PREFIX, "ACNT_PRDT_CD": suffix,
            "OVRS_EXCG_CD": "NASD", "TR_CRCY_CD": "USD",
            "CTX_AREA_FK100": "", "CTX_AREA_NK100": ""
        }
        
        try:
            res = requests.get(f"{base_url}/uapi/overseas-stock/v1/trading/inquire-balance", headers=headers, params=params)
            data = res.json()
            
            if data['rt_cd'] == '0':
                usd = float(data['output2']['frcr_dncl_amt_2'])      # 외화예수금
                orderable = float(data['output2']['ovrs_ord_psbl_amt']) # 주문가능금액
                
                print(f"👉 끝자리 [-{suffix}] 조회 결과: 예수금 ${usd} / 주문가능 ${orderable}")
                
                if usd > 0 or orderable > 0:
                    print(f"\n🎉 찾았습니다! 돈은 [-{suffix}] 계좌에 있습니다!")
                    print(f"✅ 코드의 ACCOUNT_NO를 '{ACC_PREFIX}-{suffix}' 로 수정하세요.")
                    found = True
        except:
            pass

    if not found:
        print("\n❌ 여전히 못 찾았습니다. 앱에서 '체결내역'을 다시 확인해주세요.")

if __name__ == "__main__":
    find_my_dollars()