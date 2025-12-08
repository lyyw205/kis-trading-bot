# test_binance_conn.py
from bi_client import BinanceDataFetcher

# ----------------------------------------
# 1. 키 입력 (여기에 실제 키를 넣어서 테스트)
# ----------------------------------------
MY_API_KEY = "YOUR_API_KEY_HERE"
MY_SECRET_KEY = "YOUR_SECRET_KEY_HERE"

def test():
    print("🚀 바이낸스 연결 테스트 시작...")
    
    # 브로커 생성
    fetcher = BinanceDataFetcher(
        api_key=MY_API_KEY, 
        secret_key=MY_SECRET_KEY, 
        mode="real"
    )

    # 1. 심볼 형식 테스트 (BTCUSDT)
    print("\n1️⃣ 현재가 조회 테스트 (BTCUSDT)")
    price = fetcher.get_coin_current_price("BTCUSDT")
    if price:
        print(f"   ✅ 성공! BTCUSDT 가격: {price}")
    else:
        print("   ❌ 실패! 심볼 형식이 틀렸거나 IP 차단일 수 있음.")

    # 2. 계좌 잔고 테스트
    print("\n2️⃣ 잔고 조회 테스트")
    balance = fetcher.get_coin_balance()
    if balance is not None:
        print(f"   ✅ 성공! 조회된 자산 개수: {len(balance)}")
        print(f"   내용: {balance}")
    else:
        print("   ❌ 실패! API Key/Secret을 확인하세요.")

if __name__ == "__main__":
    test()