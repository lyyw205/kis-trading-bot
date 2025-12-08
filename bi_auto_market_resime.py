# run_update_market_regime_coin.py
# 시장 상승세, 하락세 6시간마다 확인 스케줄러 (추후 서버 연결 시 자동 실행 용 / 지금은 사용안함)

from c_db_manager import BotDatabase
from bi_market_regime import update_market_regime_coin
from bi_extract_top200 import make_top150  # 네가 만든 함수 이름에 맞게 import

def main():
    db = BotDatabase()

    # BI 상위 150개 심볼 가져오기
    top_list = make_top150()   # [{"region":"BI","symbol":"BTCUSDT"}, ...]
    top_symbols = [item["symbol"] for item in top_list]

    info = update_market_regime_coin(db, top_symbols)

    print("✅ market_regime_coin 업데이트 완료")
    print(f"  regime = {info['regime']}, score = {info['score']}, breadth={info['breadth']:.2f}")
    print(f"  cond = {info['cond_detail']}")

if __name__ == "__main__":
    main()
