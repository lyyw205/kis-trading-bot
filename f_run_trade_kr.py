# run_kr_trade.py
from c_trade_runner import run_realtime_bot

if __name__ == "__main__":
    # 한국 주식만
    run_realtime_bot(region="KR")
