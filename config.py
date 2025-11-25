# config.py

# ⚠️ 여기에 네 실제 키/계좌 넣기
APP_KEY = "PS4O81PxUmBuHjfNABTfiuRFP06eYqulanDt"
APP_SECRET = "mcO3Qtqq+3cSwbFTWNSV4c0NLP3tdJQ2ABjim1xTZ+BfJt+gL+oKkAUhuXczqr7L2lTdP4Da3T8Dk+O9STBMfUVZXmHsNVqp2V5KCGrkfuF9MNyx8s2sJJ9wONda50V3Y0Vapp3q86RL3aeY33ec4yRnrsO15EITPZN3cejDGSuFO8F3O6w="
ACCOUNT_NO = "43522038-01"  # 예: "12345678-01"
MODE = "real"  # or "virtual"

TARGET_STOCKS = [
  {"region": "US", "symbol": "TSLA", "excd": "NAS"},
  {"region": "US", "symbol": "AAPL", "excd": "NAS"},
]

AI_PARAMS = {
    "lookback": 120,
    "band_pct": 0.01
}
