# bi_config.py 혹은 c_config.py

# 메이저 코인 50개 (심볼)
BI_MAJOR_SYMBOLS = [
    "BTCUSDT",
    "ETHUSDT",
    "BNBUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "TRXUSDT",
    "LINKUSDT",

    "DOTUSDT",
    "LTCUSDT",
    "ATOMUSDT",
    "NEARUSDT",
    "UNIUSDT",
    "XLMUSDT",
    "FILUSDT",
    "APTUSDT",
    "AAVEUSDT",

    "SANDUSDT",
    "MANAUSDT",
    "THETAUSDT",
    "EGLDUSDT",
    "ICPUSDT",
    "VETUSDT",
    "HBARUSDT",
    "ARBUSDT",
    "OPUSDT",
    "INJUSDT",
    "STXUSDT",

    "IMXUSDT",
    "RUNEUSDT",
    "GRTUSDT",
    "FLOWUSDT",
    "CHZUSDT",
    "XTZUSDT",
    "KAVAUSDT",

    "GMXUSDT",
    "DYDXUSDT",
    "BLURUSDT",
    "LDOUSDT",
    "ENSUSDT",
    "COMPUSDT",
    "RSRUSDT",
    "ONDOUSDT",
    "JTOUSDT",
    "PYTHUSDT",
]

# 공통 유니버스: Spot + Futures 둘 다 운용한다고 가정
BI_TARGET_STOCKS = (
    # Spot
    [
        {"region": "BI", "symbol": symbol, "market": "spot"}
        for symbol in BI_MAJOR_SYMBOLS
    ]
    +
    # Futures
    [
        {"region": "BI", "symbol": symbol, "market": "futures"}
        for symbol in BI_MAJOR_SYMBOLS
    ]
)

# 기존 코드 호환용
BI_UNIVERSE_STOCKS = BI_TARGET_STOCKS

BI_SPOT_UNIVERSE_STOCKS = [
    t for t in BI_TARGET_STOCKS
    if t.get("region") == "BI" and t.get("market") == "spot"
]

BI_FUTURES_UNIVERSE_STOCKS = [
    t for t in BI_TARGET_STOCKS
    if t.get("region") == "BI" and t.get("market") == "futures"
]
