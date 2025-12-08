import requests
import pandas as pd


def get_spot_24hr():
    url = "https://api.binance.com/api/v3/ticker/24hr"
    data = requests.get(url).json()

    spot = []
    for item in data:
        if item["symbol"].endswith("USDT"):
            spot.append({
                "market": "spot",
                "symbol": item["symbol"],
                "quoteVolume": float(item["quoteVolume"])
            })
    return spot


def get_futures_24hr():
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    data = requests.get(url).json()

    futures = []
    for item in data:
        if item["symbol"].endswith("USDT"):
            futures.append({
                "market": "futures",
                "symbol": item["symbol"],
                "quoteVolume": float(item["quoteVolume"])
            })
    return futures


# -------------------------------------------------------
# ⭐ Spot & Futures 각각 90개씩 가져오기
# -------------------------------------------------------
def make_top_spot_90():
    spot = get_spot_24hr()
    df = pd.DataFrame(spot)
    df = df.sort_values("quoteVolume", ascending=False).head(90)

    return [
        {"region": "BI", "symbol": row.symbol, "market": "spot"}
        for row in df.itertuples()
    ]


def make_top_futures_90():
    fut = get_futures_24hr()
    df = pd.DataFrame(fut)
    df = df.sort_values("quoteVolume", ascending=False).head(90)

    return [
        {"region": "BI", "symbol": row.symbol, "market": "futures"}
        for row in df.itertuples()
    ]


# -------------------------------------------------------
# ⭐ Spot 90 + Futures 90 그대로 합치기 (중복 허용)
# -------------------------------------------------------
def merge_keep_duplicates(spot_list, futures_list):
    return spot_list + futures_list


if __name__ == "__main__":
    spot90 = make_top_spot_90()
    fut90 = make_top_futures_90()

    final_universe = merge_keep_duplicates(spot90, fut90)

    print(f"최종 유니버스 개수: {len(final_universe)}")
    for item in final_universe:
        print(f"{item},")
