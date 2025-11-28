# zz_top200.py
"""
빗썸 KRW 마켓 기준
24시간 누적 거래대금(acc_trade_value_24H) 상위 N개를 뽑아서

1) 콘솔에 TOP 20 요약 출력
2) 전체 리스트를 JSON 파일로 저장
3) config.py에 붙여넣기 좋은 CR_UNIVERSE_STOCKS 예시 블록 출력

사용법:
    python zz_top200.py
"""

import json
from pathlib import Path
from typing import List, Dict

import requests

# -------------------------
# 설정값
# -------------------------
LIMIT = 200          # 상위 몇 개까지 가져올지
MIN_24H_KRW = 0      # 최소 24h 거래대금 필터 (예: 1_000_000_000 → 10억 이상만)


OUTPUT_JSON = Path("cr_universe_bithumb_top200.json")


def fetch_top_krw_markets_by_24h_volume(
    limit: int = 200,
    min_24h_krw: float = 0.0,
) -> List[Dict]:
    """
    빗썸 KRW 마켓 기준
    24시간 누적 거래대금(acc_trade_value_24H) 상위 N개 마켓을 반환.

    반환 예시:
    [
        {
            "symbol": "KRW-BTC",
            "trade_price": 98100000.0,
            "acc_trade_price_24h": 123456789000.0,
            "acc_trade_volume_24h": 1234.5678
        },
        ...
    ]
    """
    base_url = "https://api.bithumb.com"
    url = f"{base_url}/public/ticker/ALL_KRW"

    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"❌ [BITHUMB TOP 조회 실패] {e}")
        return []

    status = data.get("status")
    if status != "0000":
        print(f"⚠️ [BITHUMB TOP] status={status} raw={data}")
        return []

    ticker_dict = data.get("data", {})
    result_list: List[Dict] = []

    # ticker_dict 예시: {"BTC": {...}, "ETH": {...}, ..., "date": "timestamp"}
    for currency, item in ticker_dict.items():
        # "date" 같은 특수 키 제외
        if currency.upper() in ("DATE",):
            continue

        try:
            # 🔹 빗썸 실제 필드 이름
            # acc_trade_value_24H : 24시간 누적 거래대금 (KRW)
            # units_traded_24H    : 24시간 누적 거래량
            acc_trade_value_24h = float(item.get("acc_trade_value_24H", "0") or "0")
            acc_trade_volume_24h = float(item.get("units_traded_24H", "0") or "0")
            trade_price = float(item.get("closing_price", "0") or "0")
        except Exception:
            continue

        # 거래대금이 0이거나 최소 조건 미달이면 스킵
        if acc_trade_value_24h <= min_24h_krw:
            continue

        symbol = f"KRW-{currency.upper()}"

        result_list.append(
            {
                "symbol": symbol,
                "trade_price": trade_price,
                "acc_trade_price_24h": acc_trade_value_24h,
                "acc_trade_volume_24h": acc_trade_volume_24h,
            }
        )

    # 24h 거래대금 기준 내림차순 정렬
    result_list.sort(key=lambda x: x["acc_trade_price_24h"], reverse=True)

    # 상위 N개만 반환
    if limit is not None and limit > 0:
        result_list = result_list[:limit]

    print(
        f"📊 [BITHUMB TOP] 상위 {len(result_list)}개 마켓 반환 "
        f"(limit={limit}, min_24h_krw={min_24h_krw:,.0f})"
    )

    return result_list


def main():
    top_list = fetch_top_krw_markets_by_24h_volume(
        limit=LIMIT,
        min_24h_krw=MIN_24H_KRW,
    )

    if not top_list:
        print("⚠️ 결과가 비어 있습니다.")
        return

    # 1) 콘솔에 상위 20개만 요약 출력
    print("\n===== TOP 마켓 (상위 20개 요약) =====")
    for i, item in enumerate(top_list[:20], start=1):
        sym = item["symbol"]
        price = item["trade_price"]
        krw_24h = item["acc_trade_price_24h"]
        print(f"{i:3d}. {sym:10s} | 가격: {price:,.4f} | 24h 거래대금: {krw_24h:,.0f} KRW")

    # 2) JSON 파일로 전체 저장
    OUTPUT_JSON.write_text(
        json.dumps(top_list, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\n💾 JSON 저장 완료: {OUTPUT_JSON.resolve()}")

    # 3) config.py에 붙여넣기 좋은 CR_UNIVERSE_STOCKS 형태 출력
    print("\n===== config.py용 CR_UNIVERSE_STOCKS 예시 =====\n")
    print("CR_UNIVERSE_STOCKS = [")
    for item in top_list:
        print(f'    {{"region": "CR", "symbol": "{item["symbol"]}"}},')
    print("]")
    print("\n👉 위 블록을 config.py에 그대로 붙여넣으면 됩니다.\n")


if __name__ == "__main__":
    main()
