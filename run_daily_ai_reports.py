# daily_ai_reports.py
import os
from datetime import date

from db_manager import BotDatabase
from ai_helpers import (
    make_daily_trade_report_v2,
    brainstorm_strategy_ideas,
)
from ai_report_context import (
    DB_PATH,
    load_trades_for_date,
    build_daily_context_v2,
    build_brainstorm_context,
)


if __name__ == "__main__":
    target_date = date.today()  # 필요하면 특정 날짜로 바꿔도 됨
    date_str = target_date.strftime("%Y-%m-%d")

    # reports 폴더 없으면 생성
    os.makedirs("reports", exist_ok=True)

    db = BotDatabase(DB_PATH)

    # 시장 설정: 전체 + KR + US + CR
    market_configs = [
        (None, "ALL"),  # 전체
        ("KR", "KR"),
        ("US", "US"),
        ("CR", "CR"),
    ]

    daily_reports = {}
    strategy_ideas_map = {}

    for region_key, label in market_configs:
        # ---------------------------------
        # 1) trades 로드 (region 기준 필터)
        # ---------------------------------
        df_trades = load_trades_for_date(target_date, region=region_key)

        # ---------------------------------
        # 2) 일일 리포트용 context + AI 호출
        # ---------------------------------
        daily_ctx = build_daily_context_v2(df_trades, target_date)
        daily_report = make_daily_trade_report_v2(daily_ctx, market=region_key)

        # ---------------------------------
        # 3) 브레인스토밍용 context + AI 호출
        # ---------------------------------
        brainstorm_ctx = build_brainstorm_context(df_trades, target_date, region=region_key)
        ideas = brainstorm_strategy_ideas(brainstorm_ctx, market=region_key)

        daily_reports[label] = daily_report
        strategy_ideas_map[label] = ideas

        # ---------------------------------
        # 4) 개별 텍스트 파일 저장
        # ---------------------------------
        suffix = label.lower()
        with open(
            f"reports/{date_str}_daily_report_{suffix}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(daily_report)

        with open(
            f"reports/{date_str}_strategy_ideas_{suffix}.txt",
            "w",
            encoding="utf-8",
        ) as f:
            f.write(ideas)

        # ---------------------------------
        # 5) 콘솔 출력 (원하면 생략 가능)
        # ---------------------------------
        print("\n========================")
        if label == "ALL":
            print("📊 [전체] 일일 트레이드 리포트 (v2)")
        else:
            print(f"📊 [{label}] 일일 트레이드 리포트 (v2)")
        print("========================\n")
        print(daily_report)

        print("\n========================")
        if label == "ALL":
            print("🧠 [전체] 전략 아이디어 브레인스토밍")
        else:
            print(f"🧠 [{label}] 전략 아이디어 브레인스토밍")
        print("========================\n")
        print(ideas)

    # ---------------------------------
    # 6) DB에는 "통합 텍스트"로 한 번 저장
    #    (ai_reports 테이블 스키마 안 바꾸는 방향)
    # ---------------------------------
    combined_daily = []
    combined_ideas = []

    order = ["ALL", "KR", "US", "CR"]

    for label in order:
        combined_daily.append(f"[{label}]\n{daily_reports.get(label, '')}\n")
        combined_ideas.append(f"[{label}]\n{strategy_ideas_map.get(label, '')}\n")

    combined_daily_text = "\n\n".join(combined_daily)
    combined_ideas_text = "\n\n".join(combined_ideas)

    db.save_ai_report(
        date_str=date_str,
        daily_report=combined_daily_text,
        strategy_ideas=combined_ideas_text,
    )
