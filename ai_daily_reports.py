#  AI 일일 트레이드 리포트 & 전략 아이디어 생성 스크립트 (v2)

#  - 트레이드/시그널 DB를 기반으로 LLM에 넣을 컨텍스트를 만들고,
#    일간 리포트 + 전략 브레인스토밍 결과를 자동 생성/저장하는 엔트리 스크립트.

# 주요 기능:
# 1) target_date 기준으로 트레이드 데이터 로드 (load_trades_for_date)
# 2) build_daily_context_v2()로 일일 성과 요약 컨텍스트 생성
# 3) make_daily_trade_report_v2() 호출 → AI 일일 리포트 텍스트 생성
# 4) build_brainstorm_context()로 전략 아이디어용 컨텍스트 생성
# 5) brainstorm_strategy_ideas() 호출 → 전략 개선 아이디어 텍스트 생성
# 6) reports/ 폴더에 날짜별 텍스트 파일로 저장
#    - {날짜}_daily_report_{market}.txt
#    - {날짜}_strategy_ideas_{market}.txt
# 7) BotDatabase.save_ai_report()로 ai_reports 테이블에 저장
# 8) 콘솔에 리포트/아이디어 내용을 프린트하여 확인

# ※ market_configs 설정에 따라 ALL/KR/US/COIN 등
#    시장별 리포트를 개별 생성할 수 있으며,
#    현재는 COIN(코인)만 활성화된 상태.


import os
from datetime import date

from c_db_manager import BotDatabase
from ai_helpers import (
    make_daily_trade_report_v2,
    brainstorm_strategy_ideas,
)
from ai_report_context import (
    load_trades_for_date,
    build_daily_context_v2,
    build_brainstorm_context,
)


if __name__ == "__main__":
    target_date = date.today()  # 필요하면 특정 날짜로 바꿔도 됨
    date_str = target_date.strftime("%Y-%m-%d")

    # reports 폴더 없으면 생성
    os.makedirs("reports", exist_ok=True)

    db = BotDatabase()

    # 시장 설정: 전체 + KR + US + COIN
    # - COIN 필터를 쓰되, 내부에서는 CR/COIN 둘 다 집계
    market_configs = [
        # (None, "ALL"),   # 전체
        # ("KR", "KR"),
        # ("US", "US"),
        ("COIN", "COIN"),
        ("BI", "BI")
    ]

    daily_reports = {}
    strategy_ideas_map = {}

    for region_key, label in market_configs:
        # 1) trades 로드
        df_trades = load_trades_for_date(target_date, region=region_key)

        # 2) 일일 리포트용 context + AI 호출
        daily_ctx = build_daily_context_v2(df_trades, target_date, region=region_key)
        daily_report = make_daily_trade_report_v2(daily_ctx, market=region_key)

        # 3) 브레인스토밍용 context + AI 호출
        brainstorm_ctx = build_brainstorm_context(df_trades, target_date, region=region_key)
        ideas = brainstorm_strategy_ideas(brainstorm_ctx, market=region_key)

        daily_reports[label] = daily_report
        strategy_ideas_map[label] = ideas

        # 4) 개별 텍스트 파일 저장
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

        # 5) ✅ DB에 region별로 바로 저장
        db.save_ai_report(
            date_str=date_str,
            daily_report=daily_report,
            strategy_ideas=ideas,
            region=label,   # 'ALL' / 'KR' / 'US' / 'COIN'
        )

        # 6) 콘솔 출력 (그대로 유지)
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
