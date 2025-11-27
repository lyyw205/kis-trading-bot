# ai_helpers.py
import os
import json
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------
# 1) 진입 코멘트
# -----------------------------
def make_entry_comment(context: dict) -> str:
    """
    context 예시:
    {
      "time": "2025-11-27 05:15:01",
      "region": "US",
      "symbol": "AEHL",
      "exchange": "NAS",
      "side": "BUY",
      "qty": 3,
      "price": 3.40,
      "ml_proba": 0.71,
      "strategy": "MOMENTUM_STRONG",
      "rsi": 62.3,
      "note": "REVERSAL" or "MOMENTUM_STRONG" 등
    }
    """
    prompt = f"""
너는 단기 트레이딩 봇의 '진입 로그를 정리해주는 어시스턴트'야.

아래는 방금 진입한 포지션의 컨텍스트야:

{json.dumps(context, ensure_ascii=False, indent=2)}

요구사항:
1) 왜 이 구간에서 진입했는지(전략/지표 관점) 한글로 2~3줄로 요약해줘.
2) 너무 장황하게 쓰지 말고, 핵심 조건 위주로.
3) 말투는 '트레이딩 노트' 쓰듯이 담백하게.
"""

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )
    return resp.output[0].content[0].text.strip()


# -----------------------------
# 2) 청산/AI 코멘트
# -----------------------------
def make_exit_comment(context: dict) -> str:
    """
    context 예시:
    {
      "time": "2025-11-27 05:27:29",
      "region": "US",
      "symbol": "AEHL",
      "exchange": "NAS",
      "side": "SELL",
      "qty": 3,
      "avg_entry": 3.40,
      "exit_price": 3.47,
      "pnl_pct": 2.06,
      "reason": "PROFIT_3%",  # or CUT_LOSS, TIMEOUT_NO_TP 등
      "holding_minutes": 12.3
    }
    """
    prompt = f"""
너는 단기 트레이딩 봇의 '청산 로그를 정리해주는 어시스턴트'야.

아래는 방금 청산한 포지션 정보야:

{json.dumps(context, ensure_ascii=False, indent=2)}

요구사항:
1) 이번 청산이 어떤 상황에서 어떤 의도로 이루어진 건지 2~3줄로 정리해줘.
2) 수익/손실의 원인을 한 줄 정도로 짚어줘. (예: 너무 늦게 진입, 추세 잘 탄 케이스 등)
3) 다음에 같은 상황이 오면 어떻게 대응하면 좋을지 짧게 한 줄로 제안해줘.
4) 말투는 '하루 트레이딩 복기 노트' 느낌으로 담백하게.
"""

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=prompt,
    )
    return resp.output[0].content[0].text.strip()



# -----------------------------
# 3) 하루 장 종료 후 전략 아이디어 브레인스토밍
# -----------------------------
def brainstorm_strategy_ideas(context: dict) -> str:
    """
    context 예시:
    {
      "date": "2025-11-27",
      "overall": {
        "total_trades": 120,
        "win_rate": 0.43,
        "total_pnl": -3.4
      },
      "by_symbol": [
        {
          "symbol": "AAPL",
          "trades": 20,
          "win_rate": 0.55,
          "avg_pnl": 0.35,
          "note": "큰 변동성, 방향성 깔끔"
        },
        {
          "symbol": "IONQ",
          "trades": 15,
          "win_rate": 0.20,
          "avg_pnl": -0.8,
          "note": "테마주 / 뉴스 민감"
        }
      ],
      "by_time_block": [
        {
          "block": "US_PRIME_1H",  # 장 시작 후 1시간
          "trades": 30,
          "win_rate": 0.50,
          "avg_pnl": 0.2
        },
        {
          "block": "US_LATE",      # 장 마감 2시간
          "trades": 25,
          "win_rate": 0.28,
          "avg_pnl": -0.5
        }
      ],
      "by_pattern": [
        {
          "pattern": "첫 눌림 매수",
          "trades": 25,
          "win_rate": 0.6,
          "avg_pnl": 0.4
        },
        {
          "pattern": "고점 추격 돌파",
          "trades": 18,
          "win_rate": 0.22,
          "avg_pnl": -0.9
        }
      ],
      "model_notes": "최근 3일간 ML confidence > 0.7 인데도 손실 나는 케이스가 늘어나는 중"
    }
    """

    prompt = f"""
너는 단기 트레이딩 전략 연구소의 **리서치 담당 AI**야.

아래는 최근 트레이드/학습 데이터에서 뽑아낸 통계 요약이야.
이 데이터를 바탕으로 **새로운 전략 아이디어**를 브레인스토밍해줘.

요구사항:
- 단순히 "손절을 타이트하게" 이런 뻔한 얘기 말고,
  실제로 코드/전략으로 구현 가능한 아이디어 위주로 정리해줘.
- 크게 4파트로 나눠서 설명해줘:

[1] 지금 전략에서 '강점'으로 보이는 부분
    - 어떤 종목/시간대/패턴에서 성과가 좋은지
    - 그 강점을 더 크게 활용할 구체적인 아이디어

[2] '약점' 및 피해야 할 구간
    - 손실이 반복되는 조건 (예: 저유동성 + 점상/점하락, 뉴스 테마, 장 후반 등)
    - 완전 제외/축소/필터링 아이디어

[3] ML·데이터 관점 개선 아이디어
    - feature를 어떻게 추가/변형할지
    - 라벨링/타겟을 어떻게 바꿀지 (예: 단순 수익률 대신, '변동성 대비 효율', '평균 MFE/MAE' 등)
    - 실시간 운용시 confidence를 어떻게 활용/제한할지

[4] 내일부터 바로 실험해볼 수 있는 전략 실험 3~5개
    - 각 실험마다:
      - 실험 이름
      - 간단한 설명
      - 대략적인 조건 (코드화 가능하게)
      - 기대 효과
      - 체크해야 할 리스크

데이터(JSON):
{json.dumps(context, ensure_ascii=False, indent=2)}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "너는 실전 단기매매를 돕는 퀀트/알고리즘 트레이딩 리서처다."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
    )

    return resp.choices[0].message.content.strip()

def make_daily_trade_report_v2(context: dict) -> str:
    """
    v2 일일 리포트: trades + signals + ohlcv 요약까지 반영.

    context 예시:
    {
      "date": "2025-11-27",
      "stats": {
        "total_trades": 12,
        "total_profit": 125000,
        "win_rate": 41.7,
        "avg_profit": 10416.7,
        "max_profit": 83000,
        "max_loss": -42000
      },
      "trade_details": [
        {
          "id": 1,
          "symbol": "AAPL",
          "type": "BUY",
          "entry_time": "2025-11-27T22:05:00",
          "entry_price": 180.5,
          "qty": 3,
          "profit": 24000,
          "source": "KIS",
          "entry_comment": "돌파 후 눌림 진입",
          "exit_comment": "목표가 근처에서 절반 청산",
          "signal": {
            "region": "US",
            "at_support": 0,
            "is_bullish": 1,
            "price_up": 1,
            "lookback": 40,
            "band_pct": 0.14,
            "has_stock": 0,
            "entry_signal": 1,
            "ml_proba": 0.78,
            "entry_allowed": 1,
            "note": "MOMENTUM_STRONG"
          },
          "pre_window": {
            "bars": 20,
            "trend_pct": 3.4,
            "volatility_pct": 1.2
          },
          "post_window": {
            "bars": 20,
            "mfe_pct": 4.1,
            "mae_pct": -1.8,
            "close_pct_after_n": 2.7
          }
        },
        ...
      ]
    }

    ※ trade_details는 daily_ai_reports_v2.py(또는 네가 만든 advanced context 빌더)에서 생성.
    """
    date_str = context.get("date", "알 수 없는 날짜")
    stats = context.get("stats", {})
    trade_details = context.get("trade_details", [])

    prompt = f"""
너는 단기 트레이더의 **일일 코치이자 리뷰어**야.

아래 JSON은 {date_str} 하루 동안의 트레이드 결과를,
숫자 + 시그널 + 차트 요약(pre_window/post_window)까지 합쳐서 정리한 데이터야.

각 필드 의미:
- stats: 하루 전체 성과
  - total_trades: 총 트레이드 수
  - total_profit: 총 손익(원)
  - win_rate: 승률(%)
  - avg_profit: 트레이드당 평균 손익(원)
  - max_profit / max_loss: 개별 트레이드의 최대 이익/손실(원)

- trade_details: 트레이드별 상세 정보
  - symbol / type / entry_time / entry_price / qty / profit: 기본 체결 정보
  - signal: 진입 시점의 상태 (룰 기반 + ML 기반)
    - at_support, is_bullish, price_up, band_pct, has_stock, entry_signal,
      ml_proba, entry_allowed, note(전략 이름 등)
  - pre_window: 진입 전 차트 요약
    - trend_pct: 직전 N캔들 동안의 가격 변화율(%)
    - volatility_pct: 직전 N캔들의 평균 변동성(%)
  - post_window: 진입 후 차트 요약 (미래 차트)
    - mfe_pct: 진입 후 최고가 기준 최대 유리 구간(%)
    - mae_pct: 진입 후 최저가 기준 최대 불리 구간(%)
    - close_pct_after_n: N캔들 뒤 종가 기준 수익률(%)

이 데이터를 바탕으로,
**마치 실제 차트를 다 보고 평가하는 것처럼** 아래 항목을 한국어로 정리해줘.

[1] 오늘 전체 성과 한 줄 총평
    - 오늘 장의 난이도 / 운 vs 실력 느낌 / 감정적으로 주의해야 할 포인트

[2] 통계 기반 요약
    - 총 트레이드 수, 승률, 총 손익, 트레이드당 평균 손익
    - max_profit / max_loss가 의미하는 바 (위험 관리 측면 포함)

[3] 트레이드 패턴 분석
    - signal + pre_window + post_window를 함께 보고,
      "어떤 조건에서 수익이 잘 나고, 어떤 조건에서 반복적으로 손실이 났는지"를 정리
    - 예를 들어:
      - ml_proba가 높았는데도 손실이 나는 공통 상황
      - at_support=1인데도 깨지는 패턴
      - trend_pct가 너무 과열된 상태에서 진입한 패턴 등

[4] 베스트 / 워스트 트레이드 2~3개씩 선정
    - 각 트레이드마다:
      - 왜 좋은/나쁜 트레이드였는지 (차트 흐름 + 시그널 + 미래 결과 관점에서)
      - "같은 상황이 온다면 내일부터 어떻게 행동해야 하는지"까지 연결

[5] 오늘의 오류 패턴 TOP 3
    - 예시:
      - 진입은 괜찮았지만, MFE 대비 너무 일찍/늦게 청산한 경우
      - pre_window에서 이미 과열(상승 과대)인데 추격 진입한 경우
      - ml_proba는 낮았는데 entry_allowed=1로 진입한 위험한 패턴 등

[6] 내일부터 실행할 구체적인 행동 가이드 5가지
    - "가능하면 지키면 좋은 원칙"이 아니라
      ***정말로 코드/룰로 바로 넣을 수 있는 수준의 규칙***으로 적어줘.
      (예: 'ml_proba < 0.6 이면서 trend_pct > +5%인 경우는 진입 금지' 같은 식)

JSON 데이터:
{json.dumps(context, ensure_ascii=False, indent=2)}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "너는 실전 단기 트레이더를 돕는 코치이자 리뷰어다. 숫자와 차트 요약을 바탕으로 구체적인 행동 가이드를 제안한다."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    return resp.choices[0].message.content.strip()


def make_model_update_advice(context: dict) -> str:
    """
    매일 재학습 + 백테스트 결과를 비교해서
    - 새 모델을 채택할지 말지
    - ml_threshold / 종목 필터 / 운용 룰을 어떻게 손볼지
    에 대한 조언을 생성하는 함수.

    context 예시:
    {
      "date": "2025-11-27",
      "active": {
        "model_id": 3,
        "name": "SEQ_V1",
        "version": "1.0.3",
        "created_at": "2025-11-20 03:12:00",
        "backtest": {
          "period": "2025-11-01 ~ 2025-11-26",
          "trades": 420,
          "win_rate": 45.2,
          "avg_profit": 1100.0,
          "cum_return": 12.5,
          "max_dd": -4.3
        }
      },
      "candidate": {
        "model_id": 5,
        "name": "SEQ_V2",
        "version": "1.1.0",
        "created_at": "2025-11-27 04:20:00",
        "backtest": {
          "period": "2025-11-10 ~ 2025-11-26",
          "trades": 380,
          "win_rate": 48.9,
          "avg_profit": 1300.0,
          "cum_return": 16.8,
          "max_dd": -6.1
        }
      },
      "live_stats": {
        "recent_days": 3,
        "trades": 52,
        "win_rate": 38.5,
        "avg_profit": -800.0,
        "cum_profit": -41600.0
      },
      "settings": {
        "ml_threshold": 0.65,
        "max_positions": 3
      }
    }
    """
    date_str = context.get("date", "알 수 없는 날짜")

    prompt = f"""
너는 단기매매용 ML 모델을 운용하는 **퀀트 리서처 + 리스크 매니저**야.

아래 JSON은:
- 현재 실계좌에 사용 중인 active 모델의 백테스트 성과
- 오늘 새로 학습한 candidate 모델의 백테스트 성과
- 최근 실시간 운용 결과(live_stats)
- 현재 설정값(settings: ml_threshold 등)
을 한 번에 정리한 데이터야.

목표:
- 새 모델을 바로 교체할지 말지,
- 교체한다면 어떤 조건으로 쓸지(ml_threshold, 종목/시간대 필터),
- 당장 코드/설정에 반영할 수 있는 액션 아이템을 정리하는 것.

아래 형식으로 한국어로 답변해줘.

[1] 상황 요약
    - active vs candidate 백테스트 비교 요약
    - 최근 live 성과까지 포함해서, '지금 모델이 겪는 상태'를 한 줄로 정리

[2] 모델 교체 여부 판단
    - (A) 그대로 active 유지
    - (B) candidate로 전면 교체
    - (C) 일부 조건에서만 candidate 사용 (예: 특정 종목/시간대/전략)
    셋 중 하나를 추천하고, 근거를 구체적인 숫자 기반으로 설명해줘.

[3] ml_threshold 및 운용 규칙 조정 제안
    - 예시:
      - ml_threshold 상향/하향
      - 특정 구간(예: 변동성 과도, max_dd 구간)에서는 진입 제한
      - 하루 최대 트레이드 수 / 연속 손실 제한 등

[4] 내일부터 바로 적용할 설정 변경 체크리스트
    - "설정 키" + "추천 값" + "적용 이유" 형태로 bullet로 정리
    - settings 테이블에 들어갈 key/value 형태를 의식해서 작성해줘.
      (예: "ml_threshold": 0.7, "max_positions": 2 …)

[5] 실험 플랜
    - candidate 모델을 바로 전면 교체하지 않는다면,
      앞으로 3~7일 동안 어떤 방식으로 A/B 테스트를 할지
      (예: 심볼 일부에서만 candidate 사용, 소액/가상 계정에서 테스트 등)

JSON 데이터:
{json.dumps(context, ensure_ascii=False, indent=2)}
"""

    resp = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "system",
                "content": "너는 단기매매 ML 전략을 운용하는 퀀트 리서처이자 리스크 매니저다. 숫자와 설정을 바탕으로 구체적인 액션 플랜을 제시한다."
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
    )

    return resp.choices[0].message.content.strip()
