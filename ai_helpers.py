# ai_helpers.py
import os
import json
from datetime import datetime

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# -----------------------------------------------------------
# 공통 설정 / 헬퍼
# -----------------------------------------------------------
FAST_MODEL = "gpt-4.1-mini"  # 짧은 코멘트/로그용
FULL_MODEL = "gpt-4.1"       # 브레인스토밍/리포트용


def _call_responses_model(prompt: str, model: str = FAST_MODEL, temperature: float = 0.7) -> str:
    """
    responses API 래퍼
    - 짧은 코멘트용에 주로 사용 (entry/exit 코멘트 등)
    """
    try:
        resp = client.responses.create(
            model=model,
            input=prompt,
            temperature=temperature,
        )
        return resp.output[0].content[0].text.strip()
    except Exception as e:
        # 트레이딩 로직이 죽지 않도록, 실패 시 짧은 기본 문구 반환
        return f"[AI 코멘트 생성 실패: {e}]"


def _call_chat_model(
    system_msg: str,
    user_msg: str,
    model: str = FULL_MODEL,
    temperature: float = 0.7,
) -> str:
    """
    chat.completions API 래퍼
    - 리포트/전략 아이디어처럼 구조화된 답변이 필요한 곳에 사용
    """
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"[AI 리포트 생성 실패: {e}]"


# ===========================================================
# 1) 진입 코멘트 (시장별)
# ===========================================================
def _build_entry_prompt_kr(context: dict) -> str:
    return f"""
너는 **한국 주식 단기 트레이딩 봇**의 '진입 로그를 정리해주는 어시스턴트'야.

아래는 방금 진입한 포지션의 컨텍스트야 (JSON):

{json.dumps(context, ensure_ascii=False, indent=2)}

요구사항:
1) 왜 이 구간에서 진입했는지(전략/지표 관점)를 2~3줄로 한글로 요약해줘.
2) 코스피/코스닥 단기매매 느낌을 살려서, 너무 장황하지 않게 핵심 조건 위주로 작성.
3) 말투는 '트레이딩 노트' 쓰듯이 담백하게.
"""


def _build_entry_prompt_us(context: dict) -> str:
    return f"""
너는 **미국 주식 단기 트레이딩 봇**의 '진입 로그를 정리해주는 어시스턴트'야.

아래는 방금 진입한 포지션의 컨텍스트야 (JSON):

{json.dumps(context, ensure_ascii=False, indent=2)}

요구사항:
1) 왜 이 구간에서 진입했는지(전략/지표 관점)를 2~3줄로 한글로 요약해줘.
2) 프리/애프터, 소형주 변동성, 거래대금 같은 미국 시장 특성을 한 줄 정도 반영해도 좋아.
3) 말투는 '트레이딩 노트' 쓰듯이 담백하게. 투자 권유처럼 들리지 않게.
"""


def _build_entry_prompt_cr(context: dict) -> str:
    """
    코인 전용 진입 프롬프트
    - context 안에 새 코인 모델에서 쓰는 필드(예: strategy_name, cr_swing_proba, atr 등)가 있으면
      모델이 자연스럽게 그걸 언급해줄 수 있게 JSON 그대로 보여준다.
    """
    return f"""
너는 **코인(암호화폐) 단기 트레이딩 봇**의 '진입 로그를 정리해주는 어시스턴트'야.

아래는 방금 진입한 포지션의 컨텍스트야 (JSON):

{json.dumps(context, ensure_ascii=False, indent=2)}

요구사항:
1) 이 진입이 어떤 전략/시나리오에 기반한 것인지 2~3줄로 요약해줘.
   - strategy_name, entry_signal, ml_proba, cr_swing_proba, ATR 관련 값이 있다면
     지금 구간을 어떻게 해석했고, 앞으로 어떤 흐름(상승/조정/박스)을 기대하는지까지 짧게 써줘.
2) 손절/익절 기준(대략적인 %나 구간)을 "기록용 계획" 느낌으로 한 줄에 정리해줘.
3) 말투는 '트레이딩 노트' 느낌으로 담백하게, 투자 권유는 하지 말 것.
4) 나중에 청산 로그에서 이 진입 코멘트를 참고할 수 있도록,
   핵심 가정(예: '단기 반등 구간', '레인지 상단 돌파 기대' 등)을 한 문장으로 또렷하게 남겨줘.
"""

def _build_entry_prompt_bi(context: dict) -> str:
    """
    코인 전용 진입 프롬프트
    - context 안에 새 코인 모델에서 쓰는 필드(예: strategy_name, cr_swing_proba, atr 등)가 있으면
      모델이 자연스럽게 그걸 언급해줄 수 있게 JSON 그대로 보여준다.
    """
    return f"""
너는 **코인(암호화폐) 단기 트레이딩 봇**의 '진입 로그를 정리해주는 어시스턴트'야.

아래는 방금 진입한 포지션의 컨텍스트야 (JSON):

{json.dumps(context, ensure_ascii=False, indent=2)}

요구사항:
1) 이 진입이 어떤 전략/시나리오에 기반한 것인지 2~3줄로 요약해줘.
   - strategy_name, entry_signal, ml_proba, cr_swing_proba, ATR 관련 값이 있다면
     지금 구간을 어떻게 해석했고, 앞으로 어떤 흐름(상승/조정/박스)을 기대하는지까지 짧게 써줘.
2) 손절/익절 기준(대략적인 %나 구간)을 "기록용 계획" 느낌으로 한 줄에 정리해줘.
3) 말투는 '트레이딩 노트' 느낌으로 담백하게, 투자 권유는 하지 말 것.
4) 나중에 청산 로그에서 이 진입 코멘트를 참고할 수 있도록,
   핵심 가정(예: '단기 반등 구간', '레인지 상단 돌파 기대' 등)을 한 문장으로 또렷하게 남겨줘.
"""


def make_entry_comment_kr(context: dict) -> str:
    prompt = _build_entry_prompt_kr(context)
    return _call_responses_model(prompt, model=FAST_MODEL, temperature=0.4)


def make_entry_comment_us(context: dict) -> str:
    prompt = _build_entry_prompt_us(context)
    return _call_responses_model(prompt, model=FAST_MODEL, temperature=0.4)


def make_entry_comment_cr(context: dict) -> str:
    prompt = _build_entry_prompt_cr(context)
    return _call_responses_model(prompt, model=FAST_MODEL, temperature=0.4)

def make_entry_comment_bi(context: dict) -> str:
    prompt = _build_entry_prompt_bi(context)
    return _call_responses_model(prompt, model=FAST_MODEL, temperature=0.4)


def make_entry_comment(context: dict) -> str:
    """
    기존 공용 엔트리 코멘트 (호환용 라우터)
    - context["region"] 값(KR/US/CR/COIN)에 따라 시장별 함수 호출
    """
    region = (context.get("region") or "").upper()
    if region == "KR":
        return make_entry_comment_kr(context)
    elif region == "US":
        return make_entry_comment_us(context)
    elif region in ("CR", "COIN"):
        return make_entry_comment_cr(context)
    elif region in ("BI", "COIN"):
        return make_entry_comment_bi(context)
    # region이 명시 안 되어 있으면 일단 KR 스타일로 처리
    return make_entry_comment_kr(context)


# ===========================================================
# 2) 청산 코멘트 (시장별)
# ===========================================================
def _build_exit_prompt_kr(context: dict) -> str:
    return f"""
너는 **한국 주식 단기 트레이딩 봇**의 '청산 로그를 정리해주는 어시스턴트'야.

아래는 방금 청산한 포지션 정보야 (JSON):

{json.dumps(context, ensure_ascii=False, indent=2)}

요구사항:
1) 이번 청산이 어떤 상황에서 어떤 의도로 이루어진 건지 2~3줄로 정리해줘.
2) 수익/손실의 원인을 한 줄 정도로 짚어줘. (예: 시가갭 이후 눌림, 추세 잘 탄 케이스 등)
3) 다음에 같은 상황이 오면 어떻게 대응하면 좋을지 짧게 한 줄로 제안해줘.
4) 말투는 '하루 트레이딩 복기 노트' 느낌으로 담백하게.
"""


def _build_exit_prompt_us(context: dict) -> str:
    return f"""
너는 **미국 주식 단기 트레이딩 봇**의 '청산 로그를 정리해주는 어시스턴트'야.

아래는 방금 청산한 포지션 정보야 (JSON):

{json.dumps(context, ensure_ascii=False, indent=2)}

요구사항:
1) 이번 청산이 어떤 상황/장세(프리/애프터, 뉴욕 본장 등)에서 어떤 의도로 이루어진 건지 2~3줄로 정리해줘.
2) 수익/손실의 원인을 한 줄 정도로 짚어줘. (예: 저유동 소형주, 뉴스 변동성, 추세 이어진 케이스 등)
3) 다음에 같은 상황이 오면 어떻게 대응하면 좋을지 짧게 한 줄로 제안해줘.
4) 말투는 '하루 트레이딩 복기 노트' 느낌으로 담백하게.
"""


def _build_exit_prompt_cr(context: dict) -> str:
    return f"""
너는 **코인 단기 트레이딩 봇**의 '청산 로그를 정리해주는 어시스턴트'야.

아래는 방금 청산한 포지션 정보와 (있다면) 진입 당시 코멘트야 (JSON):

{json.dumps(context, ensure_ascii=False, indent=2)}

주의:
- context 안에 entry_comment, entry_strategy_name, entry_ml_proba 같은 필드가 있을 수 있어.
- 있다면 반드시 그 정보를 참고해서, "진입 때의 계획/가정"과 "실제 전개/결과"를 비교해줘.

요구사항:
1) 먼저, 진입 당시 어떤 시나리오/전략을 기대했는지 한 줄로 요약해줘.
   - entry_comment 나 entry_strategy_name, entry_ml_proba, cr_swing_proba, ATR 관련 값이 있다면
     "당시에는 어떤 흐름을 기대하고 들어갔는지"를 짧게 정리해줘.
2) 그 다음, 실제로 포지션이 어떻게 전개되었는지(추세/박스/횡보/급락 등)와
   이번 청산(TP/SL/TIMEOUT/ML_TAKE_PROFIT 등)이
   그 초기 시나리오와 어떻게 달라졌는지 위주로 2줄 내외로 설명해줘.
   - 특히, 진입 시 가정이 빗나간 지점이나 예상보다 좋았던 부분을 짚어줘.
3) 수익/손실의 핵심 원인을 한 줄 정도로 정리해줘.
   - 예: 변동성 과대, 예상보다 약한 반등, 레인지 상단 실패, 손절 기준 도달 등.
4) 마지막으로, "같은 전략/상황이 다시 왔을 때" 어떤 부분을 보완하면 좋을지
   (진입 타이밍, 손절 폭, 분할 진입/청산, 쿨다운 등) 한 줄로 제안해줘.
5) 전체 말투는 '하루 트레이딩 복기 노트' 느낌으로 담백하게,
   투자 권유나 확신 표현(무조건, 반드시 등)은 피하고 기록/피드백 위주로 작성해줘.
"""


def make_exit_comment_kr(context: dict) -> str:
    prompt = _build_exit_prompt_kr(context)
    return _call_responses_model(prompt, model=FAST_MODEL, temperature=0.5)


def make_exit_comment_us(context: dict) -> str:
    prompt = _build_exit_prompt_us(context)
    return _call_responses_model(prompt, model=FAST_MODEL, temperature=0.5)


def make_exit_comment_cr(context: dict) -> str:
    prompt = _build_exit_prompt_cr(context)
    return _call_responses_model(prompt, model=FAST_MODEL, temperature=0.5)


def make_exit_comment(context: dict) -> str:
    """
    기존 공용 청산 코멘트 (호환용 라우터)
    - context["region"] 값(KR/US/CR/COIN)에 따라 시장별 함수 호출
    """
    region = (context.get("region") or "").upper()
    if region == "KR":
        return make_exit_comment_kr(context)
    elif region == "US":
        return make_exit_comment_us(context)
    elif region in ("CR", "COIN"):
        return make_exit_comment_cr(context)
    return make_exit_comment_kr(context)


# ===========================================================
# 3) 하루 장 종료 후 전략 아이디어 브레인스토밍 (시장 구분 지원)
# ===========================================================
def brainstorm_strategy_ideas(context: dict, market: str | None = None) -> str:
    """
    장 마감 후, 통계/패턴 요약을 기반으로
    새로운 전략 아이디어를 브레인스토밍하는 함수.
    market: "KR" / "US" / "COIN" / None(전체)
    """
    date_str = context.get("date", "알 수 없는 날짜")
    market_str_map = {
        None: "전체 시장",
        "ALL": "전체 시장",
        "KR": "한국 주식",
        "US": "미국 주식",
        "CR": "코인",
        "COIN": "코인",
    }
    market_str = market_str_map.get((market or "ALL"), "전체 시장")

    user_prompt = f"""
아래는 {date_str} 기준 {market_str} 트레이딩 데이터에서 뽑아낸 통계 요약이야.
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

JSON 데이터:
{json.dumps(context, ensure_ascii=False, indent=2)}
"""
    system_msg = "너는 실전 단기매매를 돕는 퀀트/알고리즘 트레이딩 리서처다."
    return _call_chat_model(system_msg, user_prompt, model=FULL_MODEL, temperature=0.9)


# ===========================================================
# 4) 하루 일일 리포트 (시장 구분 지원)
# ===========================================================
def make_daily_trade_report_v2(context: dict, market: str | None = None) -> str:
    """
    v2 일일 리포트: trades + signals + ohlcv 요약까지 반영.
    market: "KR" / "US" / "COIN" / None(전체)
    """
    date_str = context.get("date", "알 수 없는 날짜")
    market_str_map = {
        None: "전체 시장",
        "ALL": "전체 시장",
        "KR": "한국 주식",
        "US": "미국 주식",
        "CR": "코인",
        "COIN": "코인",
    }
    market_str = market_str_map.get((market or "ALL"), "전체 시장")

    user_prompt = f"""
너는 단기 트레이더의 **일일 코치이자 리뷰어**야.

아래 JSON은 {date_str} 하루 동안의 {market_str} 트레이드 결과를,
숫자 + 시그널 + 차트 요약(pre_window/post_window)까지 합쳐서 정리한 데이터야.

각 필드 의미:
- stats: 하루 전체 성과
- trade_details: 트레이드별 상세 정보 (signal, pre_window, post_window 포함)

이 데이터를 바탕으로,
**마치 실제 차트를 다 보고 평가하는 것처럼** 아래 항목을 한국어로 정리해줘.

[1] 오늘 전체 성과 한 줄 총평
[2] 통계 기반 요약
[3] 트레이드 패턴 분석
[4] 베스트 / 워스트 트레이드 2~3개씩 선정
[5] 오늘의 오류 패턴 TOP 3
[6] 내일부터 실행할 구체적인 행동 가이드 5가지
    - 코드/룰로 바로 넣을 수 있는 수준의 규칙 위주로.

JSON 데이터:
{json.dumps(context, ensure_ascii=False, indent=2)}
"""
    system_msg = (
        "너는 실전 단기 트레이더를 돕는 코치이자 리뷰어다. "
        "숫자와 차트 요약을 바탕으로 구체적인 행동 가이드를 제안한다."
    )
    return _call_chat_model(system_msg, user_prompt, model=FULL_MODEL, temperature=0.7)


# ===========================================================
# 5) 모델 업데이트 조언 (시장 구분 지원)
# ===========================================================
def make_model_update_advice(context: dict, market: str | None = None) -> str:
    """
    active vs candidate 모델의 백테스트 + 최근 실전 성과를 비교해서
    - 모델 교체 여부
    - threshold/룰 조정
    - A/B 테스트 플랜
    을 제안하는 함수.
    market: "KR" / "US" / "COIN" / None(전체)
    """
    date_str = context.get("date", "알 수 없는 날짜")
    market_str_map = {
        None: "전체 시장",
        "ALL": "전체 시장",
        "KR": "한국 주식",
        "US": "미국 주식",
        "CR": "코인",
        "COIN": "코인",
    }
    market_str = market_str_map.get((market or "ALL"), "전체 시장")

    user_prompt = f"""
너는 {market_str}용 단기매매 ML 모델을 운용하는 **퀀트 리서처 + 리스크 매니저**야.

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
[2] 모델 교체 여부 판단 (A/B/C 중 하나 + 근거)
[3] ml_threshold 및 운용 규칙 조정 제안
[4] 내일부터 바로 적용할 설정 변경 체크리스트 (settings key/value 형태 의식)
[5] 실험 플랜 (A/B 테스트 방안)

JSON 데이터:
{json.dumps(context, ensure_ascii=False, indent=2)}
"""
    system_msg = (
        "너는 단기매매 ML 전략을 운용하는 퀀트 리서처이자 리스크 매니저다. "
        "숫자와 설정을 바탕으로 구체적인 액션 플랜을 제시한다."
    )
    return _call_chat_model(system_msg, user_prompt, model=FULL_MODEL, temperature=0.7)
