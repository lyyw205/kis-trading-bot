from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd

from bi_exit_ms import decide_exit_ms
from bi_exit_rev import decide_exit_rev
from bi_exit_momo import decide_exit_momo
from bi_exit_short import decide_exit_ms_short


def decide_exit_cr(
    pos,
    df_5m: pd.DataFrame,
    cur_price: float,
    now_dt: datetime,
    strategy_name: Optional[str] = None,
    params_by_strategy: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    strategy_name 예시 (엔트리 쪽에서 사용하는 값 기준):

        - "MS"
        - "MS_SHORT"
        - "REV"
        - "MOMO"

    또는 과거/다른 버전:

        - "BI_MS_STRONG" / "BI_MS_NORMAL" / ...
        - "BI_MS_SHORT_..."
        - "BI_REV_ENTRY"
        - "BI_MOMO_ENTRY"
    """
    params_by_strategy = params_by_strategy or {}

    # -------------------------------------------------
    # 0) strategy_name 정규화 (None / 이상값 방어)
    # -------------------------------------------------
    raw_name: Optional[str] = strategy_name

    # pos 안에 strategy_name이 들어있으면 fallback 용으로 사용
    if not isinstance(raw_name, str):
        raw_name = getattr(pos, "strategy_name", None)

    if not isinstance(raw_name, str):
        # 여전히 없으면 그냥 빈 문자열로 두고,
        # 아래에서 MS 기본 룰로 fallback 되게 한다.
        strategy_key = ""
    else:
        strategy_key = raw_name.strip()

    # 대소문자 / prefix 섞여도 처리되게 통일
    sk = strategy_key.upper()

    # -------------------------------------------------
    # 1) prefix 기준으로 exit 함수 선택
    #    - "BI_MS_SHORT...", "MS_SHORT" 둘 다 숏으로 처리
    #    - "BI_MS...", "MS" → MS 롱
    #    - "BI_REV...", "REV..." → REV
    #    - "BI_MOMO...", "MOMO..." → MOMO
    # -------------------------------------------------
    if sk.startswith("BI_MS_SHORT") or sk.startswith("MS_SHORT"):
        # MS_SHORT 전용 숏 청산
        exit_fn = decide_exit_ms_short
        ex_params = params_by_strategy.get("MS_SHORT")

    elif sk.startswith("BI_MS") or sk == "MS":
        # 기존 MS(Long) 청산
        exit_fn = decide_exit_ms
        ex_params = params_by_strategy.get("MS")

    elif sk.startswith("BI_REV") or sk.startswith("REV"):
        exit_fn = decide_exit_rev
        ex_params = params_by_strategy.get("REV")

    elif sk.startswith("BI_MOMO") or sk.startswith("MOMO"):
        exit_fn = decide_exit_momo
        ex_params = params_by_strategy.get("MOMO")

    else:
        # 모르면 일단 MS 룰로 처리 (fallback)
        # sk가 ""인 경우(전략 이름 전혀 모를 때)도 여기로 들어옴
        exit_fn = decide_exit_ms
        ex_params = params_by_strategy.get("MS")

    result = exit_fn(
        pos=pos,
        df_5m=df_5m,
        cur_price=cur_price,
        now_dt=now_dt,
        params=ex_params,
    )

    if "debug" not in result:
        result["debug"] = {}

    # (원하면 여기서 strategy_name 도 같이 넣어서 상위에서 로그 찍을 수 있게)
    result["debug"].setdefault("strategy_name", strategy_key)

    return result