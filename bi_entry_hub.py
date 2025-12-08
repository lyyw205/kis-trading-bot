""" 코인 멀티전략 엔트리 허브 (tcn_entry_hub.py)

역할:
 - 여러 코인 엔트리 전략(MS / REV / MOMO / MS_SHORT)의
   점수(entry_score)를 '전략별 가중치'와 '컷 조건'을 반영해서
   1) 심볼별로 최적 전략을 고르고
   2) 유니버스 전체에서 최종 진입 대상 심볼을 1개 뽑는 허브 모듈.

주요 구성:

1) STRATEGY_SCORE_WEIGHTS
   - 전략별 점수 가중치 테이블
     · "MS": 1.2      (롱 MS 전략 버프)
     · "REV": 0.8     (리버설 전략 너프)
     · "MOMO": 1.0
     · "MS_SHORT": 1.3 (숏 MS 전략 강하게 버프)
   - raw entry_score × weight 로 weighted_score 계산.

2) evaluate_strategies_for_symbol(symbol, df_5m, strategies, params_by_strategy)
   - 입력:
       · symbol: 심볼명 (BTCUSDT 등)
       · df_5m : 해당 심볼의 5분봉 DataFrame
       · strategies: {"MS": func, "REV": func, ...}
     동작:
       1) 각 전략 함수(make_entry_signal_coin_ms/rev/momo/ms_short)를 실행
       2) entry_signal=True 이고 entry_score 있는 것만 후보로 필터
       3) 전략별 가중치를 곱한 weighted_score 계산
       4) weighted_score가 가장 높은 전략 1개를 선택
   - 출력:
       · has_entry: 진입 후보 있는지 여부
       · selected_strategy: 선택된 전략 이름
       · entry: 해당 전략이 반환한 dict (side, entry_score, note, etc.)
       · weighted_score: 가중치 반영 후 최종 점수
       · all_results: 모든 전략의 원본 결과(디버깅용)

3) pick_best_entry_across_universe(df_by_symbol, ...)
   - 입력:
       · df_by_symbol:
           {"BTCUSDT": df_btc_5m, "ETHUSDT": df_eth_5m, ...}
       · strategies: 전략 이름 → 함수 매핑 (미지정 시 기본값: MS/REV/MOMO,
         BEAR 레짐일 때만 MS_SHORT 추가)
       · min_final_score: 전체 공통 컷
       · per_strategy_min_score: 전략별 개별 컷 (없으면 공통 컷 사용)
       · market_regime: "BULL" / "BEAR" / "NEUTRAL" 등 (BEAR일 때 숏 전략 활성화)
   - 동작:
       1) 각 심볼에 대해 evaluate_strategies_for_symbol 호출
       2) raw_candidates: 컷 적용 전 후보 (심볼, 점수, 상세)
       3) per_strategy_min_score / min_final_score로 점수 컷 적용
       4) 컷 통과한 후보 중 weighted_score 최고 심볼을 최종 진입 대상으로 선택
   - 출력:
       · has_final_entry: 최종 진입 여부
       · symbol: 최종 선택된 심볼
       · strategy: 선정된 전략 이름
       · entry: 전략 결과 + symbol + final_score + entry_version 포함 dict
       · detail: 컷 적용 전 후보 리스트(raw_candidates)
       · reason: NO_SYMBOL_ENTRY / BEST_SCORE_TOO_LOW / OK 등
       · version: ENTRY_VERSION (엔트리 버전 태깅용)

특징:
 - 개별 전략 모듈(tcn_entry_ms / tcn_entry_rev / tcn_entry_momo / tcn_entry_ms_short)을
   묶어서 “엔트리 의사결정 중앙 허브” 역할을 함.
 - 실시간 트레이더, 백테스트 엔진에서 공통으로 사용 가능하도록
   브로커/거래소/자산군 디테일은 전혀 알지 않고,
   오로지 'df_5m 시계열 + 전략 함수들'만 다루는 구조.
"""

from typing import Dict, Any, Optional, Callable, List, Tuple
import pandas as pd

from bi_entry_lib import ENTRY_VERSION
from bi_entry_ms import DEFAULT_ENTRY_PARAMS_MS, make_entry_signal_coin_ms
from bi_entry_rev import make_entry_signal_coin_rev
from bi_entry_momo import make_entry_signal_coin_momo
from bi_entry_short import make_entry_signal_coin_ms_short


StrategyFunc = Callable[[pd.DataFrame, Optional[Dict[str, Any]]], Dict[str, Any]]

STRATEGY_SCORE_WEIGHTS = {
    "MS": 1.2,       # MS 롱은 살짝 버프
    "REV": 0.8,      # REV는 너프
    "MOMO": 1.0,
    "MS_SHORT": 1.3, # 숏은 더 버프
}

# -------------------------------------------------------------
# 1) 한 심볼 안에서: 전략들 중 '가중치 반영 점수' 최고 선택
# -------------------------------------------------------------
def evaluate_strategies_for_symbol(
    symbol: str,
    df_5m: pd.DataFrame,
    strategies: Dict[str, StrategyFunc],
    params_by_strategy: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    한 심볼에 대해 여러 전략을 돌려보고,
    '가중치가 반영된 최종 점수'가 가장 높은 전략 하나를 리턴합니다.
    """
    params_by_strategy = params_by_strategy or {}
    results: Dict[str, Dict[str, Any]] = {}

    # 1. 각 전략 실행
    for name, func in strategies.items():
        res = func(df_5m, params_by_strategy.get(name))
        results[name] = res

    # 2. 진입 시그널이 있는 것만 필터링
    candidates: List[Tuple[str, Dict[str, Any], float]] = []

    for name, res in results.items():
        if res.get("entry_signal") and res.get("entry_score") is not None:
            raw_score = float(res.get("entry_score", 0.0))
            weight = float(STRATEGY_SCORE_WEIGHTS.get(name, 1.0))
            weighted_score = raw_score * weight

            candidates.append((name, res, weighted_score))

    if not candidates:
        return {
            "symbol": symbol,
            "has_entry": False,
            "selected_strategy": None,
            "entry": None,
            "weighted_score": 0.0,
            "all_results": results,
        }

    # 3. 가중 점수 기준 최고 전략 선택
    best_name, best_res, best_weighted_score = max(
        candidates,
        key=lambda x: x[2],
    )

    return {
        "symbol": symbol,
        "has_entry": True,
        "selected_strategy": best_name,
        "entry": best_res,
        "weighted_score": best_weighted_score,
        "all_results": results,
    }


# -------------------------------------------------------------
# 2) 유니버스 전체: 최종 진입 코인 한 개 선택
# -------------------------------------------------------------
def pick_best_entry_across_universe(
    df_by_symbol: Dict[str, pd.DataFrame],
    strategies: Optional[Dict[str, StrategyFunc]] = None,
    params_by_strategy: Optional[Dict[str, Dict[str, Any]]] = None,
    min_final_score: Optional[float] = None,
    market_regime: Optional[str] = None,
    per_strategy_min_score: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    df_by_symbol: { "BTC": df_5m_btc, "ETH": df_5m_eth, ... }

    per_strategy_min_score:
        예) {
            "MS": 0.012,
            "MS_SHORT": 0.007,
            "REV": 0.02,
        }
        → 전략별로 서로 다른 컷 적용
    """
    # 기본 전략 셋
    if strategies is None:
        strategies = {
            "MS": make_entry_signal_coin_ms,
            "REV": make_entry_signal_coin_rev,
            "MOMO": make_entry_signal_coin_momo,
        }
        # 시장 레짐이 BEAR일 때만 숏 전략 추가 (원하면 항상 추가로 바꿔도 됨)
        if market_regime in ("BEAR", "NEUTRAL"):
            strategies["MS_SHORT"] = make_entry_signal_coin_ms_short

    params_by_strategy = params_by_strategy or {}
    if min_final_score is None:
        min_final_score = float(DEFAULT_ENTRY_PARAMS_MS["ms_min_final_score"])

    per_strategy_min_score = per_strategy_min_score or {}

    raw_candidates: List[Tuple[str, float, Dict[str, Any]]] = []
    filtered_candidates: List[Tuple[str, float, Dict[str, Any]]] = []

    # ------------------------------------
    # 1) 심볼별로 최고 전략 + 점수 계산
    # ------------------------------------
    for symbol, df in df_by_symbol.items():
        res_sym = evaluate_strategies_for_symbol(
            symbol=symbol,
            df_5m=df,
            strategies=strategies,
            params_by_strategy=params_by_strategy,
        )

        if not res_sym["has_entry"]:
            continue

        strategy_name = res_sym["selected_strategy"]
        final_score = float(res_sym["weighted_score"])

        raw_candidates.append((symbol, final_score, res_sym))

        # 전략별 컷(or 공통 컷) 적용
        eff_min = per_strategy_min_score.get(strategy_name, min_final_score)

        if final_score >= eff_min:
            filtered_candidates.append((symbol, final_score, res_sym))

    # ------------------------------------
    # 2) 어떤 심볼도 시그널이 없는 경우
    # ------------------------------------
    if not raw_candidates:
        return {
            "has_final_entry": False,
            "symbol": None,
            "strategy": None,
            "entry": None,
            "detail": [],
            "reason": "NO_SYMBOL_ENTRY",
            "version": ENTRY_VERSION,
        }

    # ------------------------------------
    # 3) 시그널은 있지만 컷을 통과한 애가 없는 경우
    # ------------------------------------
    if not filtered_candidates:
        # 디버깅용으로: 컷 전 후보 중 최고 점수 정보 제공
        best_symbol, best_score, best_res_sym = max(
            raw_candidates,
            key=lambda x: x[1],
        )
        best_strategy = best_res_sym["selected_strategy"]

        return {
            "has_final_entry": False,
            "symbol": None,
            "strategy": None,
            "entry": None,
            "detail": raw_candidates,
            "reason": (
                f"BEST_SCORE_TOO_LOW("
                f"{best_strategy}:{best_score:.4f}, "
                f"global_min={min_final_score:.4f}, "
                f"per_strategy_min={per_strategy_min_score.get(best_strategy, min_final_score):.4f}"
                ")"
            ),
            "version": ENTRY_VERSION,
        }

    # ------------------------------------
    # 4) 컷을 통과한 후보 중 최고 선택 + 선택 이유 생성
    # ------------------------------------
    filtered_candidates.sort(key=lambda x: x[1], reverse=True)
    best_symbol, best_score, best_res_sym = filtered_candidates[0]

    best_strategy = best_res_sym["selected_strategy"]
    best_entry = best_res_sym["entry"] or {}
    best_raw_score = float(best_entry.get("entry_score", best_score))

    # 이 전략에 적용된 컷 값 (없으면 공통 컷)
    best_cut = per_strategy_min_score.get(best_strategy, min_final_score)

    # 2위 후보 정보 (있으면 비교용으로 사용)
    if len(filtered_candidates) >= 2:
        second_symbol, second_score, second_res_sym = filtered_candidates[1]
        second_strategy = second_res_sym["selected_strategy"]
        diff = best_score - second_score

        selection_reason = (
            f"{best_symbol} / {best_strategy} 선택 이유: "
            f"가중점수 {best_score:.4f} (raw {best_raw_score:.4f})가 "
            f"해당 전략 컷({best_cut:.4f}) 이상이고, "
            f"2위 {second_symbol}/{second_strategy} (가중 {second_score:.4f}) 대비 "
            f"{diff:.4f}만큼 더 높아서 최종 진입 대상으로 선정."
        )
    else:
        selection_reason = (
            f"{best_symbol} / {best_strategy} 선택 이유: "
            f"가중점수 {best_score:.4f} (raw {best_raw_score:.4f})가 "
            f"해당 전략 컷({best_cut:.4f}) 이상이며, "
            f"컷을 통과한 유일한 후보라서 최종 진입 대상으로 선정."
        )

    final_entry = best_entry.copy()
    final_entry["symbol"] = best_symbol
    final_entry["selected_strategy"] = best_strategy
    final_entry["final_score"] = best_score
    final_entry["entry_version"] = ENTRY_VERSION

    return {
        "has_final_entry": True,
        "symbol": best_symbol,
        "strategy": best_strategy,
        "entry": final_entry,
        "detail": raw_candidates,  # 컷 전 후보 전체는 그대로 유지
        "reason": "OK",
        "selection_reason": selection_reason,  # 🔹 여기에 선택 이유 텍스트 추가
        "version": ENTRY_VERSION,
    }
