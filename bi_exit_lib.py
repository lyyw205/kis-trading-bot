# bi_exit_lib.py
# CR 코인 청산용 공통 Core (TP/SL/Timeout + ML 재예측)

from typing import Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd

from bi_entry_lib import run_bi_swing_ml


EXIT_VERSION = "BI_MS_ENTRY_v3_TRANS_2025-12-07"

# ------------------------------------------------------
# 포지션 정보 구조 (편의용 dataclass)
# ------------------------------------------------------
@dataclass
class CrPosition:
    region: str          # "CR"
    symbol: str          # "KRW-BTC" 등
    side: str            # "BUY" / "SELL" / "LONG" / "SHORT"
    qty: float
    entry_price: float
    entry_time: datetime
    # 선택: 엔트리 시점 ML/리스크 정보
    ml_score_entry: Optional[float] = None
    ml_worst_entry: Optional[float] = None
    atr_ratio_entry: Optional[float] = None
    # ✅ Trailing/추가 상태 저장용 (전략에서 자유롭게 사용)
    max_pnl_pct: Optional[float] = None  # Trailing용 최고 수익률 저장


# ------------------------------------------------------
# 기본 파라미터 (공통 Base)
# ------------------------------------------------------
DEFAULT_EXIT_PARAMS: Dict[str, Any] = {
    # 퍼센트 기준 TP/SL
    "tp_rate": 0.03,           # 기존 TP 후보 (+3%)
    "sl_rate": -0.02,          # -2% 손실 시 SL 후보

    # 👉 새로 추가: 소프트 익절 (현재는 MS에서만 활용)
    "soft_tp_rate": 0.020,     # +2% 정도에서 '적당히' 익절 후보
    "soft_tp_min_bars": 4,     # 최소 4봉은 버틴 후에만 소프트 익절

    # 최대 보유 봉 수 (5분봉 기준)
    "max_hold_bars": 10,

    # TIMEOUT이 너무 빨리 나가는 문제 방지용 최소 보유 봉
    "min_hold_bars_for_timeout": 5,

    # TIMEOUT deadband (±수익이 너무 크면 TIMEOUT 안 쓰기 위한 범위)
    "timeout_deadband": 0.010,

    # ML 재예측 관련
    "use_ml_exit": False,
    "ml_recheck_min_bars": 3,
    "ml_recheck_interval": 3,

    # 수익 보호용 ML 조건
    "ml_protect_profit_drop": 0.004,
    "ml_protect_worst_floor": -0.005,

    # 손실 제한용 ML 조건
    "ml_accel_cut_score": 0.0,
    "ml_accel_cut_worst": -0.01,

    # ✅ Trailing 관련 공통 파라미터 추가 (기본은 OFF)
    "use_trailing": False,          # 전략에서 True로 켜면 Trailing 사용
    "tp_start_rate": 0.03,          # 이 수익률 이상부터 Trailing 영역 (기본 3%)
    "trail_gap": 0.015,             # 최고 수익 대비 몇 % 되돌림에서 컷
    "min_bars_for_trailing": 0,     # 최소 몇 봉 이상일 때부터 Trailing 허용

    # ML EXIT 를 어느 PnL 구간에서만 허용할지 (게이트)
    "ml_pnl_min": -0.05,      # 예: -5% 이하에선 ML EXIT 안 쓰자 (나중에 조정)
    "ml_pnl_max": 0.08,       # 예: +8% 이상 수익 구간에선 Trailing/TP만 믿자
}

# ✅ 전략별 모듈에서 공통으로 쓰는 BASE 이름
DEFAULT_EXIT_PARAMS_BASE: Dict[str, Any] = DEFAULT_EXIT_PARAMS.copy()

# -------------------------------------------------------------
# 공통: holding_bars 계산 (시간 기준 → 5m 봉 개수)
# -------------------------------------------------------------
def compute_holding_bars(
    entry_time: datetime,
    now_dt: datetime,
    bar_minutes: int = 5,
) -> int:
    """
    entry_time ~ now_dt 사이 경과 시간을 bar_minutes 단위 봉 수로 변환.
    - 음수 방지
    - entry_time 이상할 때는 now_dt로 대체
    """
    try:
        entry_ts = pd.to_datetime(entry_time)
    except Exception:
        entry_ts = now_dt

    delta_min = max((now_dt - entry_ts).total_seconds() / 60.0, 0.0)

    if bar_minutes <= 0:
        return 0

    # 🔹 floor / round 는 전략적으로 선택 가능.
    #    과도한 앞당김을 막으려면 floor가 조금 더 보수적.
    return int(delta_min // bar_minutes)

# ------------------------------------------------------
# PnL / 경과 봉수 계산 (MS, REV, MOMO 모두 사용 가능)
# ------------------------------------------------------
def calc_pnl_and_bars(
    pos: CrPosition,
    df_5m: pd.DataFrame,
    cur_price: float,
    now_dt: datetime,
):
    """
    - pnl_pct : 포지션 수익률 (side 반영)
    - holding_bars : entry_time ~ now_dt 까지 경과한 5분봉 개수
    """
    # ---------- 1) PnL 계산 ----------
    entry_price = float(pos.entry_price or 0.0)
    cur_price_f = float(cur_price)

    side = str(pos.side).upper()
    direction = 1.0 if side in ("BUY", "LONG") else -1.0

    if entry_price > 0:
        pnl_pct = direction * (cur_price_f - entry_price) / entry_price
    else:
        pnl_pct = 0.0

    # ---------- 2) holding_bars 계산 (now_dt 기준, 공통 헬퍼 사용) ----------
    holding_bars = compute_holding_bars(entry_time=pos.entry_time, now_dt=now_dt, bar_minutes=5)

    return pnl_pct, holding_bars

# -------------------------------------------------------------
# (보조) PnL / 경과 bar 계산 - 단순 버전
#  → REV/MOMO에서 쓰는 기본 TP/SL/Timeout 체크용
# -------------------------------------------------------------
def compute_pnl_pct(entry_price: float, cur_price: float) -> float:
    if entry_price <= 0:
        return 0.0
    return (cur_price - entry_price) / entry_price


def estimate_holding_bars(entry_time: datetime, now_dt: datetime, bar_minutes: int = 5) -> int:
    """
    ✅ 기존 함수명 유지 (REV/MOMO 등에서 사용 중)
    내부 구현은 compute_holding_bars 로 통일.
    """
    return compute_holding_bars(entry_time=entry_time, now_dt=now_dt, bar_minutes=bar_minutes)

# -------------------------------------------------------------
# 공통: Trailing Stop 업데이트 & Exit 판단
# -------------------------------------------------------------

def update_trailing_and_check_exit(
    pos: CrPosition,
    pnl_pct: float,
    holding_bars: int,
    params: Dict[str, Any],
):
    """
    공통 Trailing Stop 로직 + 로그 출력

    - pos.max_pnl_pct   : 지금까지의 최고 수익률
    - pos.trailing_active : Trailing 모드 ON/OFF

    반환:
        (should_exit, reason, note)
    """
    symbol = getattr(pos, "symbol", "?")
    side = getattr(pos, "side", "?")

    tp_start = float(params.get("tp_start_rate", params.get("tp_rate", 0.03)))
    trail_gap = float(params.get("trail_gap", 0.02))
    min_bars_for_trailing = int(params.get("min_bars_for_trailing", 0))

    # 1) 상태 초기화
    if not hasattr(pos, "max_pnl_pct") or pos.max_pnl_pct is None:
        pos.max_pnl_pct = pnl_pct

    if not hasattr(pos, "trailing_active") or pos.trailing_active is None:
        pos.trailing_active = False

    # 2) Trailing 모드 진입 조건 체크
    if (
        (not pos.trailing_active) and
        (holding_bars >= min_bars_for_trailing) and
        (pnl_pct >= tp_start)
    ):
        pos.trailing_active = True
        pos.max_pnl_pct = max(pos.max_pnl_pct, pnl_pct)

        # 🔵 Trailing 시작 로그
        print(
            f"[TRAILING_START] {symbol} {side} "
            f"pnl={pnl_pct:.4f}, bars={holding_bars}, "
            f"tp_start={tp_start:.4f}"
        )

        print(
            f"[TRAILING_MAX_INIT] {symbol} {side} "
            f"max_pnl={pos.max_pnl_pct:.4f}"
        )

    # 3) Trailing 모드인 상태에서만 최고점 추적 + Exit 체크
    if pos.trailing_active:
        # 최고점 갱신 체크
        if pnl_pct > pos.max_pnl_pct + 1e-8:
            pos.max_pnl_pct = pnl_pct
            # 🟢 최고점 갱신 로그
            print(
                f"[TRAILING_MAX_UPDATE] {symbol} {side} "
                f"max_pnl={pos.max_pnl_pct:.4f}, bars={holding_bars}"
            )

        drawdown = pos.max_pnl_pct - pnl_pct

        if drawdown >= trail_gap:
            # 🔴 Trailing Stop 청산 로그
            print(
                f"[TRAILING_EXIT] {symbol} {side} "
                f"pnl={pnl_pct:.4f}, max={pos.max_pnl_pct:.4f}, "
                f"dd={drawdown:.4f}, gap={trail_gap:.4f}"
            )

            note = (
                f"TRAILING_STOP(pnl={pnl_pct:.4f}, "
                f"max={pos.max_pnl_pct:.4f}, dd={drawdown:.4f})"
            )
            return True, "TRAILING_STOP", note

    # 청산 신호 없음
    return False, "", ""

# -------------------------------------------------------------
# 공통: 하드 TP/SL/Timeout 체크
# -------------------------------------------------------------
def check_basic_exit_rules(
    entry_price: float,
    cur_price: float,
    entry_time: datetime,
    now_dt: datetime,
    params: Dict[str, Any],
):
    """
    - TP / SL / Timeout만 보는 기본 룰
    반환:
        (should_exit, reason, note)
    """
    tp_rate = params.get("tp_rate", 0.02)
    sl_rate = params.get("sl_rate", -0.02)
    max_hold_bars = params.get("max_hold_bars", 24)

    pnl_pct = compute_pnl_pct(entry_price, cur_price)
    bars = estimate_holding_bars(entry_time, now_dt, bar_minutes=5)

    if pnl_pct >= tp_rate:
        return True, "TP", f"TP_HIT(pnl={pnl_pct:.4f})"
    if pnl_pct <= sl_rate:
        return True, "SL", f"SL_HIT(pnl={pnl_pct:.4f})"
    if bars >= max_hold_bars:
        return True, "TIMEOUT", f"TIMEOUT({bars} bars)"

    return False, "", ""


# -------------------------------------------------------------
# 공통: ML 재예측 기반 조기 청산
# -------------------------------------------------------------
def check_ml_based_exit(
    df_5m: pd.DataFrame,
    cur_price: float,
    entry_price: float,
    entry_ml_score: Optional[float],
    entry_ml_worst: Optional[float],
    now_dt: datetime,
    params: Dict[str, Any],
):
    """
    run_cr_swing_ml 로 현재 구간을 다시 예측해서
    - 수익 중인데 score/worst가 많이 악화 → 조기 익절
    - 손실 중인데 score/worst가 더 안 좋아짐 → 손절 가속
    """
    if not params.get("use_ml_exit", True):
        return False, "", "", None

    ml = run_bi_swing_ml(df_5m, params)
    if ml is None:
        return False, "", "", None

    cur_score = ml["score"]
    cur_worst = ml["worst"]

    pnl_pct = compute_pnl_pct(entry_price, cur_price)

    pnl_min = params.get("ml_pnl_min", -999)
    pnl_max = params.get("ml_pnl_max", 999)
    
    if not (pnl_min < pnl_pct < pnl_max):
        # 이 구간에선 ML EXIT 사용 안 함
        return False, "", "", None
    
    # 기준값
    protect_drop = params.get("ml_protect_profit_drop", 0.004)
    protect_worst = params.get("ml_protect_worst_floor", -0.005)

    accel_score = params.get("ml_accel_cut_score", 0.0)
    accel_worst = params.get("ml_accel_cut_worst", -0.01)

    # 수익 중 → 수익 보호 로직
    if pnl_pct > 0 and entry_ml_score is not None:
        score_drop = entry_ml_score - cur_score
        if score_drop >= protect_drop or cur_worst <= protect_worst:
            reason = "ML_PROTECT_PROFIT"
            note = f"ML_PROTECT(score_drop={score_drop:.4f}, worst={cur_worst:.4f}, pnl={pnl_pct:.4f})"
            return True, reason, note, ml

    # 손실 중 → 손절 가속 로직
    if pnl_pct < 0:
        if (cur_score <= accel_score) or (cur_worst <= accel_worst):
            reason = "ML_ACCEL_CUT"
            note = f"ML_ACCEL(score={cur_score:.4f}, worst={cur_worst:.4f}, pnl={pnl_pct:.4f})"
            return True, reason, note, ml

    return False, "", "", ml
