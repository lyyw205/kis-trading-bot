# auto_train_worker.py
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
import joblib

from db import BotDatabase

DB_PATH = "trading.db"
MODEL_DIR = "models"
BASE_MODEL_NAME = "seq_model"

SEQ_LEN = 30   # 시퀀스 길이 (trainer / trader와 동일하게 유지)
TP_PCT = 0.03  # +3% 익절
SL_PCT = -0.04 # -4% 손절


# --------------------------------------------------
# 1) 데이터 로딩 유틸
# --------------------------------------------------
def load_ml_seq_samples():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM ml_seq_samples", conn)
    conn.close()
    return df


def load_all_ohlcv():
    """
    ohlcv_data를 region+symbol+interval 별로 DataFrame으로 묶어서 dict로 반환
    key: (region, symbol, interval)
    val: df (index=dt, columns=[open, high, low, close, volume])
    """
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """
        SELECT region, symbol, interval, dt, open, high, low, close, volume
        FROM ohlcv_data
        ORDER BY region, symbol, interval, dt
        """,
        conn,
    )
    conn.close()

    if df.empty:
        return {}

    df["dt"] = pd.to_datetime(df["dt"])

    groups = {}
    for (region, symbol, interval), g in df.groupby(
        ["region", "symbol", "interval"], sort=False
    ):
        g = g.copy()
        g = g.sort_values("dt")
        g.set_index("dt", inplace=True)
        g = g[["open", "high", "low", "close", "volume"]].astype(float)
        groups[(region, symbol, interval)] = g

    return groups


def build_feature_from_seq(df_seq: pd.DataFrame):
    """
    df_seq: index=dt, columns=['open','high','low','close','volume']
    길이 = SEQ_LEN

    피처 구성:
      - close/high/low 를 첫 캔들 종가 대비 수익률로 정규화
      - volume 을 평균 대비 비율로 정규화
      => 총 4 * SEQ_LEN 차원
    """
    if len(df_seq) != SEQ_LEN:
        return None

    close = df_seq["close"].values
    high = df_seq["high"].values
    low = df_seq["low"].values
    vol = df_seq["volume"].values

    base = close[0]
    if base <= 0:
        return None

    close_rel = close / base - 1.0
    high_rel = high / base - 1.0
    low_rel = low / base - 1.0

    vol_mean = np.mean(vol) if np.mean(vol) > 0 else 1.0
    vol_norm = vol / vol_mean

    feat = np.concatenate([close_rel, high_rel, low_rel, vol_norm])
    return feat


# --------------------------------------------------
# 2) 새 모델 학습 + 밸리데이션 세트 준비
# --------------------------------------------------
def train_new_model(db: BotDatabase):
    df_samples = load_ml_seq_samples()
    if df_samples.empty:
        db.log("❌ ml_seq_samples 테이블이 비어 있음 → 학습 불가")
        return None

    # label 0/1만 사용
    df_samples = df_samples[df_samples["label"].isin([0, 1])].copy()
    if df_samples.empty:
        db.log("❌ 라벨 0/1 샘플 없음 → 학습 불가")
        return None

    ohlcv_dict = load_all_ohlcv()
    if not ohlcv_dict:
        db.log("❌ ohlcv_data 비어 있음 → 학습 불가")
        return None

    X_list = []
    y_list = []
    # 백테스트용 정보도 같이 들고가기
    future_max_list = []
    future_min_list = []
    meta_list = []  # (region, symbol, interval, dt_entry)

    skip_count = 0

    for _, row in df_samples.iterrows():
        region = row["region"]
        symbol = row["symbol"]
        interval = row["interval"]
        dt_entry_str = row["dt_entry"]
        label = int(row["label"])

        future_max_ret = float(row["future_max_ret"])
        future_min_ret = float(row["future_min_ret"])

        key = (region, symbol, interval)
        if key not in ohlcv_dict:
            skip_count += 1
            continue

        df_ohlcv = ohlcv_dict[key]

        dt_entry = pd.to_datetime(dt_entry_str)
        if dt_entry not in df_ohlcv.index:
            # 타임존/분단위 오차 등으로 인덱스 불일치 시 스킵
            skip_count += 1
            continue

        pos = df_ohlcv.index.get_loc(dt_entry)
        if isinstance(pos, slice):
            pos = pos.stop - 1

        if pos < SEQ_LEN - 1:
            skip_count += 1
            continue

        df_seq = df_ohlcv.iloc[pos - SEQ_LEN + 1 : pos + 1]
        feat = build_feature_from_seq(df_seq)
        if feat is None:
            skip_count += 1
            continue

        X_list.append(feat)
        y_list.append(label)
        future_max_list.append(future_max_ret)
        future_min_list.append(future_min_ret)
        meta_list.append((region, symbol, interval, dt_entry_str))

    if not X_list:
        db.log("❌ 유효 피처 샘플 없음 → 학습 중단")
        return None

    X = np.array(X_list, dtype=float)
    y = np.array(y_list, dtype=int)
    future_max_arr = np.array(future_max_list, dtype=float)
    future_min_arr = np.array(future_min_list, dtype=float)

    db.log(
        f"✅ 새 모델 학습용 샘플: {len(X)}개 | 스킵: {skip_count}개 | 피처 차원: {X.shape[1]}"
    )

    X_train, X_val, y_train, y_val, fut_max_train, fut_max_val, fut_min_train, fut_min_val, meta_train, meta_val = train_test_split(
        X,
        y,
        future_max_arr,
        future_min_arr,
        meta_list,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # 기본 threshold=0.5 기준 분류 성능
    y_pred = model.predict(X_val)
    f1 = f1_score(y_val, y_pred)
    prec = precision_score(y_val, y_pred, zero_division=0)
    rec = recall_score(y_val, y_pred, zero_division=0)

    db.log("=== [새 모델] Classification (val) ===")
    db.log(f"F1={f1:.4f}, Precision={prec:.4f}, Recall={rec:.4f}")
    db.log("Confusion Matrix:")
    db.log(str(confusion_matrix(y_val, y_pred)))
    db.log("Report:")
    db.log(classification_report(y_val, y_pred, digits=4))

    metrics = {
        "f1": f1,
        "precision": prec,
        "recall": rec,
    }

    # 백테스트용 정보도 함께 반환
    val_info = {
        "X_val": X_val,
        "y_val": y_val,
        "future_max": fut_max_val,
        "future_min": fut_min_val,
        "meta": meta_val,
    }

    return model, metrics, val_info


# --------------------------------------------------
# 3) 간이 백테스트 (샘플 레벨) → trades / win_rate / cum_return 등 계산
# --------------------------------------------------
def evaluate_model_trades(
    model,
    X_val,
    future_max,
    future_min,
    ml_threshold: float,
):
    """
    val 샘플들에 대해:
      - model.predict_proba() >= threshold 인 경우만 "진입"
      - 미래 구간에서 TP/SL 여부를 가지고 per-trade 수익률 계산
    """
    proba = model.predict_proba(X_val)[:, 1]

    # 진입하는 시점만 필터
    enter_mask = proba >= ml_threshold
    if not enter_mask.any():
        return {
            "trades": 0,
            "win_rate": 0.0,
            "avg_profit": 0.0,
            "cum_return": 0.0,
            "max_dd": 0.0,
            "threshold": ml_threshold,
        }

    max_r = future_max[enter_mask]
    min_r = future_min[enter_mask]

    # per-trade 수익률 계산
    trade_returns = []
    for mx, mn in zip(max_r, min_r):
        tp_hit = mx >= TP_PCT
        sl_hit = mn <= SL_PCT

        if tp_hit and not sl_hit:
            trade_returns.append(TP_PCT)   # +3%
        elif sl_hit and not tp_hit:
            trade_returns.append(SL_PCT)   # -4%
        elif tp_hit and sl_hit:
            # 순서 모름 → 보수적으로 손절로 처리
            trade_returns.append(SL_PCT)
        else:
            # 둘 다 안 맞으면 미미한 변동 → 0 수익으로 가정
            trade_returns.append(0.0)

    trade_returns = np.array(trade_returns, dtype=float)
    n_trades = len(trade_returns)
    wins = (trade_returns > 0).sum()
    win_rate = (wins / n_trades) * 100 if n_trades > 0 else 0.0
    avg_profit = trade_returns.mean() * 100 if n_trades > 0 else 0.0

    # 누적 수익률 (단순 compounding)
    equity_curve = (1.0 + trade_returns).cumprod()
    cum_return = (equity_curve[-1] - 1.0) * 100  # %

    # 최대 낙폭(MDD)
    peak = np.maximum.accumulate(equity_curve)
    dd = (equity_curve / peak) - 1.0
    max_dd = dd.min() * 100  # %

    return {
        "trades": int(n_trades),
        "win_rate": float(win_rate),
        "avg_profit": float(avg_profit),
        "cum_return": float(cum_return),
        "max_dd": float(max_dd),
        "threshold": float(ml_threshold),
    }


# --------------------------------------------------
# 4) 기존 활성 모델도 동일 방식으로 평가
# --------------------------------------------------
def eval_existing_model_trades(db: BotDatabase, X_val, future_max, future_min, ml_threshold: float):
    active_path = db.get_setting("active_model_path", "")
    if not active_path or not os.path.exists(active_path):
        db.log("⚠️ 기존 활성 모델 없음 → 비교 스킵")
        return None, None

    try:
        model = joblib.load(active_path)
    except Exception as e:
        db.log(f"⚠️ 기존 모델 로드 실패: {e}")
        return None, None

    metrics_trades = evaluate_model_trades(
        model,
        X_val=X_val,
        future_max=future_max,
        future_min=future_min,
        ml_threshold=ml_threshold,
    )
    return active_path, metrics_trades


# --------------------------------------------------
# 5) 전체 파이프라인: 학습 + 간이백테스트 + 모델비교 + 저장/활성화
# --------------------------------------------------
def auto_train_and_update():
    os.makedirs(MODEL_DIR, exist_ok=True)
    db = BotDatabase(DB_PATH)
    db.log("🧠 [AUTO] 시퀀스 기반 ML 자동 재학습 시작")

    # 새 모델 학습
    result = train_new_model(db)
    if result is None:
        db.log("❌ 새 모델 학습 실패 → 종료")
        return

    new_model, cls_metrics, val_info = result
    X_val = val_info["X_val"]
    future_max = val_info["future_max"]
    future_min = val_info["future_min"]

    # threshold는 settings에서 읽거나 기본값 사용
    ml_thr_str = db.get_setting("ml_threshold", "0.55")
    try:
        ml_thr = float(ml_thr_str)
    except:
        ml_thr = 0.55

    # 새 모델에 대한 간이 백테스트
    new_trade_metrics = evaluate_model_trades(
        new_model,
        X_val=X_val,
        future_max=future_max,
        future_min=future_min,
        ml_threshold=ml_thr,
    )
    db.log("=== [새 모델] Trade-level 백테스트 (val) ===")
    db.log(str(new_trade_metrics))

    # 기존 활성 모델 평가
    old_path, old_trade_metrics = eval_existing_model_trades(
        db,
        X_val=X_val,
        future_max=future_max,
        future_min=future_min,
        ml_threshold=ml_thr,
    )

    # 모델 저장
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_filename = f"{BASE_MODEL_NAME}_{ts}.pkl"
    model_path = os.path.join(MODEL_DIR, model_filename)
    joblib.dump(new_model, model_path)
    db.log(f"💾 새 모델 저장 완료: {model_path}")

    # model_versions 기록
    params = {
        "model_type": "RandomForestClassifier",
        "n_estimators": 300,
        "seq_len": SEQ_LEN,
        "features": "close_rel+high_rel+low_rel+vol_norm",
        "cls_metrics": cls_metrics,
        "trade_metrics": new_trade_metrics,
        "ml_threshold": ml_thr,
    }
    version_str = ts
    note = f"auto-train, F1={cls_metrics['f1']:.4f}, cum_return={new_trade_metrics['cum_return']:.2f}%"

    model_id = db.save_model_version(
        name="seq_model",
        version=version_str,
        params=params,
        note=note,
    )
    if model_id:
        db.log(f"📝 model_versions 등록 완료 (id={model_id})")
    else:
        db.log("⚠️ model_versions 저장 실패")

    # backtests 테이블에 새 모델 결과 저장
    db.save_backtest(
        model_id=model_id if model_id else None,
        start_date="(val_auto)",
        end_date="(val_auto)",
        trades=new_trade_metrics["trades"],
        win_rate=new_trade_metrics["win_rate"],
        avg_profit=new_trade_metrics["avg_profit"],
        cum_return=new_trade_metrics["cum_return"],
        max_dd=new_trade_metrics["max_dd"],
        note="auto-train val 백테스트",
    )

    # 기존 모델과 비교 (단순히 cum_return 기준)
    def metric_desc(m):
        return f"trades={m['trades']}, win={m['win_rate']:.2f}%, cum={m['cum_return']:.2f}%"

    if old_trade_metrics is not None:
        db.log("=== [기존 활성 모델] Trade-level 백테스트 (val) ===")
        db.log(str(old_trade_metrics))

        new_cum = new_trade_metrics["cum_return"]
        old_cum = old_trade_metrics["cum_return"]

        db.log(f"비교 → 새모델: {metric_desc(new_trade_metrics)}")
        db.log(f"비교 → 구모델: {metric_desc(old_trade_metrics)}")

        if new_cum > old_cum:
            # 새 모델로 교체
            db.set_setting("active_model_path", model_path)
            db.log("🎯 새 모델이 더 우수 → active_model_path 교체 완료")
        else:
            db.log("⏸ 기존 모델이 더 좋거나 비슷 → active 유지, 새 모델은 기록만")
    else:
        # 기존 모델이 없으면 무조건 새 모델 활성화
        db.set_setting("active_model_path", model_path)
        db.log("✅ 기존 활성 모델 없음 → 새 모델을 active로 설정")

    db.log("🎉 [AUTO] 자동 재학습 + 간이 백테스트 + 모델 비교 완료")


if __name__ == "__main__":
    auto_train_and_update()
