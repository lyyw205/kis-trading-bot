# bi_train_models.py
# 멀티스케일 TCN 모델 학습(train) 스크립트
import os
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from datetime import datetime

import numpy as np             # ✅ 이미 main 안에서 쓰고 있었으면 위로 올려도 됨
import pandas as pd            # ✅ 추가

from c_config import BI_UNIVERSE_STOCKS
from bi_multiscale_loader import load_ohlcv_multiscale_for_symbol
from bi_create_dataset import MultiScaleOhlcvDatasetCR  # ✅ Dataset만
from bi_define_models import MultiScaleTCNTransformer
from bi_features import FEATURE_COLS, SEQ_LENS, HORIZONS, build_multiscale_samples_cr


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 🔧 분류 손실 가중치 λ (loss = loss_reg + λ * loss_cls)
CLS_LOSS_WEIGHT = 1.0

def make_cls_labels(y: torch.Tensor) -> torch.Tensor:
    """
    회귀 타겟 y (수익률, shape: (B, H)) 로부터
    분류용 라벨 생성: y > 0 이면 1, 아니면 0
    """
    return (y > 0).float()

def load_positions_all() -> pd.DataFrame:
    """
    positions 전체를 DataFrame으로 불러오는 헬퍼.
    - region='BI' 만 사용하는 게 자연스러움.
    - 실제 구현은 네가 쓰는 DB/클라이언트에 맞춰 채우면 됨.
    """
    # 예시) 만약 Supabase/Postgres를 직접 연결한다면 여기서 읽기
    # 아래는 '직접 구현 필요'인 자리 표시자
    # return pd.read_sql("SELECT * FROM public.positions WHERE region = 'BI';", conn)

    raise NotImplementedError("load_positions_all() 안을 실제 DB 코드로 구현하세요.")

def build_trade_labels_for_symbol(base_dt_array, positions_sym_df: pd.DataFrame):
    """
    base_dt_array: build_multiscale_samples_cr(..., return_index=True) 에서 받은 (N,) dt 배열
    positions_sym_df: 해당 symbol 포지션들 (region/symbol 필터 후)

    반환:
      y_trade   : (N,) float32,  0.0/1.0
      trade_mask: (N,) float32,  0.0/1.0   (포지션 라벨이 있는 시점만 1)
    """
    # 1) 닫힌 포지션 + pnl_pct/entry_time 있는 것만 사용
    pos = positions_sym_df.copy()
    pos = pos[
        (pos["status"] == "CLOSED")
        & (~pos["pnl_pct"].isna())
        & (~pos["entry_time"].isna())
    ]

    N = len(base_dt_array)
    if pos.empty:
        # 이 심볼에 쓸 포지션 라벨 없음
        return np.zeros(N, dtype=np.float32), np.zeros(N, dtype=np.float32)

    # 2) entry_time → KST 기준 5분봉 타임스텝으로 정렬
    pos["entry_time"] = pd.to_datetime(pos["entry_time"], utc=True)
    pos["entry_kst"] = pos["entry_time"].dt.tz_convert("Asia/Seoul")
    pos["entry_5m"] = pos["entry_kst"].dt.floor("5min")

    # 3) entry_5m 시점별로 라벨 매핑
    #    pnl_pct > 0 → 1.0, else 0.0
    label_map = {}
    for _, row in pos.iterrows():
        dt = row["entry_5m"]
        pnl_pct = float(row["pnl_pct"])
        label = 1.0 if pnl_pct > 0 else 0.0
        # 한 시점에 포지션 여러 개면 마지막 것 기준(필요하면 평균 등으로 바꿔도 됨)
        label_map[dt] = label

    # 4) base_dt_array 순서대로 y_trade / mask 채우기
    base_dt_series = pd.to_datetime(base_dt_array)

    y_trade = np.zeros(N, dtype=np.float32)
    trade_mask = np.zeros(N, dtype=np.float32)

    for i, dt in enumerate(base_dt_series):
        # df_5m.index 는 보통 naive KST datetime → tz_localize 필요
        if dt.tzinfo is None:
            dt_kst = dt.tz_localize("Asia/Seoul")
        else:
            dt_kst = dt.tz_convert("Asia/Seoul")

        if dt_kst in label_map:
            y_trade[i] = label_map[dt_kst]
            trade_mask[i] = 1.0

    return y_trade, trade_mask

def main():
    # =====================
    # 1) 데이터 모으기
    # =====================
    feature_cols = FEATURE_COLS 
    seq_lens = SEQ_LENS           
    horizons = HORIZONS          

    try:
        positions_all = load_positions_all()
    except NotImplementedError:
        # 아직 구현 안 했으면 positions 없이 학습 (기존 방식)
        positions_all = None
        print("[WARN] load_positions_all()가 구현되지 않아 positions 기반 라벨 없이 학습합니다.")

    X5_list_all = []
    X15_list_all = []
    X30_list_all = []
    X1h_list_all = []
    Y_list_all = []
    y_trade_list_all = []
    trade_mask_list_all = []
    
    total_count = len(BI_UNIVERSE_STOCKS)
    print(f"🚀 [Start] 총 {total_count}개 코인 데이터 로딩 시작...")

    for i, t in enumerate(BI_UNIVERSE_STOCKS):
        region = t["region"]
        symbol = t["symbol"]

        print(f"  -> [{i+1}/{total_count}] {symbol} 데이터 처리 중...", end="\r")

        try:
            df_5m, df_15m, df_30m, df_1h = load_ohlcv_multiscale_for_symbol(
                region=region,
                symbol=symbol,
                base_interval="5m",
            )
        except ValueError as e:
            print(f"[WARN] {region} {symbol} OHLCV 로딩 실패: {e}")
            continue

        try:
            X_5m, X_15m, X_30m, X_1h, Y, base_dt = build_multiscale_samples_cr(
                df_5m=df_5m,
                df_15m=df_15m,
                df_30m=df_30m,
                df_1h=df_1h,
                feature_cols=feature_cols,
                seq_lens=seq_lens,
                horizons=horizons,
                return_index=True,
            )
        except ValueError as e:
            print(f"[WARN] {region} {symbol} 샘플 생성 실패: {e}")
            continue

        X5_list_all.append(X_5m)
        X15_list_all.append(X_15m)
        X30_list_all.append(X_30m)
        X1h_list_all.append(X_1h)
        Y_list_all.append(Y)

        # ✅ 이 심볼에 해당하는 positions 추출 & 라벨 생성
        if positions_all is not None:
            pos_sym = positions_all[
                (positions_all["region"] == region)
                & (positions_all["symbol"] == symbol)
            ]
            y_trade_sym, trade_mask_sym = build_trade_labels_for_symbol(
                base_dt, pos_sym
            )
        else:
            # positions를 아직 안 쓰는 경우: 0으로 채우기 (사실상 trade_loss=0이 됨)
            y_trade_sym = np.zeros(len(Y), dtype=np.float32)
            trade_mask_sym = np.zeros(len(Y), dtype=np.float32)
            
        # ✅ 라벨도 함께 모으기
        y_trade_list_all.append(y_trade_sym)
        trade_mask_list_all.append(trade_mask_sym)

    if not X5_list_all:
        raise RuntimeError("BI_UNIVERSE_STOCKS 전체에서 유효한 샘플이 하나도 없습니다.")

    X_5m = np.concatenate(X5_list_all, axis=0)
    X_15m = np.concatenate(X15_list_all, axis=0)
    X_30m = np.concatenate(X30_list_all, axis=0)
    X_1h = np.concatenate(X1h_list_all, axis=0)
    Y = np.concatenate(Y_list_all, axis=0)

    # ✅ 포지션 라벨도 concat
    y_trade_all = np.concatenate(y_trade_list_all, axis=0)
    trade_mask_all = np.concatenate(trade_mask_list_all, axis=0)

    dataset = MultiScaleOhlcvDatasetCR(
        X_5m, X_15m, X_30m, X_1h, Y,
        y_trade=y_trade_all,
        trade_mask=trade_mask_all,
    )

    # 🔎 데이터 요약 로그
    print("")
    print("✅ [DATA SUMMARY]")
    print(f"  - 총 심볼 수: {len(X5_list_all)}")
    print(f"  - 총 샘플 수: {len(dataset)}")
    print(f"  - 5m 시퀀스 shape: {X_5m.shape}")
    print(f"  - Y shape: {Y.shape}")
    print(f"  - Y 통계: mean={Y.mean():.6f}, std={Y.std():.6f}", flush=True)

    # =====================
    # 2) train / val split
    # =====================
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train 

    indices = torch.arange(n_total)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_set = torch.utils.data.Subset(dataset, train_indices)
    val_set = torch.utils.data.Subset(dataset, val_indices)

    print("✅ [SPLIT]")
    print(f"  - train 샘플: {n_train}")
    print(f"  - val 샘플:   {n_val}", flush=True)

    # [수정] batch_size를 64 -> 512 또는 1024로 늘리세요. (학습 속도 대폭 향상)
    # [수정] num_workers를 0 -> 4 정도로 설정하세요. (CPU 코어 활용)
    # 단, Windows에서는 num_workers > 0 일 때 에러가 날 수도 있습니다. 
    # 에러 나면 다시 0으로, 안 나면 4가 훨씬 빠릅니다.
    
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=512, shuffle=False, num_workers=0)

    # =====================
    # 3) 모델 준비
    # =====================
    in_features = len(feature_cols)

    model = MultiScaleTCNTransformer(
        in_features=in_features,
        horizons=horizons,
        hidden_channels=64,
        tcn_layers_per_scale=4,
        transformer_layers=2,
        nhead=4,
        dropout=0.1,
        use_classification=True,  # 회귀 + 분류 멀티태스크
        use_trade_head=True, 
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_reg_fn = nn.SmoothL1Loss()
    loss_cls_fn = nn.BCEWithLogitsLoss()

    # ✅ 포지션 기반 이진 분류용 loss (마스크를 씌워야 해서 reduction='none')
    loss_trade_fn = nn.BCEWithLogitsLoss(reduction="none")
    TRADE_LOSS_WEIGHT = 0.5  # 처음엔 0.2~0.5 정도로 시작 추천

    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"multiscale_BI_model_{timestamp}.pth")

    # =====================
    # 4) 학습 루프
    # =====================
    num_epochs = 30
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        model.train()
        total_loss = 0.0
        total_reg_loss = 0.0
        total_trade_loss = 0.0
        total_cls_loss = 0.0
        total_cls_correct = 0.0
        total_cls_elems = 0

        for batch in train_loader:
            x_5m = batch["x_5m"].to(DEVICE)
            x_15m = batch["x_15m"].to(DEVICE)
            x_30m = batch["x_30m"].to(DEVICE)
            x_1h = batch["x_1h"].to(DEVICE)
            y = batch["y"].to(DEVICE)  # (B, H)

            y_cls = make_cls_labels(y)  # (B, H)

            # ✅ 포지션 라벨/마스크 (없을 수도 있어서 get 사용)
            y_trade = batch.get("y_trade")
            trade_mask = batch.get("trade_mask")
            if y_trade is not None:
                y_trade = y_trade.to(DEVICE)         # (B,)
                trade_mask = trade_mask.to(DEVICE)   # (B,)

            optimizer.zero_grad()
            out = model(x_5m, x_15m, x_30m, x_1h)

            pred_reg = out["reg"]       # (B, H)
            logits = out["logits"]      # (B, H)

            loss_reg = loss_reg_fn(pred_reg, y)
            loss_cls = loss_cls_fn(logits, y_cls)

            # ✅ 기본 loss
            loss = loss_reg + CLS_LOSS_WEIGHT * loss_cls

            # ✅ trade head가 있고, y_trade가 있을 때만 trade loss 추가
            trade_loss = torch.tensor(0.0, device=DEVICE)
            if y_trade is not None and "trade_logits" in out:
                trade_logits = out["trade_logits"]      # (B,)
                raw_trade_loss = loss_trade_fn(trade_logits, y_trade)  # (B,)
                # 마스크 적용해서 라벨 있는 시점만 평균
                trade_loss = (raw_trade_loss * trade_mask).sum() / (trade_mask.sum() + 1e-6)
                loss = loss + TRADE_LOSS_WEIGHT * trade_loss

            loss.backward()
            optimizer.step()

            B = y.size(0)

            total_loss += loss.item() * B
            total_reg_loss += loss_reg.item() * B
            total_cls_loss += loss_cls.item() * B
            total_trade_loss += trade_loss.item() * B 

            # 분류 정확도 계산
            with torch.no_grad():
                prob = torch.sigmoid(logits)              # (B, H)
                pred_bin = (prob >= 0.5).float()          # (B, H)
                correct = (pred_bin == y_cls).float().sum().item()
                total_cls_correct += correct
                total_cls_elems += y_cls.numel()

        avg_train_loss = total_loss / n_train
        avg_train_reg_loss = total_reg_loss / n_train
        avg_train_cls_loss = total_cls_loss / n_train
        train_cls_acc = total_cls_correct / total_cls_elems if total_cls_elems > 0 else 0.0
        avg_train_trade_loss = total_trade_loss / n_train
        # ----- validation -----
        model.eval()
        val_loss = 0.0
        val_reg_loss = 0.0
        val_cls_loss = 0.0
        val_cls_correct = 0.0
        val_cls_elems = 0

        with torch.no_grad():
            for batch in val_loader:
                x_5m = batch["x_5m"].to(DEVICE)
                x_15m = batch["x_15m"].to(DEVICE)
                x_30m = batch["x_30m"].to(DEVICE)
                x_1h = batch["x_1h"].to(DEVICE)
                y = batch["y"].to(DEVICE)

                y_cls = make_cls_labels(y)

                out = model(x_5m, x_15m, x_30m, x_1h)
                pred_reg = out["reg"]
                logits = out["logits"]

                loss_reg = loss_reg_fn(pred_reg, y)
                loss_cls = loss_cls_fn(logits, y_cls)
                loss = loss_reg + CLS_LOSS_WEIGHT * loss_cls

                B = y.size(0)
                val_loss += loss.item() * B
                val_reg_loss += loss_reg.item() * B
                val_cls_loss += loss_cls.item() * B

                prob = torch.sigmoid(logits)
                pred_bin = (prob >= 0.5).float()
                correct = (pred_bin == y_cls).float().sum().item()
                val_cls_correct += correct
                val_cls_elems += y_cls.numel()

        avg_val_loss = val_loss / n_val
        avg_val_reg_loss = val_reg_loss / n_val
        avg_val_cls_loss = val_cls_loss / n_val
        val_cls_acc = val_cls_correct / val_cls_elems if val_cls_elems > 0 else 0.0

        epoch_sec = time.time() - epoch_start

        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"train_loss={avg_train_loss:.6f} "
            f"(reg={avg_train_reg_loss:.6f}, cls={avg_train_cls_loss:.6f}, trade={avg_train_trade_loss:.6f}, acc={train_cls_acc*100:.2f}%) | "
            f"val_loss={avg_val_loss:.6f} "
            f"(reg={avg_val_reg_loss:.6f}, cls={avg_val_cls_loss:.6f}, acc={val_cls_acc*100:.2f}%) | "
            f"time={epoch_sec:.1f}s",
            flush=True,
        )

        # 모델 저장 기준은 전체 loss 기준 (필요하면 reg_loss 기준으로 바꿔도 됨)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Best model updated: {save_path} (val_loss={avg_val_loss:.6f})")


if __name__ == "__main__":
    main()
