# train_multiscale_cr.py
# 멀티스케일 TCN 모델 학습(train) 스크립트
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import CR_UNIVERSE_STOCKS
from tcn_multiscale_loader import load_ohlcv_multiscale_for_symbol
from tcn_create_dataset import (
    build_multiscale_samples_cr,
    MultiScaleOhlcvDatasetCR,
)
from tcn_define_models import MultiScaleTCNTransformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    # =====================
    # 1) 데이터 모으기
    # =====================
    feature_cols = ["open", "high", "low", "close", "volume"]

    seq_lens = {
        "5m": 48,   # 240 * 5m = 1200분 (~20시간)
        "15m": 24,
        "30m": 16,
        "1h": 10,
    }

    # 5m 기준 N개 뒤 수익률 (예: 3=15m, 24=2h)
    horizons = [3, 6, 12, 24]

    X5_list_all = []
    X15_list_all = []
    X30_list_all = []
    X1h_list_all = []
    Y_list_all = []

    for t in CR_UNIVERSE_STOCKS:
        region = t["region"]
        symbol = t["symbol"]

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
            X_5m, X_15m, X_30m, X_1h, Y = build_multiscale_samples_cr(
                df_5m=df_5m,
                df_15m=df_15m,
                df_30m=df_30m,
                df_1h=df_1h,
                feature_cols=feature_cols,
                seq_lens=seq_lens,
                horizons=horizons,
            )
        except ValueError as e:
            print(f"[WARN] {region} {symbol} 샘플 생성 실패: {e}")
            continue

        X5_list_all.append(X_5m)
        X15_list_all.append(X_15m)
        X30_list_all.append(X_30m)
        X1h_list_all.append(X_1h)
        Y_list_all.append(Y)

    if not X5_list_all:
        raise RuntimeError("CR_UNIVERSE_STOCKS 전체에서 유효한 샘플이 하나도 없습니다.")

    import numpy as np

    X_5m = np.concatenate(X5_list_all, axis=0)
    X_15m = np.concatenate(X15_list_all, axis=0)
    X_30m = np.concatenate(X30_list_all, axis=0)
    X_1h = np.concatenate(X1h_list_all, axis=0)
    Y = np.concatenate(Y_list_all, axis=0)

    dataset = MultiScaleOhlcvDatasetCR(X_5m, X_15m, X_30m, X_1h, Y)

    # =====================
    # 2) train / val split
    # =====================
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train

    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)

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
        use_classification=False,  # 우선 회귀만
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.SmoothL1Loss()

    save_dir = "models"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "multiscale_cr_model.pth")

    # =====================
    # 4) 학습 루프
    # =====================
    num_epochs = 10
    best_val_loss = float("inf")

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x_5m = batch["x_5m"].to(DEVICE)
            x_15m = batch["x_15m"].to(DEVICE)
            x_30m = batch["x_30m"].to(DEVICE)
            x_1h = batch["x_1h"].to(DEVICE)
            y = batch["y"].to(DEVICE)  # (B, H)

            optimizer.zero_grad()
            out = model(x_5m, x_15m, x_30m, x_1h)
            pred = out["reg"]

            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)

        avg_train_loss = total_loss / n_train

        # ----- validation -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_5m = batch["x_5m"].to(DEVICE)
                x_15m = batch["x_15m"].to(DEVICE)
                x_30m = batch["x_30m"].to(DEVICE)
                x_1h = batch["x_1h"].to(DEVICE)
                y = batch["y"].to(DEVICE)

                out = model(x_5m, x_15m, x_30m, x_1h)
                pred = out["reg"]
                loss = loss_fn(pred, y)
                val_loss += loss.item() * y.size(0)

        avg_val_loss = val_loss / n_val

        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f}"
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  Best model updated: {save_path} (val_loss={avg_val_loss:.6f})")


if __name__ == "__main__":
    main()
