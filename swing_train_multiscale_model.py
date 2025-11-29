# train_multiscale_model.py
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import pandas as pd

from models_multiscale import MultiScaleTCNTransformer
from dataset_multiscale import build_multiscale_samples, MultiScaleOhlcvDataset


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_ohlcv_multiscale_for_symbol(symbol: str):
    """
    TODO: 여기서 실제로 df_1m, df_5m, df_15m, df_1h 를 만들어야 함.
    - 예: trading.db에서 읽거나
    - CSV에서 읽거나
    - 네가 이미 쓰던 OHLCV 로더 사용

    아래는 예시형태 (임시 코드):

        df_1m = pd.read_csv(f"data/{symbol}_1m.csv", parse_dates=["datetime"], index_col="datetime")
        df_5m = pd.read_csv(f"data/{symbol}_5m.csv", parse_dates=["datetime"], index_col="datetime")
        ...

    실제 환경에 맞게 구현해야 함.
    """
    raise NotImplementedError("load_ohlcv_multiscale_for_symbol 구현 필요")


def main():
    symbol = "BTC_KRW"  # 예시

    # 1) 데이터 로드
    # df_1m, df_5m, df_15m, df_1h = load_ohlcv_multiscale_for_symbol(symbol)

    # 일단 예시: 여기서부터는 df_*가 준비되어 있다고 가정
    # df_*에는 최소한 ["open","high","low","close","volume"]가 있어야 함.
    # feature_cols는 나중에 파생피처를 추가할 수 있음.
    feature_cols = ["open", "high", "low", "close", "volume"]

    # 예시 seq 길이(원하면 조정 가능)
    seq_lens = {
        "1m": 240,
        "5m": 120,
        "15m": 60,
        "1h": 48,
    }

    horizons = [3, 6, 12, 24]  # 1분봉 기준 n개 뒤 수익률

    # 2) 멀티스케일 샘플 생성
    # X_1m, X_5m, X_15m, X_1h, Y = build_multiscale_samples(
    #     df_1m=df_1m,
    #     df_5m=df_5m,
    #     df_15m=df_15m,
    #     df_1h=df_1h,
    #     feature_cols=feature_cols,
    #     seq_lens=seq_lens,
    #     horizons=horizons,
    # )

    # ===========================
    #  ⚠️ 주의: 위 부분은 데이터 준비 후 활성화
    #  지금은 구조만 보여주기 위해 NotImplementedError로 막아둘게.
    # ===========================
    raise NotImplementedError("먼저 df_* 로딩 + build_multiscale_samples부터 구현해야 합니다.")

    dataset = MultiScaleOhlcvDataset(X_1m, X_5m, X_15m, X_1h, Y)

    # train/val split 간단하게 (후에 더 정교하게 나눌 수 있음)
    n_total = len(dataset)
    n_train = int(n_total * 0.8)
    n_val = n_total - n_train

    train_set, val_set = torch.utils.data.random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=0)

    # 3) 모델 준비
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

    # 4) 학습 루프
    num_epochs = 10
    best_val_loss = float("inf")
    save_path = "models/multiscale_cr_model.pth"
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            x_1m = batch["x_1m"].to(DEVICE)
            x_5m = batch["x_5m"].to(DEVICE)
            x_15m = batch["x_15m"].to(DEVICE)
            x_1h = batch["x_1h"].to(DEVICE)
            y = batch["y"].to(DEVICE)  # (B, H)

            optimizer.zero_grad()
            out = model(x_1m, x_5m, x_15m, x_1h)
            pred = out["reg"]  # (B, H)

            loss = loss_fn(pred, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * y.size(0)

        avg_train_loss = total_loss / n_train

        # ---- validation ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_1m = batch["x_1m"].to(DEVICE)
                x_5m = batch["x_5m"].to(DEVICE)
                x_15m = batch["x_15m"].to(DEVICE)
                x_1h = batch["x_1h"].to(DEVICE)
                y = batch["y"].to(DEVICE)

                out = model(x_1m, x_5m, x_15m, x_1h)
                pred = out["reg"]
                loss = loss_fn(pred, y)
                val_loss += loss.item() * y.size(0)

        avg_val_loss = val_loss / n_val

        print(
            f"[Epoch {epoch}/{num_epochs}] "
            f"train_loss={avg_train_loss:.6f} | val_loss={avg_val_loss:.6f}"
        )

        # best model 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  🔥 Best model updated: {save_path} (val_loss={avg_val_loss:.6f})")


if __name__ == "__main__":
    main()
