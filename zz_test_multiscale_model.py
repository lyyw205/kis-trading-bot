# test_multiscale_model.py
import torch
from swing_models_multiscale import MultiScaleTCNTransformer

if __name__ == "__main__":
    B = 4      # batch size
    F = 16     # feature 수 (나중에 실제 feature 개수에 맞춰 변경)
    L_1m = 240
    L_5m = 120
    L_15m = 60
    L_1h = 48

    model = MultiScaleTCNTransformer(
        in_features=F,
        horizons=(3, 6, 12, 24),
        hidden_channels=64,
    )

    x_1m = torch.randn(B, L_1m, F)
    x_5m = torch.randn(B, L_5m, F)
    x_15m = torch.randn(B, L_15m, F)
    x_1h = torch.randn(B, L_1h, F)

    out = model(x_1m, x_5m, x_15m, x_1h)

    print("reg shape :", out["reg"].shape)     # (B, 4)
    if "prob" in out:
        print("prob shape:", out["prob"].shape)
    print("per_scale :", out["per_scale"].shape)  # (B, 4, hidden)
