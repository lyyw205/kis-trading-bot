# bi_define_models.py
# 멀티스케일 TCN + Transformer 모델 구조 정의
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# TCN Block
# -----------------------------
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=padding,
        )

        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)

        self.res_conv = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else None
        )

    def forward(self, x):
        """
        x: (B, C_in, L)
        """
        residual = x

        out = self.conv1(x)[:, :, :-self.conv1.padding[0]]  # causal trim
        out = out.transpose(1, 2)  # (B, L, C)
        out = self.norm1(out)
        out = F.gelu(out)
        out = out.transpose(1, 2)

        out = self.conv2(out)[:, :, :-self.conv2.padding[0]]
        out = out.transpose(1, 2)
        out = self.norm2(out)
        out = F.gelu(out)
        out = out.transpose(1, 2)

        if self.res_conv is not None:
            residual = self.res_conv(residual)

        return out + residual


# -----------------------------
# Scale Encoder (한 타임프레임용 TCN)
# -----------------------------
class ScaleEncoder(nn.Module):
    def __init__(
        self,
        in_features,      # 입력 feature 수 (예: 16)
        hidden_channels,  # 내부 채널 수 (예: 64)
        num_layers=4,
        kernel_size=3,
    ):
        super().__init__()

        layers = []
        in_ch = in_features
        for i in range(num_layers):
            dilation = 2 ** i
            layers.append(
                TCNBlock(
                    in_channels=in_ch,
                    out_channels=hidden_channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
            )
            in_ch = hidden_channels

        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        """
        x: (B, L, F)
        return: (B, L, C)
        """
        x = x.transpose(1, 2)  # (B, F, L)
        out = self.tcn(x)      # (B, C, L)
        out = out.transpose(1, 2)  # (B, L, C)
        return out


# -----------------------------
# Scale Pooling (시퀀스 → 토큰)
# -----------------------------
class ScalePooling(nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, h_seq):
        """
        h_seq: (B, L, C)
        return: (B, C)
        """
        last = h_seq[:, -1, :]  # 마지막 타임스텝
        return self.norm(last)


# -----------------------------
# Cross-Scale Transformer Fusion
# -----------------------------
class ScaleTransformerFusion(nn.Module):
    def __init__(self, hidden_channels, num_scales, num_layers=2, nhead=4, dropout=0.1):
        super().__init__()
        self.scale_embed = nn.Embedding(num_scales, hidden_channels)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_channels,
            nhead=nhead,
            dim_feedforward=hidden_channels * 4,
            dropout=dropout,
            batch_first=True,  # (B, S, C)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_channels)

    def forward(self, scale_tokens):
        """
        scale_tokens: (B, S, C)
        """
        B, S, C = scale_tokens.shape
        scale_ids = torch.arange(S, device=scale_tokens.device).unsqueeze(0).expand(B, S)
        scale_emb = self.scale_embed(scale_ids)  # (B, S, C)

        x = scale_tokens + scale_emb
        x = self.encoder(x)  # (B, S, C)
        x = self.norm(x)

        global_token = x.mean(dim=1)  # (B, C)
        return global_token, x        # (global, per_scale)


# -----------------------------
# Multi-Horizon Head
# -----------------------------
class MultiHorizonHead(nn.Module):
    def __init__(
        self,
        hidden_channels,
        horizons,
        use_classification=True,
        use_trade_head=False,      # ✅ 추가
    ):
        super().__init__()
        self.horizons = horizons
        self.use_classification = use_classification
        self.use_trade_head = use_trade_head

        self.reg_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, len(horizons)),
        )

        if use_classification:
            self.cls_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, len(horizons)),
            )
        else:
            self.cls_head = None

        # ✅ 포지션 기반 이진 분류 헤드 (h → 1 logit)
        if use_trade_head:
            self.trade_head = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.GELU(),
                nn.Linear(hidden_channels, 1),
            )
        else:
            self.trade_head = None

    def forward(self, h):
        """
        h: (B, C)
        return: dict
          - reg: (B, H)
          - prob, logits (선택)
          - trade_logits, trade_prob (선택)
        """
        r = self.reg_head(h)  # (B, H)
        out = {"reg": r}

        if self.cls_head is not None:
            logits = self.cls_head(h)           # (B, H)
            prob = torch.sigmoid(logits)
            out["logits"] = logits
            out["prob"] = prob

        # ✅ 포지션 기반 이진 분류 출력
        if self.trade_head is not None:
            trade_logits = self.trade_head(h).squeeze(-1)  # (B,)
            trade_prob = torch.sigmoid(trade_logits)
            out["trade_logits"] = trade_logits
            out["trade_prob"] = trade_prob

        return out


# -----------------------------
# 최종 모델: MultiScaleTCNTransformer
# -----------------------------
class MultiScaleTCNTransformer(nn.Module):
    def __init__(
        self,
        in_features,
        horizons=(3, 6, 12, 24),
        hidden_channels=64,
        tcn_layers_per_scale=4,
        transformer_layers=2,
        nhead=4,
        dropout=0.1,
        use_classification=True,
        use_trade_head=False,
    ):
        super().__init__()

        # 사용할 스케일 정의 (필요하면 수정 가능)
        self.scales = ["5m", "15m", "30m", "1h"]
        num_scales = len(self.scales)

        # 1) 스케일별 TCN 인코더 + 풀링
        self.scale_encoders = nn.ModuleDict()
        self.scale_pools = nn.ModuleDict()
        for s in self.scales:
            self.scale_encoders[s] = ScaleEncoder(
                in_features=in_features,
                hidden_channels=hidden_channels,
                num_layers=tcn_layers_per_scale,
            )
            self.scale_pools[s] = ScalePooling(hidden_channels)

        # 2) 스케일 간 Transformer Fusion
        self.fusion = ScaleTransformerFusion(
            hidden_channels=hidden_channels,
            num_scales=num_scales,
            num_layers=transformer_layers,
            nhead=nhead,
            dropout=dropout,
        )

        # 3) Multi-horizon 헤드
        self.head = MultiHorizonHead(
            hidden_channels=hidden_channels,
            horizons=horizons,
            use_classification=use_classification,
            use_trade_head=use_trade_head,
        )

    def forward(self, x_5m, x_15m, x_30m, x_1h):
        """
        x_*: (B, L_scale, F)
        return: dict
          - reg: (B, H)
          - prob, logits (선택)
          - trade_logits, trade_prob (선택)
          - per_scale: (B, S, C)
        """
        scale_inputs = {
            "5m": x_5m,
            "15m": x_15m,
            "30m": x_30m,
            "1h": x_1h,
        }

        scale_tokens = []
        for s in self.scales:
            h_seq = self.scale_encoders[s](scale_inputs[s])  # (B, L, C)
            token = self.scale_pools[s](h_seq)               # (B, C)
            scale_tokens.append(token)

        scale_tokens = torch.stack(scale_tokens, dim=1)  # (B, S, C)

        global_token, per_scale = self.fusion(scale_tokens)  # (B, C), (B, S, C)

        out = self.head(global_token)
        out["per_scale"] = per_scale

        return out
