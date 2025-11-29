# swing_model_cr.py
"""
CR 고수익 단타용 멀티호라이즌 예측 모델
- 입력: (batch, seq_len=60, feat_dim=11)
- 출력: (batch, 3)  # [r_3, r_6, r_12]
"""

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 설정용 dataclass
# -----------------------------
@dataclass
class CrSwingConfig:
    seq_len: int = 60
    feat_dim: int = 11      # 지금 데이터셋 기준
    d_model: int = 64       # Conv/Transformer 채널 수
    tcn_branches: int = 3   # multi-scale 브랜치 개수
    tcn_kernel_size: int = 3
    tcn_dilations = (1, 2, 4)  # dilation 값
    num_transformer_layers: int = 1
    num_heads: int = 4
    dim_feedforward: int = 128
    dropout: float = 0.1
    output_dim: int = 3     # r_3, r_6, r_12


# -----------------------------
# Positional Encoding (Transformer용)
# -----------------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # (max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) *
            (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


# -----------------------------
# Multi-scale TCN Block
# -----------------------------
class MultiScaleTCN(nn.Module):
    """
    여러 dilation Conv1d 브랜치를 병렬로 두고 concat → projection
    입력/출력: (batch, seq_len, d_model)
    """
    def __init__(self, d_model: int, kernel_size: int = 3, dilations=(1, 2, 4), dropout: float = 0.1):
        super().__init__()
        self.branches = nn.ModuleList()
        padding_map = []

        for d in dilations:
            conv = nn.Conv1d(
                in_channels=d_model,
                out_channels=d_model,
                kernel_size=kernel_size,
                padding=(kernel_size - 1) * d,
                dilation=d,
            )
            self.branches.append(conv)
            padding_map.append(d)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model * len(dilations), d_model)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        # Conv1d는 (batch, channels, seq_len) 형식
        x_in = x.transpose(1, 2)  # (B, C, L)
        branch_outputs = []

        for conv in self.branches:
            y = conv(x_in)             # (B, C, L')
            # padding으로 길이가 seq_len보다 길어질 수 있으니 앞부분 잘라서 맞춰줌
            if y.size(-1) > x_in.size(-1):
                y = y[..., -x_in.size(-1):]
            branch_outputs.append(y)

        # (B, C * num_branches, L)
        h = torch.cat(branch_outputs, dim=1)
        # 다시 (B, L, C * num_branches)
        h = h.transpose(1, 2)

        # projection + residual
        h = self.out_proj(h)
        h = self.dropout(self.activation(h))
        out = self.norm(x + h)  # residual + layernorm

        return out


# -----------------------------
# 메인 모델
# -----------------------------
class CrSwingModel(nn.Module):
    def __init__(self, config: Optional[CrSwingConfig] = None):
        super().__init__()
        self.config = config or CrSwingConfig()

        cfg = self.config
        d_model = cfg.d_model

        # 1) feature projection
        self.input_proj = nn.Linear(cfg.feat_dim, d_model)

        # 2) multi-scale TCN
        self.tcn = MultiScaleTCN(
            d_model=d_model,
            kernel_size=cfg.tcn_kernel_size,
            dilations=cfg.tcn_dilations,
            dropout=cfg.dropout,
        )

        # 3) Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,  # (batch, seq_len, d_model)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.num_transformer_layers,
        )

        self.pos_encoder = PositionalEncoding(d_model=d_model, dropout=cfg.dropout, max_len=cfg.seq_len + 10)

        # 4) 헤드: 마지막 토큰(or 평균) → MLP → 3개 출력
        self.pool = "last"  # "last" 또는 "mean" 으로 바꿔도 됨

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d_model, cfg.output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, feat_dim)
        return: (batch, 3)
        """
        # 1) feature projection
        h = self.input_proj(x)  # (B, L, d_model)

        # 2) TCN
        h = self.tcn(h)         # (B, L, d_model)

        # 3) Positional Encoding + Transformer
        h = self.pos_encoder(h)
        h = self.transformer(h)  # (B, L, d_model)

        # 4) pooling
        if self.pool == "last":
            pooled = h[:, -1, :]          # 마지막 시점
        else:
            pooled = h.mean(dim=1)        # 평균 pooling

        # 5) head
        out = self.head(pooled)           # (B, 3)
        return out
