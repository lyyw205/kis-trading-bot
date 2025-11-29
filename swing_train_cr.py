# train_cr_swing.py
"""
CR 고수익 단타용 멀티호라이즌 모델 학습 스크립트

- 입력 데이터: datasets/cr_swing/cr_swing_*.npz
- 모델: swing_model_cr.CrSwingModel
- 출력: models/cr_swing/cr_swing_best.pt, cr_swing_last.pt
"""

import os
import time
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from swing_model_cr import CrSwingModel, CrSwingConfig


DATA_DIR = "datasets/cr_swing"
MODEL_DIR = "models/cr_swing"

BATCH_SIZE = 256
NUM_EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-5
GRAD_CLIP = 1.0  # 0이면 사용 안 함

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["X"], data["Y"]


def prepare_dataloaders() -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    train/valid/test npz를 로드해서 DataLoader로 변환.
    - 만약 train/valid가 비어 있으면 fallback 로직 적용:
      * train 비었으면: test를 train으로 사용
      * valid 비었으면: train의 일부분을 valid로 분할
    """
    train_path = os.path.join(DATA_DIR, "cr_swing_train.npz")
    valid_path = os.path.join(DATA_DIR, "cr_swing_valid.npz")
    test_path = os.path.join(DATA_DIR, "cr_swing_test.npz")

    X_train, Y_train = (np.empty((0, 60, 11)), np.empty((0, 3)))
    X_valid, Y_valid = (np.empty((0, 60, 11)), np.empty((0, 3)))
    X_test, Y_test = load_npz(test_path)

    # train/valid 파일이 존재하면 로드 시도
    if os.path.exists(train_path):
        Xt, Yt = load_npz(train_path)
        if Xt.shape[0] > 0:
            X_train, Y_train = Xt, Yt

    if os.path.exists(valid_path):
        Xv, Yv = load_npz(valid_path)
        if Xv.shape[0] > 0:
            X_valid, Y_valid = Xv, Yv

    # 🔥 fallback 1: train이 비어 있으면 test를 train으로 사용
    if X_train.shape[0] == 0:
        print("⚠️ train 데이터가 비어 있어 test를 train으로 사용합니다.")
        X_train, Y_train = X_test, Y_test

    # 🔥 fallback 2: valid가 비어 있으면 train에서 일부를 떼어 사용 (시간 무시, 단순 분할)
    if X_valid.shape[0] == 0:
        print("⚠️ valid 데이터가 비어 있어 train의 10%를 valid로 분할합니다.")
        n_train = X_train.shape[0]
        n_valid = max(int(n_train * 0.1), 1)
        # 뒤쪽 10%를 valid로 사용 (시간순이라고 가정)
        X_valid, Y_valid = X_train[-n_valid:], Y_train[-n_valid:]
        X_train, Y_train = X_train[:-n_valid], Y_train[:-n_valid]

    # numpy → tensor
    X_train_t = torch.from_numpy(X_train).float()
    Y_train_t = torch.from_numpy(Y_train).float()
    X_valid_t = torch.from_numpy(X_valid).float()
    Y_valid_t = torch.from_numpy(Y_valid).float()
    X_test_t = torch.from_numpy(X_test).float()
    Y_test_t = torch.from_numpy(Y_test).float()

    train_ds = TensorDataset(X_train_t, Y_train_t)
    valid_ds = TensorDataset(X_valid_t, Y_valid_t)
    test_ds = TensorDataset(X_test_t, Y_test_t)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    print("\n📊 데이터셋 크기:")
    print(f"  - Train: {len(train_ds)} 샘플")
    print(f"  - Valid: {len(valid_ds)} 샘플")
    print(f"  - Test : {len(test_ds)} 샘플")

    return train_loader, valid_loader, test_loader


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict:
    """
    간단한 회귀 지표 계산 (MSE, MAE)
    pred, target: (batch, 3)
    """
    mse = nn.functional.mse_loss(pred, target, reduction="mean").item()
    mae = nn.functional.l1_loss(pred, target, reduction="mean").item()
    return {"mse": mse, "mae": mae}


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    n_samples = 0

    for Xb, Yb in loader:
        Xb = Xb.to(DEVICE)
        Yb = Yb.to(DEVICE)

        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, Yb)
        loss.backward()

        if GRAD_CLIP and GRAD_CLIP > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        bs = Xb.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

    return total_loss / max(n_samples, 1)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    n_samples = 0

    all_preds = []
    all_targets = []

    for Xb, Yb in loader:
        Xb = Xb.to(DEVICE)
        Yb = Yb.to(DEVICE)

        pred = model(Xb)
        loss = criterion(pred, Yb)

        bs = Xb.size(0)
        total_loss += loss.item() * bs
        n_samples += bs

        all_preds.append(pred.cpu())
        all_targets.append(Yb.cpu())

    avg_loss = total_loss / max(n_samples, 1)

    metrics = {}
    if all_preds:
        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(preds, targets)

    metrics["loss"] = avg_loss
    return metrics


def save_checkpoint(model, config, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model_state_dict": model.state_dict(),
        "config": config.__dict__,
    }
    torch.save(state, path)
    print(f"💾 모델 저장: {path}")


def main():
    print("🚀 CR Swing 모델 학습 시작")

    train_loader, valid_loader, test_loader = prepare_dataloaders()

    # config: 데이터셋 shape 기준으로 feat_dim 자동 지정
    example_batch = next(iter(train_loader))[0]
    seq_len = example_batch.shape[1]
    feat_dim = example_batch.shape[2]

    cfg = CrSwingConfig(
        seq_len=seq_len,
        feat_dim=feat_dim,
        d_model=64,
        num_transformer_layers=1,
        num_heads=4,
        dim_feedforward=128,
        dropout=0.1,
        output_dim=3,
    )

    model = CrSwingModel(cfg).to(DEVICE)
    print(model)

    # 손실함수: 스윙 수익률 회귀 → MSE 또는 SmoothL1
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    best_valid_loss = float("inf")
    best_epoch = -1

    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        valid_metrics = evaluate(model, valid_loader, criterion)
        dt = time.time() - t0

        print(f"\n[Epoch {epoch}/{NUM_EPOCHS}] ({dt:.1f}s)")
        print(f"  - Train Loss : {train_loss:.6f}")
        print(
            f"  - Valid Loss : {valid_metrics['loss']:.6f} "
            f"(MSE={valid_metrics['mse']:.6f}, MAE={valid_metrics['mae']:.6f})"
        )

        # 베스트 모델 저장
        if valid_metrics["loss"] < best_valid_loss:
            best_valid_loss = valid_metrics["loss"]
            best_epoch = epoch
            save_checkpoint(model, cfg, os.path.join(MODEL_DIR, "cr_swing_best.pt"))

        # 마지막 모델 항상 저장
        save_checkpoint(model, cfg, os.path.join(MODEL_DIR, "cr_swing_last.pt"))

    print(f"\n✅ 학습 완료. Best epoch: {best_epoch}, best valid loss: {best_valid_loss:.6f}")

    # 최종적으로 test 셋에서 성능 한 번 평가
    best_state = torch.load(os.path.join(MODEL_DIR, "cr_swing_best.pt"), map_location=DEVICE)
    model.load_state_dict(best_state["model_state_dict"])

    test_metrics = evaluate(model, test_loader, criterion)
    print(
        f"\n📊 Test 결과:"
        f"\n  - Loss: {test_metrics['loss']:.6f}"
        f"\n  - MSE : {test_metrics['mse']:.6f}"
        f"\n  - MAE : {test_metrics['mae']:.6f}"
    )


if __name__ == "__main__":
    main()
