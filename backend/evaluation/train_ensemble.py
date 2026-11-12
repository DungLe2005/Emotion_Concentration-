"""
EduFocus AI — Huấn Luyện & Đánh Giá Mô Hình Baseline EmotionMLP
Sử dụng thuần 1 mình Baseline EmotionMLP trên không gian 544D toàn phần (512D Visual + 32D Hình học/Ngữ nghĩa).
Đã được chứng minh thực nghiệm đạt 86.99% Acc / 79.11% F1 trên RAF-DB, vượt trội hơn Residual MLP.
"""

import sys
import os
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import torch.nn.functional as F

# Thêm sys.path gốc backend
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from models.mlp import EmotionMLP
from dataset.dataset_loader import DatasetRegistry, STANDARD_LABEL_TO_IDX, STANDARD_EMOTIONS
from dataset.dataset_preprocessor import load_dataset_cache, combine_dataset_caches


class FocalLoss(nn.Module):
    """Focal Loss kết hợp Balanced Class Weighting và Label Smoothing nhẹ."""

    def __init__(self, weight=None, gamma=1.5, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce_loss.clamp(max=20.0))
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_mlp_model(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    num_classes: int = 7,
    epochs: int = 100,
    lr: float = 0.0004,
    device: torch.device = None,
):
    """Huấn luyện Baseline EmotionMLP với Focal Loss và Cosine Annealing."""
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    full_weights = np.ones(num_classes, dtype=np.float32)
    unique_cls, counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    for c, cnt in zip(unique_cls, counts):
        full_weights[c] = total_samples / (num_classes * cnt + 1e-6)
    full_weights = full_weights / np.mean(full_weights)
    weight_tensor = torch.tensor(full_weights, dtype=torch.float32).to(device)

    criterion = FocalLoss(weight=weight_tensor, gamma=1.5, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    train_ds = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    val_ds = TensorDataset(torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.long))

    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    best_val_f1 = 0.0
    best_weights = None
    patience = 20
    patience_cnt = 0

    print(f"🚀 Bắt đầu huấn luyện Baseline EmotionMLP ({epochs} epochs, lr={lr})...")
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(by)
        scheduler.step()

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                logits = model(bx)
                preds = torch.argmax(logits, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(by.cpu().numpy())

        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro", zero_division=0)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = copy.deepcopy(model.state_dict())
            patience_cnt = 0
            if epoch % 10 == 0 or epoch == 1:
                print(f"  [Epoch {epoch:03d}] Loss: {total_loss/len(X_train):.4f} | Val Acc: {val_acc*100:.2f}% | Val Macro-F1: {val_f1*100:.2f}% ⭐ (Best)")
        else:
            patience_cnt += 1
            if patience_cnt >= patience:
                print(f"  🛑 Early stopping kích hoạt tại epoch {epoch} (Best Val F1: {best_val_f1*100:.2f}%)")
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)
    return model


def evaluate_dataset(dataset_name: str, epochs: int = 100):
    print("\n" + "=" * 80)
    print(f"🔥 HUẤN LUYỆN & ĐÁNH GIÁ MÔ HÌNH BASELINE EMOTIONMLP [{dataset_name.upper()}] TRÊN 544D")
    print("=" * 80)

    cache_file = os.path.join(BACKEND_DIR, f"cache_{dataset_name.lower()}.npz")
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Không tìm thấy file cache: {cache_file}")

    data = np.load(cache_file)
    X_train, y_train = data["X_train"], data["y_train"]
    X_test, y_test = data["X_test"], data["y_test"]

    if "X_val" in data and len(data["X_val"]) > 0:
        X_val, y_val = data["X_val"], data["y_val"]
    else:
        np.random.seed(42)
        perm = np.random.permutation(len(X_train))
        val_size = int(0.15 * len(X_train))
        X_val, y_val = X_train[perm[:val_size]], y_train[perm[:val_size]]

    feature_dim = X_train.shape[1]
    num_classes = len(STANDARD_EMOTIONS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"  - Kích thước tập Train: {X_train.shape[0]} mẫu")
    print(f"  - Kích thước tập Val: {X_val.shape[0]} mẫu")
    print(f"  - Kích thước tập Test: {X_test.shape[0]} mẫu")
    print(f"  - Không gian đặc trưng: {feature_dim} chiều (Full 544D)")
    print(f"  - Thiết bị tính toán: {device}")

    # Khởi tạo mô hình Baseline EmotionMLP
    model = EmotionMLP(input_dim=feature_dim, num_classes=num_classes).to(device)

    # Huấn luyện
    model = train_mlp_model(
        model,
        X_train,
        y_train,
        X_val,
        y_val,
        num_classes=num_classes,
        epochs=epochs,
        device=device,
    )

    # Đánh giá trên tập kiểm thử Test
    model.eval()
    with torch.no_grad():
        test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        logits = model(test_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    macro_prec = precision_score(y_test, preds, average="macro", zero_division=0)
    macro_rec = recall_score(y_test, preds, average="macro", zero_division=0)

    print("\n" + "=" * 80)
    print(f"🏆 KẾT QUẢ ĐÁNH GIÁ BASELINE EMOTIONMLP [{dataset_name.upper()}]")
    print("=" * 80)
    print(f"  Accuracy:  {acc*100:.2f}%")
    print(f"  Macro-F1:  {macro_f1*100:.2f}%")
    print(f"  Precision: {macro_prec*100:.2f}%")
    print(f"  Recall:    {macro_rec*100:.2f}%")
    print("-" * 80)
    print("\nChi tiết báo cáo phân loại (Classification Report):")
    print(classification_report(y_test, preds, target_names=STANDARD_EMOTIONS, zero_division=0))

    # Lưu trọng số mô hình
    save_path_default = os.path.join(BACKEND_DIR, "emotion_mlp_weights.pth")
    save_path_ds = os.path.join(BACKEND_DIR, f"emotion_mlp_weights_{dataset_name.lower()}.pth")
    torch.save(model.state_dict(), save_path_default)
    torch.save(model.state_dict(), save_path_ds)
    # Tương thích ngược: lưu thêm file emotion_resmlp_weights.pth
    torch.save(model.state_dict(), os.path.join(BACKEND_DIR, "emotion_resmlp_weights.pth"))
    print(f"✅ Đã lưu trọng số Baseline EmotionMLP tại: {save_path_default} và {save_path_ds}")

    return {
        "dataset": dataset_name,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "precision": macro_prec,
        "recall": macro_rec,
    }


def main():
    parser = argparse.ArgumentParser(description="Huấn luyện Baseline EmotionMLP trên 544D")
    parser.add_argument("--datasets", nargs="+", default=["new_data"], help="Danh sách dataset (Mặc định: new_data)")
    parser.add_argument("--epochs", type=int, default=100, help="Số epochs huấn luyện (Mặc định: 100)")
    args = parser.parse_args()

    for ds in args.datasets:
        evaluate_dataset(ds, epochs=args.epochs)


if __name__ == "__main__":
    main()
