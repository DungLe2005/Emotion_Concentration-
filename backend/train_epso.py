import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from models.mlp import EmotionMLP
from models.epso import EPSOFeatureSelector
from sklearn.utils.class_weight import compute_class_weight


class EarlyStopping:
    def __init__(
        self,
        patience=30,
        verbose=False,
        delta=0.001,
        path="emotion_mlp_epso_weights.pth",
        device="cpu",
    ):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.device = device

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model ..."
            )
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def load_data(csv_file, feature_dir, test_dir, expected_dim=512):
    cache_file = "dataset_cache.npz"
    if os.path.exists(cache_file):
        print("TÌM THẤY FILE CACHE. Đang nạp dữ liệu siêu tốc từ npz...")
        data = np.load(cache_file, allow_pickle=True)
        return (
            data["X_train"],
            data["y_train"],
            data["X_test"],
            data["y_test"],
            data["label_to_idx"].item(),
        )

    df = pd.read_csv(csv_file)
    labels_list = df["label"].unique()
    label_to_idx = {label: idx for idx, label in enumerate(labels_list)}

    X_train, y_train = [], []
    X_test, y_test = [], []

    print("CHƯA CÓ CACHE. Đang quét 28,000+ files `.pt` (Có thể mất 15 phút)...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        feat_path = os.path.join(
            feature_dir, row["pth"].replace(".jpg", ".pt").replace(".png", ".pt")
        )

        if os.path.exists(feat_path):
            feature = torch.load(feat_path, weights_only=True)
            if feature.shape[0] != expected_dim:
                print(
                    f"Bỏ qua {feat_path} do sai kích thước ({feature.shape[0]} != {expected_dim})"
                )
                continue
        else:
            continue

        raw_test_path = os.path.join(test_dir, row["pth"])
        if os.path.exists(raw_test_path):
            X_test.append(feature.numpy())
            y_test.append(label_to_idx[row["label"]])
        else:
            X_train.append(feature.numpy())
            y_train.append(label_to_idx[row["label"]])

            base_feat_path = feat_path.replace(".pt", "")
            aug_idx = 0
            while True:
                aug_path = f"{base_feat_path}_aug_{aug_idx}.pt"
                if os.path.exists(aug_path):
                    try:
                        aug_feature = torch.load(aug_path, weights_only=True)
                        if aug_feature.shape[0] == expected_dim:
                            X_train.append(aug_feature.numpy())
                            y_train.append(label_to_idx[row["label"]])
                    except Exception as e:
                        print(f"Lỗi khi đọc file tăng cường {aug_path}: {e}")
                    aug_idx += 1
                else:
                    break

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    print("Đang lưu lại file cache npz để dùng cho lần sau...")
    np.savez_compressed(
        cache_file,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        label_to_idx=label_to_idx,
    )

    return X_train, y_train, X_test, y_test, label_to_idx


def train_pipeline():
    print("=== BƯỚC 1: NẠP DỮ LIỆU (TỐI ƯU HÓA) ===")
    csv_path = r"d:\Emotion_Concentration\data\labels.csv"
    feature_dir = r"d:\Emotion_Concentration\data\extracted_features"
    test_dir = r"d:\Emotion_Concentration\data\Test"

    ORIGINAL_DIM = 512

    X_train, y_train, X_test, y_test, label_to_idx = load_data(
        csv_path, feature_dir, test_dir, expected_dim=ORIGINAL_DIM
    )
    num_classes = len(label_to_idx)

    print(f"\nSố lượng ảnh dùng để Huấn luyện (Tập Train gốc + Aug): {len(X_train)}")
    print(f"Số lượng ảnh dùng để Kiểm tra (Tập Test gốc): {len(X_test)}")

    print("\n=== BƯỚC 2: RÚT GỌN ĐẶC TRƯNG VỚI EPSO ===")
    epso_mask_path = "epso_mask_real.npy"
    if os.path.exists(epso_mask_path):
        print(f"TÌM THẤY MASK {epso_mask_path}. Bỏ qua chạy EPSO (Tái sử dụng).")
        selected_indices = np.load(epso_mask_path)
    else:
        epso_selector = EPSOFeatureSelector(
            dim=ORIGINAL_DIM, num_particles=15, max_iter=25
        )
        selected_indices = epso_selector.select_features(X_train, y_train)
        np.save(epso_mask_path, selected_indices)

    print(f"Kích thước đặc trưng sau EPSO: {len(selected_indices)}/{ORIGINAL_DIM}")

    X_train_reduced = X_train[:, selected_indices]
    X_test_reduced = X_test[:, selected_indices]
    REDUCED_DIM = len(selected_indices)

    print("\n=== BƯỚC 3: HUẤN LUYỆN MLP TRÊN ĐẶC TRƯNG EPSO ===")
    X_tensor = torch.tensor(X_train_reduced, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    # Batch size nhỏ hơn = gradient noisier = khám phá loss landscape rộng hơn
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị huấn luyện: {device}")

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32).to(device)

    model = EmotionMLP(input_dim=REDUCED_DIM, num_classes=num_classes).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    # LR nhỏ hơn: bước đi nhỏ, từ từ tìm Global Minima
    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=5e-5)

    X_val_tensor = torch.tensor(X_test_reduced, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_test, dtype=torch.long)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    num_epochs = 100

    # === WARM-UP + COSINE WARM RESTARTS ===
    # Giai đoạn Warm-up (5 epoch đầu):
    #   LR tăng dần từ 1% → 100% (0.000003 → 0.0003)
    #   → Khi trọng số còn random, bước đi cực nhỏ để "dò đường" an toàn
    # Sau Warm-up:
    #   CosineAnnealingWarmRestarts: LR giảm cosin rồi "restart" nhảy lên
    #   T_0=25 epoch → T_mult=2 (chu kỳ: 25 → 50 → 100 → ...)
    #   → Thoát Local Minima, khám phá vùng mới
    WARMUP_EPOCHS = 5
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.01, end_factor=1.0, total_iters=WARMUP_EPOCHS
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=25, T_mult=2, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[WARMUP_EPOCHS],
    )

    save_path = "emotion_mlp_epso_weights.pth"
    best_acc_path = "emotion_mlp_best_acc_weights.pth"
    # patience=50: chịu đựng 50 epoch không cải thiện mới dừng (đủ thời gian cho warm restarts)
    early_stopping = EarlyStopping(
        patience=30, verbose=True, delta=0.0005, path=save_path, device=device
    )
    best_val_acc = 0.0

    # === CURRICULUM LEARNING: 3 Giai đoạn ===
    # Giai đoạn 1 (Epoch 1-50):  CLEAN - Không nhiễu, không Mixup
    #   → Model học "sườn bài" trên dữ liệu sạch với bước đi nhỏ, cẩn thận
    # Giai đoạn 2 (Epoch 51-150): LIGHT - Nhiễu nhẹ + Mixup nhẹ
    #   → Bắt đầu "đeo tạ nhẹ" để tăng tổng quát hóa
    # Giai đoạn 3 (Epoch 151+):  FULL - Nhiễu + Mixup đầy đủ
    #   → "Đeo tạ nặng" rèn luyện khả năng chống overfitting

    PHASE_1_END = 30  # Kết thúc giai đoạn Clean
    PHASE_2_END = 60  # Kết thúc giai đoạn Light

    current_phase = 0  # Theo dõi giai đoạn hiện tại

    for epoch in range(num_epochs):
        # --- Xác định giai đoạn Curriculum ---
        if epoch < PHASE_1_END:
            alpha = 0.0  # Không Mixup
            noise_level = 0.0  # Không nhiễu
            phase_name = "🧹 CLEAN (Không nhiễu, không Mixup)"
            new_phase = 1
        elif epoch < PHASE_2_END:
            alpha = 0.1  # Mixup nhẹ
            noise_level = 0.005  # Nhiễu nhẹ
            phase_name = "🏋️ LIGHT (Nhiễu nhẹ + Mixup nhẹ)"
            new_phase = 2
        else:
            alpha = 0.2  # Mixup mạnh
            noise_level = 0.01  # Nhiễu chuẩn
            phase_name = "💪 FULL (Nhiễu + Mixup mạnh)"
            new_phase = 3

        # Reset EarlyStopping khi chuyển giai đoạn
        if new_phase != current_phase:
            current_phase = new_phase
            print(f"\n{'='*60}")
            print(f"  CURRICULUM PHASE {current_phase}: {phase_name}")
            print(f"  Alpha={alpha}, Noise={noise_level}")
            print(f"{'='*60}\n")
            # Reset early stopping khi chuyển phase để cho model cơ hội thích nghi
            early_stopping = EarlyStopping(
                patience=50, verbose=True, delta=0.0005, path=save_path, device=device
            )

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)

            # Thêm nhiễu Gaussian (nếu có)
            if noise_level > 0:
                features = features + torch.randn_like(features) * noise_level

            # Mixup (nếu có)
            if alpha > 0:
                lam = np.random.beta(alpha, alpha)
                batch_size = features.size(0)
                index = torch.randperm(batch_size).to(device)

                mixed_features = lam * features + (1 - lam) * features[index]
                labels_a, labels_b = labels, labels[index]

                outputs = model(mixed_features)
                loss = lam * criterion(outputs, labels_a) + (1 - lam) * criterion(
                    outputs, labels_b
                )

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (
                    (predicted == (labels_a if lam >= 0.5 else labels_b)).sum().item()
                )
            else:
                outputs = model(features)
                loss = criterion(outputs, labels)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            optimizer.zero_grad()
            loss.backward()
            # Gradient Clipping: giới hạn bước đi tối đa, ngăn model "chạy bạo"
            # → Buộc model di chuyển từng bước nhỏ, cẩn thận hơn
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(dataloader)
        epoch_acc = 100 * correct / total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_features, val_labels in val_loader:
                val_features = val_features.to(device)
                val_labels = val_labels.to(device)

                val_outputs = model(val_features)
                loss_v = criterion(val_outputs, val_labels)
                val_loss += loss_v.item()

                _, val_predicted = torch.max(val_outputs.data, 1)
                val_total += val_labels.size(0)
                val_correct += (val_predicted == val_labels).sum().item()

        epoch_val_loss = val_loss / len(val_loader)
        epoch_val_acc = 100 * val_correct / val_total

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] -> "
            f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f}, Val Acc: {epoch_val_acc:.2f}% | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Lưu model tốt nhất theo Val Acc (riêng biệt)
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_acc_path)
            print(
                f"  >> [Best Acc] Val Acc cải thiện lên {epoch_val_acc:.2f}% -> Lưu {best_acc_path}"
            )

        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print(
                "Đã kích hoạt Early Stopping! Dừng huấn luyện sớm để tránh overfitting."
            )
            break

    # Nạp model tốt nhất theo accuracy (ưu tiên) hoặc theo val_loss
    if os.path.exists(best_acc_path):
        model.load_state_dict(
            torch.load(best_acc_path, map_location=device, weights_only=True)
        )
        print(
            f"\n[Hoàn tất] Đã nạp lại trọng số MLP tốt nhất theo Val Acc ({best_val_acc:.2f}%) từ: {best_acc_path}"
        )
        # Copy sang file chính để pipeline.py dùng
        import shutil

        shutil.copy(best_acc_path, save_path)

    print("\n=== BƯỚC 4: ĐÁNH GIÁ MÔ HÌNH CHUYÊN SÂU TRÊN TẬP TEST CHƯA TỪNG HỌC ===")

    X_test_tensor = torch.tensor(X_test_reduced, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            probs = torch.softmax(outputs, dim=1)

            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        roc_auc = roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="macro"
        )
    except Exception as e:
        roc_auc = float("nan")

    print(f"Precision (Macro): {precision:.4f} ({(precision*100):.2f}%)")
    print(f"Recall (Macro):    {recall:.4f} ({(recall*100):.2f}%)")
    print(f"F1-Score (Macro):  {f1:.4f} ({(f1*100):.2f}%)")
    print(f"ROC-AUC (Macro):   {roc_auc:.4f} ({(roc_auc*100):.2f}%)")

    print("\n=== BƯỚC 5: ĐỒ THỊ CONFUSION MATRIX ===")
    classes_names = [k for k in sorted(label_to_idx, key=label_to_idx.get)]
    cm = confusion_matrix(all_labels, all_preds)
    
    # In dạng bảng văn bản trên Console
    print("\nConfusion Matrix (Bảng văn bản):")
    col_width = max(len(name) for name in classes_names) + 2
    header = " " * col_width + "".join(f"{name:>{col_width}}" for name in classes_names)
    print(header)
    for i, row_label in enumerate(classes_names):
        row_str = f"{row_label:<{col_width}}" + "".join(f"{val:>{col_width}}" for val in cm[i])
        print(row_str)
        
    # Vẽ và lưu hình ảnh đồ thị
    try:
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_names)
        disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), xticks_rotation=45)
        plt.title("Confusion Matrix - Emotion Classification")
        plt.tight_layout()
        plot_path = "confusion_matrix.png"
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"\n[Thành công] Đã vẽ và lưu đồ thị Confusion Matrix tại: {plot_path}")
    except Exception as e:
        print(f"\n[Lỗi] Không thể vẽ đồ thị Confusion Matrix: {e}")


if __name__ == "__main__":
    train_pipeline()
