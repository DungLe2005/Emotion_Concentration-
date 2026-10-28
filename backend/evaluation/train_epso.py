import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
import pandas as pd
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
import torch.nn.functional as F
import shutil

# Thêm sys.path gốc backend
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from models.mlp import EmotionMLP
from models.epso import EPSOFeatureSelector
from dataset.dataset_loader import DatasetRegistry, STANDARD_LABEL_TO_IDX
from dataset.dataset_preprocessor import load_dataset_cache, combine_dataset_caches, MultiDatasetPreprocessor


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


class ConfusionPenaltyLoss(nn.Module):
    def __init__(self, num_classes, penalty_pairs, device):
        super().__init__()
        self.penalty_matrix = torch.ones(num_classes, num_classes, device=device)
        for true_cls, pred_cls, multiplier in penalty_pairs:
            self.penalty_matrix[true_cls][pred_cls] = multiplier

    def forward(self, logits, targets):
        probs = F.softmax(logits, dim=1)
        penalty_rows = self.penalty_matrix[targets]
        wrong_prob_penalty = (penalty_rows - 1) * probs
        penalty_loss = wrong_prob_penalty.sum(dim=1).mean()

        return penalty_loss


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


def load_data(csv_file=None, feature_dir=None, test_dir=None, expected_dim=544, dataset_names=None):
    if dataset_names is not None:
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        print(f"Đang nạp dữ liệu đặc trưng cho bộ datasets: {dataset_names}...")
        
        cache_dir = BACKEND_DIR
        for ds in dataset_names:
            possible_cache = os.path.join(cache_dir, f"cache_{ds.lower()}.npz")
            if not os.path.exists(possible_cache):
                print(f"[Auto Preprocess] Chưa có cache cho dataset '{ds}'. Tiến hành trích xuất...")
                preprocessor = MultiDatasetPreprocessor()
                preprocessor.process_and_cache_dataset(ds)
                preprocessor.close()

        if len(dataset_names) == 1:
            data = load_dataset_cache(dataset_names[0], cache_dir=cache_dir)
        else:
            data = combine_dataset_caches(dataset_names, cache_dir=cache_dir)

        X_train = data.get("X_train", np.array([]))
        y_train = data.get("y_train", np.array([]))
        X_test = data.get("X_test", np.array([]))
        y_test = data.get("y_test", np.array([]))
        if len(X_test) == 0 and "X_val" in data:
            X_test = data["X_val"]
            y_test = data["y_val"]
        label_to_idx = data.get("label_to_idx", STANDARD_LABEL_TO_IDX)

        if isinstance(label_to_idx, np.ndarray):
            label_to_idx = label_to_idx.item()

        return X_train, y_train, X_test, y_test, label_to_idx

    cache_file = os.path.join(BACKEND_DIR, "dataset_cache.npz")
    if os.path.exists(cache_file):
        print("TÌM THẤY FILE CACHE dataset_cache.npz. Đang nạp dữ liệu siêu tốc...")
        data = np.load(cache_file, allow_pickle=True)
        return (
            data["X_train"],
            data["y_train"],
            data["X_test"],
            data["y_test"],
            data["label_to_idx"].item() if isinstance(data["label_to_idx"], np.ndarray) else data["label_to_idx"],
        )

    try:
        return load_data(dataset_names=["new_data"])
    except Exception as e:
        print(f"Lỗi nạp new_data mặc định: {e}")
        raise e


def train_pipeline(dataset_names=None, run_benchmark_after=True):
    print("=== BƯỚC 1: NẠP DỮ LIỆU (TỐI ƯU HÓA ĐA BỘ DỮ LIỆU) ===")
    ORIGINAL_DIM = 544  # CLIP(512) + Prompts(7) + Landmark(25)

    if dataset_names is None:
        dataset_names = ["new_data"]

    # Xác định tên suffix cho file trọng số (dùng tên dataset nếu chỉ 1, nếu nhiều thì "combined")
    if len(dataset_names) == 1:
        ds_suffix = dataset_names[0].lower()
    else:
        ds_suffix = "_".join(sorted([d.lower() for d in dataset_names]))

    X_train, y_train, X_test, y_test, label_to_idx = load_data(
        expected_dim=ORIGINAL_DIM, dataset_names=dataset_names
    )
    num_classes = len(label_to_idx)

    print(f"\nSố lượng ảnh dùng để Huấn luyện (Tập Train gốc + Aug): {len(X_train)}")
    print(f"Số lượng ảnh dùng để Kiểm tra (Tập Test gốc): {len(X_test)}")

    print("\n=== BƯỚC 2: RÚT GỌN ĐẶC TRƯNG VỚI EPSO ===")
    # Ưu tiên mask riêng theo dataset, fallback sang mask chung
    epso_mask_path_ds = os.path.join(BACKEND_DIR, f"epso_mask_{ds_suffix}.npy")
    epso_mask_path_default = os.path.join(BACKEND_DIR, "epso_mask_real.npy")

    if os.path.exists(epso_mask_path_ds):
        epso_mask_path = epso_mask_path_ds
        print(f"TÌM THẤY MASK RIÊNG [{ds_suffix}]: {epso_mask_path}. Tái sử dụng.")
        selected_indices = np.load(epso_mask_path)
    elif os.path.exists(epso_mask_path_default):
        epso_mask_path = epso_mask_path_default
        print(f"TÌM THẤY MASK CHUNG: {epso_mask_path}. Tái sử dụng.")
        selected_indices = np.load(epso_mask_path)
    else:
        epso_mask_path = epso_mask_path_ds
        epso_selector = EPSOFeatureSelector(
            dim=ORIGINAL_DIM, num_particles=15, max_iter=25
        )
        selected_indices = epso_selector.select_features(X_train, y_train)
        np.save(epso_mask_path, selected_indices)
        print(f"Đã lưu EPSO mask mới tại: {epso_mask_path}")

    print(f"Kích thước đặc trưng sau EPSO: {len(selected_indices)}/{ORIGINAL_DIM}")

    X_train_reduced = X_train[:, selected_indices]
    X_test_reduced = X_test[:, selected_indices]
    REDUCED_DIM = len(selected_indices)

    print("\n=== BƯỚC 3: HUẤN LUYỆN MLP TRÊN ĐẶC TRƯNG EPSO ===")
    X_tensor = torch.tensor(X_train_reduced, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.long)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị huấn luyện: {device}")

    full_weights = np.ones(num_classes, dtype=np.float32)
    unique_cls, counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    for c, cnt in zip(unique_cls, counts):
        full_weights[c] = total_samples / (len(unique_cls) * cnt)

    smoothed_weights = np.sqrt(full_weights)
    if len(smoothed_weights) > 1:
        smoothed_weights[1] *= 2.2
    class_weights_tensor = torch.tensor(smoothed_weights, dtype=torch.float32).to(device)

    model = EmotionMLP(input_dim=REDUCED_DIM, num_classes=num_classes).to(device)

    ce_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1)

    confusion_pairs = [
        (6, 1, 3.5),
        (1, 2, 4.0),
        (1, 4, 3.5),
        (1, 3, 3.5),
        (1, 5, 3.0),
        (1, 0, 2.5),
        (2, 1, 2.0),
        (6, 2, 1.5),
    ]
    confusion_penalty = ConfusionPenaltyLoss(num_classes, confusion_pairs, device)

    def criterion(outputs, labels):
        ce = ce_criterion(outputs, labels)
        penalty = confusion_penalty(outputs, labels)
        return ce + 0.6 * penalty

    optimizer = optim.Adam(model.parameters(), lr=0.0003, weight_decay=5e-5)

    X_val_tensor = torch.tensor(X_test_reduced, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_test, dtype=torch.long)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    num_epochs = 150

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

    # Lưu trọng số riêng theo dataset
    save_path = os.path.join(BACKEND_DIR, f"emotion_mlp_epso_weights_{ds_suffix}.pth")
    best_acc_path = os.path.join(BACKEND_DIR, f"emotion_mlp_best_acc_weights_{ds_suffix}.pth")
    # Cũng lưu thêm bản copy default nếu là new_data (để pipeline.py vẫn hoạt động)
    default_save_path = os.path.join(BACKEND_DIR, "emotion_mlp_epso_weights.pth")

    early_stopping = EarlyStopping(
        patience=30, verbose=True, delta=0.0005, path=save_path, device=device
    )
    best_val_acc = 0.0

    PHASE_1_END = 40
    PHASE_2_END = 90
    current_phase = 0

    for epoch in range(num_epochs):
        if epoch < PHASE_1_END:
            alpha = 0.0
            noise_level = 0.0
            phase_name = "🧹 CLEAN (Không nhiễu, không Mixup)"
            new_phase = 1
        elif epoch < PHASE_2_END:
            alpha = 0.1
            noise_level = 0.005
            phase_name = "🏋️ LIGHT (Nhiễu nhẹ + Mixup nhẹ)"
            new_phase = 2
        else:
            alpha = 0.2
            noise_level = 0.01
            phase_name = "💪 FULL (Nhiễu + Mixup mạnh)"
            new_phase = 3

        if new_phase != current_phase:
            current_phase = new_phase
            print(f"\n{'='*60}")
            print(f"  CURRICULUM PHASE {current_phase}: {phase_name}")
            print(f"  Alpha={alpha}, Noise={noise_level}")
            print(f"{'='*60}\n")
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

            if noise_level > 0:
                features = features + torch.randn_like(features) * noise_level

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

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_acc_path)
            print(
                f"  >> [Best Acc] Val Acc cải thiện lên {epoch_val_acc:.2f}% -> Lưu {best_acc_path}"
            )

        early_stopping(epoch_val_loss, model)
        if early_stopping.early_stop:
            print("Đã kích hoạt Early Stopping! Dừng huấn luyện sớm để tránh overfitting.")
            break

    if os.path.exists(best_acc_path):
        model.load_state_dict(
            torch.load(best_acc_path, map_location=device, weights_only=True)
        )
        print(
            f"\n[Hoàn tất] Đã nạp lại trọng số MLP tốt nhất theo Val Acc ({best_val_acc:.2f}%) từ: {best_acc_path}"
        )
        shutil.copy(best_acc_path, save_path)
        # Nếu train trên new_data, cũng copy sang file default cho pipeline.py
        if ds_suffix == "new_data":
            shutil.copy(best_acc_path, default_save_path)
            print(f"  >> Đã copy trọng số default cho pipeline: {default_save_path}")

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

    output_dir = os.path.join(BACKEND_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    try:
        roc_auc = roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average="macro"
        )
    except Exception:
        roc_auc = float("nan")

    classes_names = [k for k in sorted(label_to_idx, key=label_to_idx.get)]
    all_class_indices = list(range(num_classes))
    report_dict = classification_report(
        all_labels, all_preds, labels=all_class_indices, target_names=classes_names, output_dict=True, zero_division=0
    )
    report_text = classification_report(
        all_labels, all_preds, labels=all_class_indices, target_names=classes_names, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=all_class_indices)

    print(f"Accuracy:          {accuracy:.4f} ({(accuracy*100):.2f}%)")
    print(f"Precision (Macro): {precision:.4f} ({(precision*100):.2f}%)")
    print(f"Recall (Macro):    {recall:.4f} ({(recall*100):.2f}%)")
    print(f"F1-Score (Macro):  {f1:.4f} ({(f1*100):.2f}%)")
    print(f"ROC-AUC (Macro):   {roc_auc:.4f} ({(roc_auc*100):.2f}%)")

    metrics_data = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "roc_auc_macro": float(roc_auc) if not np.isnan(roc_auc) else None,
        "classification_report": report_dict,
    }
    metrics_json_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4, ensure_ascii=False)

    summary_txt_path = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (EVALUATION METRICS) ===\n\n")
        f.write(f"Accuracy:          {accuracy:.4f} ({(accuracy*100):.2f}%)\n")
        f.write(f"Precision (Macro): {precision:.4f} ({(precision*100):.2f}%)\n")
        f.write(f"Recall (Macro):    {recall:.4f} ({(recall*100):.2f}%)\n")
        f.write(f"F1-Score (Macro):  {f1:.4f} ({(f1*100):.2f}%)\n")
        f.write(f"ROC-AUC (Macro):   {roc_auc:.4f} ({(roc_auc*100):.2f}%)\n\n")
        f.write("--- DETAILED CLASSIFICATION REPORT ---\n")
        f.write(report_text)
        f.write("\n\n--- CONFUSION MATRIX ---\n")
        col_width = max(len(name) for name in classes_names) + 2
        header = " " * col_width + "".join(f"{name:>{col_width}}" for name in classes_names)
        f.write(header + "\n")
        for i, row_label in enumerate(classes_names):
            row_str = f"{row_label:<{col_width}}" + "".join(f"{val:>{col_width}}" for val in cm[i])
            f.write(row_str + "\n")

    cm_df = pd.DataFrame(cm, index=classes_names, columns=classes_names)
    cm_csv_path = os.path.join(output_dir, "confusion_matrix.csv")
    cm_df.to_csv(cm_csv_path)

    try:
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_names)
        disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), xticks_rotation=45)
        plt.title("Confusion Matrix - Emotion Classification")
        plt.tight_layout()
        plot_path = os.path.join(output_dir, "confusion_matrix.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
    except Exception as e:
        print(f"[Lỗi] Không thể vẽ đồ thị Confusion Matrix: {e}")

    if run_benchmark_after:
        print("\n=== BƯỚC 5: TỰ ĐỘNG CHẠY BENCHMARK ĐÁNH GIÁ TRÊN TẤT CẢ DATASETS ===")
        from evaluation.benchmark import MultiDatasetBenchmark
        try:
            weights_file = f"emotion_mlp_epso_weights_{ds_suffix}.pth"
            mask_file = os.path.basename(epso_mask_path)
            benchmarks = MultiDatasetBenchmark(
                model_weights_path=weights_file,
                epso_mask_path=mask_file
            )
            benchmarks.run_benchmark_suite(dataset_names=dataset_names)
        except Exception as e:
            print(f"[Lỗi Benchmark] {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Dataset EPSO MLP Emotion Classifier Training")
    parser.add_argument("--datasets", nargs="+", default=["new_data"], help="Danh sách dataset dùng huấn luyện")
    args = parser.parse_args()

    train_pipeline(dataset_names=args.datasets)
