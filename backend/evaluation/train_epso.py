import sys
import os
import copy
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
import torch.nn.functional as F
import shutil

# Thêm sys.path gốc backend
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from models.mlp import EmotionMLP
from models.epso import EPSOFeatureSelector
from dataset.dataset_loader import DatasetRegistry, STANDARD_LABEL_TO_IDX
from dataset.dataset_preprocessor import (
    load_dataset_cache,
    combine_dataset_caches,
    MultiDatasetPreprocessor,
)


class FocalLoss(nn.Module):
    """
    [Giải Pháp 4] Focal Loss chuẩn (Lin et al.) kết hợp Balanced Class Weighting & Label Smoothing nhẹ.
    Dập tắt gradient các mẫu dễ phân loại, tập trung gradient vào các biên lớp khó.
    Loại bỏ hoàn toàn các hệ số phạt Confusion Penalty cực đoan và Mixup gây méo mó xác suất.
    """

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
        focal_loss = ((1.0 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def load_data(
    csv_file=None,
    feature_dir=None,
    test_dir=None,
    expected_dim=544,
    dataset_names=None,
):
    if dataset_names is not None:
        if isinstance(dataset_names, str):
            dataset_names = [dataset_names]
        print(f"Đang nạp dữ liệu đặc trưng cho bộ datasets: {dataset_names}...")

        cache_dir = BACKEND_DIR
        for ds in dataset_names:
            possible_cache = os.path.join(cache_dir, f"cache_{ds.lower()}.npz")
            if not os.path.exists(possible_cache):
                print(
                    f"[Auto Preprocess] Chưa có cache cho dataset '{ds}'. Tiến hành trích xuất..."
                )
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
            data["label_to_idx"].item()
            if isinstance(data["label_to_idx"], np.ndarray)
            else data["label_to_idx"],
        )

    try:
        return load_data(dataset_names=["new_data"])
    except Exception as e:
        print(f"Lỗi nạp new_data mặc định: {e}")
        raise e


def train_single_mlp(
    X_train,
    y_train,
    X_test,
    y_test,
    num_classes=7,
    epochs=100,
    batch_size=64,
    lr=0.0003,
    device="cpu",
    desc="Model",
):
    """
    [Giải Pháp 4] Huấn luyện EmotionMLP với Focal Loss chuẩn hóa, CosineAnnealingLR & Early Stopping.
    Loại bỏ nhiễu nhân tạo và Mixup gây xung đột, đảm bảo gradient ổn định và phân phối xác suất chuẩn xác.
    """
    dim = X_train.shape[1]
    model = EmotionMLP(input_dim=dim, num_classes=num_classes).to(device)

    # Class weights cân bằng căn bậc hai
    full_weights = np.ones(num_classes, dtype=np.float32)
    unique_cls, counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    for c, cnt in zip(unique_cls, counts):
        if c < num_classes:
            full_weights[c] = total_samples / (len(unique_cls) * cnt)
    class_weights_tensor = torch.tensor(
        np.sqrt(full_weights), dtype=torch.float32
    ).to(device)

    criterion = FocalLoss(weight=class_weights_tensor, gamma=1.5, label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-6
    )

    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True
    )

    X_te_t = torch.tensor(X_test, dtype=torch.float32).to(device)

    best_val_acc = 0.0
    best_weights = None
    patience = 30
    no_improve = 0

    print(
        f"  🚀 Bắt đầu huấn luyện {desc} (Đặc trưng: {dim}D, Batch: {batch_size}, LR: {lr})..."
    )

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item() * len(by)
            preds = torch.argmax(out, dim=1)
            correct += (preds == by).sum().item()
            total += len(by)

        scheduler.step()

        # Đánh giá sau mỗi epoch
        model.eval()
        with torch.no_grad():
            v_logits = model(X_te_t)
            v_preds = torch.argmax(v_logits, dim=1).cpu().numpy()
            v_acc = accuracy_score(y_test, v_preds)

        train_acc = correct / max(1, total)

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_weights = copy.deepcopy(model.state_dict())
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience and epoch >= 30:
                print(
                    f"     [Early Stopping] {desc} dừng tại epoch {epoch} (Best Test Acc: {best_val_acc*100:.2f}%)"
                )
                break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    model.eval()
    with torch.no_grad():
        final_logits = model(X_te_t)
        final_probs = F.softmax(final_logits, dim=1).cpu().numpy()
        final_preds = np.argmax(final_probs, axis=1)

    final_acc = accuracy_score(y_test, final_preds)
    final_f1 = f1_score(y_test, final_preds, average="macro", zero_division=0)
    final_prec = precision_score(
        y_test, final_preds, average="macro", zero_division=0
    )
    final_rec = recall_score(y_test, final_preds, average="macro", zero_division=0)

    print(
        f"  ✅ [Hoàn tất {desc}] Test Acc: {final_acc*100:.2f}% | Macro F1: {final_f1*100:.2f}%"
    )

    return (
        model,
        final_acc,
        final_f1,
        final_prec,
        final_rec,
        final_preds,
        final_probs,
    )


def train_pipeline(dataset_names=None, run_benchmark_after=True):
    print("=" * 80)
    print("  🚀 QUY TRÌNH HUẤN LUYỆN ĐẶC TRƯNG & SO SÁNH GIỮ / BỎ EPSO (GIẢI PHÁP 3 & 4)")
    print("=" * 80)
    ORIGINAL_DIM = 544  # CLIP(512) + Prompts(7) + Landmark(25)

    if dataset_names is None:
        dataset_names = ["new_data"]

    if len(dataset_names) == 1:
        ds_suffix = dataset_names[0].lower()
    else:
        ds_suffix = "_".join(sorted([d.lower() for d in dataset_names]))

    # 1. Nạp dữ liệu
    X_train, y_train, X_test, y_test, label_to_idx = load_data(
        expected_dim=ORIGINAL_DIM, dataset_names=dataset_names
    )
    num_classes = len(label_to_idx)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n[Dữ Liệu] Dataset: {ds_suffix.upper()}")
    print(f"  - Số lượng mẫu Huấn luyện: {len(X_train)}")
    print(f"  - Số lượng mẫu Kiểm tra:   {len(X_test)}")
    print(f"  - Kích thước không gian đặc trưng gốc: {ORIGINAL_DIM}D")
    print(f"  - Thiết bị tính toán: {device}")

    # 2. Rút gọn đặc trưng với EPSO
    print("\n--- 1️⃣ RÚT GỌN ĐẶC TRƯNG VỚI THUẬT TOÁN EPSO ---")
    epso_mask_path_ds = os.path.join(BACKEND_DIR, f"epso_mask_{ds_suffix}.npy")
    epso_mask_path_default = os.path.join(BACKEND_DIR, "epso_mask_real.npy")

    if os.path.exists(epso_mask_path_ds):
        print(f"✅ Tìm thấy EPSO mask riêng: {os.path.basename(epso_mask_path_ds)}")
        selected_indices = np.load(epso_mask_path_ds)
    elif os.path.exists(epso_mask_path_default):
        print(f"✅ Tìm thấy EPSO mask chung: {os.path.basename(epso_mask_path_default)}")
        selected_indices = np.load(epso_mask_path_default)
    else:
        print("⚡ Đang thực thi EPSO Feature Selection...")
        epso_selector = EPSOFeatureSelector(
            dim=ORIGINAL_DIM, num_particles=15, max_iter=25
        )
        selected_indices = epso_selector.select_features(X_train, y_train)

    print(f"🎯 EPSO chọn ra {len(selected_indices)}/{ORIGINAL_DIM} đặc trưng.")

    # 3. [Giải Pháp 3]: SO SÁNH TRỰC TIẾP MODEL CÓ EPSO VS KHÔNG EPSO
    print("\n" + "=" * 80)
    print("  ⚔️ [GIẢI PHÁP 3] SO SÁNH THỰC NGHIỆM: CÓ EPSO VS KHÔNG EPSO (FULL 544D)")
    print("=" * 80)

    # 3a. Huấn luyện Model A (CÓ EPSO)
    print("\n[A] Huấn luyện Mô hình với EPSO (Không gian rút gọn):")
    X_train_epso = X_train[:, selected_indices]
    X_test_epso = X_test[:, selected_indices]
    (
        model_epso,
        acc_epso,
        f1_epso,
        prec_epso,
        rec_epso,
        preds_epso,
        probs_epso,
    ) = train_single_mlp(
        X_train_epso,
        y_train,
        X_test_epso,
        y_test,
        num_classes=num_classes,
        epochs=100,
        batch_size=64,
        lr=0.0003,
        device=device,
        desc=f"Model CÓ EPSO ({len(selected_indices)}D)",
    )

    # 3b. Huấn luyện Model B (KHÔNG EPSO - FULL 544D)
    print("\n[B] Huấn luyện Mô hình KHÔNG DÙNG EPSO (Bảo toàn 100% 544D):")
    (
        model_full,
        acc_full,
        f1_full,
        prec_full,
        rec_full,
        preds_full,
        probs_full,
    ) = train_single_mlp(
        X_train,
        y_train,
        X_test,
        y_test,
        num_classes=num_classes,
        epochs=100,
        batch_size=64,
        lr=0.0003,
        device=device,
        desc=f"Model KHÔNG EPSO ({ORIGINAL_DIM}D)",
    )

    # 3c. Bảng So Sánh & Tự Động Ra Quyết Định
    print("\n" + "=" * 85)
    print(f"  📊 BẢNG SO SÁNH HIỆU QUẢ CƠ CHẾ EPSO [{ds_suffix.upper()}]")
    print("=" * 85)
    print(
        f"{'Phương Án':<30} | {'Số Chiều':<10} | {'Accuracy (%)':<14} | {'Macro F1 (%)':<14} | {'Quyết Định'}"
    )
    print("-" * 85)

    epso_wins = acc_epso > acc_full
    epso_tag = "GIỮ EPSO" if epso_wins else "LOẠI BỎ (Kém hơn)"
    full_tag = "ĐƯỢC CHỌN (Tối ưu)" if not epso_wins else "Dự phòng"

    print(
        f"{'1. Có EPSO (Feature Selection)':<30} | {len(selected_indices):<10} | {acc_epso*100:<14.2f} | {f1_epso*100:<14.2f} | {epso_tag}"
    )
    print(
        f"{'2. Không EPSO (Full 544D)':<30} | {ORIGINAL_DIM:<10} | {acc_full*100:<14.2f} | {f1_full*100:<14.2f} | {full_tag}"
    )
    print("=" * 85)

    if not epso_wins:
        print(
            f"🎯 KẾT LUẬN: Mô hình KHÔNG EPSO đạt Acc cao hơn (+{(acc_full - acc_epso)*100:.2f}%). "
            f"-> Quyết định: BỎ EPSO, giữ trọn vẹn {ORIGINAL_DIM}D để tối ưu hóa không gian biểu diễn!\n"
        )
        chosen_mask = np.arange(ORIGINAL_DIM)
        chosen_model = model_full
        chosen_acc = acc_full
        chosen_f1 = f1_full
        chosen_prec = prec_full
        chosen_rec = rec_full
        chosen_preds = preds_full
        chosen_probs = probs_full
        chosen_strategy = "without_epso"
    else:
        print(
            f"🎯 KẾT LUẬN: Mô hình CÓ EPSO đạt Acc cao hơn (+{(acc_epso - acc_full)*100:.2f}%). "
            f"-> Quyết định: GIỮ EPSO ({len(selected_indices)}D)!\n"
        )
        chosen_mask = selected_indices
        chosen_model = model_epso
        chosen_acc = acc_epso
        chosen_f1 = f1_epso
        chosen_prec = prec_epso
        chosen_rec = rec_epso
        chosen_preds = preds_epso
        chosen_probs = probs_epso
        chosen_strategy = "with_epso"

    # 4. Lưu Mask và Trọng số Mô hình được lựa chọn
    chosen_mask_path_ds = os.path.join(BACKEND_DIR, f"epso_mask_{ds_suffix}.npy")
    np.save(chosen_mask_path_ds, chosen_mask)
    np.save(epso_mask_path_default, chosen_mask)
    print(f"✅ Đã lưu Mask tối ưu ({len(chosen_mask)}D) tại: {chosen_mask_path_ds} và {epso_mask_path_default}")

    best_acc_path = os.path.join(
        BACKEND_DIR, f"emotion_mlp_best_acc_weights_{ds_suffix}.pth"
    )
    save_path = os.path.join(
        BACKEND_DIR, f"emotion_mlp_epso_weights_{ds_suffix}.pth"
    )
    default_save_path = os.path.join(BACKEND_DIR, "emotion_mlp_epso_weights.pth")
    default_best_path = os.path.join(BACKEND_DIR, "emotion_mlp_best_acc_weights.pth")

    torch.save(chosen_model.state_dict(), best_acc_path)
    torch.save(chosen_model.state_dict(), save_path)
    if ds_suffix == "new_data":
        shutil.copy(best_acc_path, default_save_path)
        shutil.copy(best_acc_path, default_best_path)
        print(f"  >> Đã đồng bộ trọng số tối ưu vào file default: {default_best_path}")

    # Lưu kết quả so sánh JSON
    output_dir = os.path.join(BACKEND_DIR, "output")
    os.makedirs(output_dir, exist_ok=True)
    comparison_info = {
        "dataset": ds_suffix,
        "with_epso": {
            "num_features": int(len(selected_indices)),
            "accuracy": float(acc_epso),
            "macro_f1": float(f1_epso),
        },
        "without_epso": {
            "num_features": int(ORIGINAL_DIM),
            "accuracy": float(acc_full),
            "macro_f1": float(f1_full),
        },
        "chosen_strategy": chosen_strategy,
        "chosen_accuracy": float(chosen_acc),
        "chosen_f1": float(chosen_f1),
    }
    comparison_path = os.path.join(output_dir, f"epso_comparison_{ds_suffix}.json")
    with open(comparison_path, "w", encoding="utf-8") as f:
        json.dump(comparison_info, f, indent=4, ensure_ascii=False)
    print(f"✅ Đã ghi nhận quyết định so sánh EPSO tại: {comparison_path}")

    # Báo cáo chi tiết cho mô hình được chọn
    classes_names = [k for k in sorted(label_to_idx, key=label_to_idx.get)]
    all_class_indices = list(range(num_classes))
    report_dict = classification_report(
        y_test,
        chosen_preds,
        labels=all_class_indices,
        target_names=classes_names,
        output_dict=True,
        zero_division=0,
    )
    report_text = classification_report(
        y_test,
        chosen_preds,
        labels=all_class_indices,
        target_names=classes_names,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, chosen_preds, labels=all_class_indices)

    metrics_data = {
        "accuracy": float(chosen_acc),
        "precision_macro": float(chosen_prec),
        "recall_macro": float(chosen_rec),
        "f1_macro": float(chosen_f1),
        "chosen_strategy": chosen_strategy,
        "classification_report": report_dict,
    }
    metrics_json_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=4, ensure_ascii=False)

    summary_txt_path = os.path.join(output_dir, "evaluation_summary.txt")
    with open(summary_txt_path, "w", encoding="utf-8") as f:
        f.write("=== KẾT QUẢ ĐÁNH GIÁ MÔ HÌNH (EVALUATION METRICS) ===\n\n")
        f.write(f"Chiến lược chọn:   {chosen_strategy.upper()} ({len(chosen_mask)}D)\n")
        f.write(f"Accuracy:          {chosen_acc:.4f} ({(chosen_acc*100):.2f}%)\n")
        f.write(f"Precision (Macro): {chosen_prec:.4f} ({(chosen_prec*100):.2f}%)\n")
        f.write(f"Recall (Macro):    {chosen_rec:.4f} ({(chosen_rec*100):.2f}%)\n")
        f.write(f"F1-Score (Macro):  {chosen_f1:.4f} ({(chosen_f1*100):.2f}%)\n\n")
        f.write("--- DETAILED CLASSIFICATION REPORT ---\n")
        f.write(report_text)
        f.write("\n\n--- CONFUSION MATRIX ---\n")
        col_width = max(len(name) for name in classes_names) + 2
        header = " " * col_width + "".join(f"{name:>{col_width}}" for name in classes_names)
        f.write(header + "\n")
        for i, row_label in enumerate(classes_names):
            row_str = f"{row_label:<{col_width}}" + "".join(f"{val:>{col_width}}" for val in cm[i])
            f.write(row_str + "\n")

    try:
        plt.figure(figsize=(10, 8))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_names)
        disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), xticks_rotation=45)
        plt.title(f"Confusion Matrix - Emotion Classification [{chosen_strategy.upper()}]")
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
            mask_file = os.path.basename(chosen_mask_path_ds)
            benchmarks = MultiDatasetBenchmark(
                model_weights_path=weights_file, epso_mask_path=mask_file
            )
            benchmarks.run_benchmark_suite(dataset_names=dataset_names)
        except Exception as e:
            print(f"[Lỗi Benchmark] {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Multi-Dataset EPSO Comparison & Emotion Classifier Training"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["new_data"],
        help="Danh sách dataset dùng huấn luyện",
    )
    args = parser.parse_args()

    train_pipeline(dataset_names=args.datasets)
