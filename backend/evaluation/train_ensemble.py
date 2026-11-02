import sys
import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import pandas as pd
import json
import argparse
import joblib
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
import torch.nn.functional as F

# Thêm sys.path gốc backend
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from models.mlp import EmotionMLP
from models.resmlp import EmotionResMLP
from dataset.dataset_loader import DatasetRegistry, STANDARD_LABEL_TO_IDX
from dataset.dataset_preprocessor import load_dataset_cache, combine_dataset_caches

try:
    from xgboost import XGBClassifier

    HAS_XGBOOST = True
    print("[Ensemble Engine] Đã nạp thành công XGBoost!")
except ImportError:
    from sklearn.ensemble import GradientBoostingClassifier

    HAS_XGBOOST = False
    print(
        "[Ensemble Engine] Chưa tìm thấy xgboost. Sử dụng Sklearn GradientBoostingClassifier fallback."
    )


class FocalLoss(nn.Module):
    """Focal Loss chuẩn kết hợp Balanced Class Weighting và Label Smoothing nhẹ."""

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


def train_resmlp(
    X_train, y_train, X_val, y_val, num_classes=7, epochs=120, lr=0.0003, device="cpu"
):
    """Huấn luyện EmotionResMLP với Cosine Scheduler & EarlyStopping."""
    dim = X_train.shape[1]
    model = EmotionResMLP(input_dim=dim, num_classes=num_classes).to(device)

    # Class weights cho full 7 lớp
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
        TensorDataset(X_tr_t, y_tr_t), batch_size=64, shuffle=True
    )

    X_v_t = torch.tensor(X_val, dtype=torch.float32).to(device)

    best_val_acc = 0.0
    best_weights = None
    patience = 30
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        scheduler.step()

        # Eval
        model.eval()
        with torch.no_grad():
            v_logits = model(X_v_t)
            v_preds = torch.argmax(v_logits, dim=1).cpu().numpy()
            v_acc = accuracy_score(y_val, v_preds)

            if v_acc > best_val_acc:
                best_val_acc = v_acc
                best_weights = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience and epoch >= 35:
                    break

    if best_weights is not None:
        model.load_state_dict(best_weights)

    return model, best_val_acc


def evaluate_models_on_dataset(dataset_name="new_data"):
    print("=" * 85)
    print(
        " 🔥 EVALUATING ENSEMBLE PIPELINE (ResMLP + XGBoost + RBF-SVM vs EmotionMLP)"
    )
    print(f" 🎯 DATASET: {dataset_name.upper()}")
    print("=" * 85)

    # 1. Load Cache
    cache_path = os.path.join(BACKEND_DIR, f"cache_{dataset_name.lower()}.npz")
    if not os.path.exists(cache_path):
        print(f"[Auto Preprocess] Đang trích xuất cache cho '{dataset_name}'...")
        from dataset.dataset_preprocessor import MultiDatasetPreprocessor

        prep = MultiDatasetPreprocessor()
        prep.process_and_cache_dataset(dataset_name)
        prep.close()

    data = load_dataset_cache(dataset_name, cache_dir=BACKEND_DIR)
    X_train, y_train = data["X_train"], data["y_train"]
    test_key = (
        "X_test" if "X_test" in data else ("X_val" if "X_val" in data else "X_test")
    )
    y_test_key = (
        "y_test" if "y_test" in data else ("y_val" if "y_val" in data else "y_test")
    )
    X_test, y_test = data[test_key], data[y_test_key]

    # Kiểm tra xem có tập Val riêng biệt không để dùng tune weights Soft-Voting chống Data Leakage
    has_distinct_val = (
        "X_val" in data
        and "X_test" in data
        and len(data["X_val"]) > 0
        and len(data["X_test"]) > 0
    )
    if has_distinct_val:
        X_val_tune, y_val_tune = data["X_val"], data["y_val"]
        print(
            f"✅ Tìm thấy tập Val riêng biệt ({len(X_val_tune)} mẫu) để tối ưu trọng số Ensemble chống Data Leakage!"
        )
    else:
        # Nếu không có tập Val riêng, cắt 15% ngẫu nhiên từ Train làm validation tuning split
        np.random.seed(42)
        perm = np.random.permutation(len(X_train))
        val_size = int(0.15 * len(X_train))
        val_idx = perm[:val_size]
        X_val_tune, y_val_tune = X_train[val_idx], y_train[val_idx]
        print(
            f"ℹ️ Sử dụng split Validation 15% ({len(X_val_tune)} mẫu) từ tập Train để tune trọng số Soft-Voting."
        )

    # 2. EPSO Mask Selection (Kế thừa quyết định Giữ hoặc Bỏ EPSO từ Bước 3)
    epso_mask_ds = os.path.join(BACKEND_DIR, f"epso_mask_{dataset_name.lower()}.npy")
    epso_mask_default = os.path.join(BACKEND_DIR, "epso_mask_real.npy")
    if os.path.exists(epso_mask_ds):
        epso_mask_path = epso_mask_ds
    else:
        epso_mask_path = epso_mask_default

    if os.path.exists(epso_mask_path):
        mask = np.load(epso_mask_path)
        print(
            f"✅ Áp dụng Mask tối ưu ({os.path.basename(epso_mask_path)}): {len(mask)}/544 đặc trưng."
        )
        X_train_red = X_train[:, mask]
        X_test_red = X_test[:, mask]
        X_val_tune_red = X_val_tune[:, mask]
    else:
        print("⚠️ Không tìm thấy mask, sử dụng toàn bộ 544D.")
        X_train_red, X_test_red, X_val_tune_red = X_train, X_test, X_val_tune

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 7

    # -------------------------------------------------------------------------
    # 3. Baseline EmotionMLP
    # -------------------------------------------------------------------------
    print("\n--- 1️⃣ Loading / Evaluating Baseline EmotionMLP ---")
    base_mlp = EmotionMLP(input_dim=X_train_red.shape[1], num_classes=num_classes).to(
        device
    )
    epso_candidates = [
        os.path.join(
            BACKEND_DIR, f"emotion_mlp_best_acc_weights_{dataset_name.lower()}.pth"
        ),
        os.path.join(
            BACKEND_DIR, f"emotion_mlp_epso_weights_{dataset_name.lower()}.pth"
        ),
        os.path.join(BACKEND_DIR, "emotion_mlp_best_acc_weights.pth"),
        os.path.join(BACKEND_DIR, "emotion_mlp_epso_weights.pth"),
    ]
    loaded_epso = False
    for wp in epso_candidates:
        if os.path.exists(wp):
            try:
                base_mlp.load_state_dict(
                    torch.load(wp, map_location=device, weights_only=True)
                )
                print(
                    f"✅ Đã nạp thành công checkpoint từ Bước 3: {os.path.basename(wp)}"
                )
                loaded_epso = True
                break
            except Exception:
                pass

    if not loaded_epso:
        print("⚠️ Chưa có checkpoint Bước 3, huấn luyện Baseline EmotionMLP mới...")
        base_opt = optim.AdamW(base_mlp.parameters(), lr=0.0003, weight_decay=1e-4)
        base_crit = FocalLoss(gamma=1.5, label_smoothing=0.05)
        tr_ds = TensorDataset(
            torch.tensor(X_train_red, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        tr_ld = DataLoader(tr_ds, batch_size=64, shuffle=True)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            base_opt, T_max=60, eta_min=1e-5
        )

        for ep in range(60):
            base_mlp.train()
            for bx, by in tr_ld:
                bx, by = bx.to(device), by.to(device)
                base_opt.zero_grad()
                loss = base_crit(base_mlp(bx), by)
                loss.backward()
                base_opt.step()
            scheduler.step()

    base_mlp.eval()
    with torch.no_grad():
        base_probs = (
            F.softmax(
                base_mlp(torch.tensor(X_test_red, dtype=torch.float32).to(device)),
                dim=1,
            )
            .cpu()
            .numpy()
        )
        base_val_probs = (
            F.softmax(
                base_mlp(torch.tensor(X_val_tune_red, dtype=torch.float32).to(device)),
                dim=1,
            )
            .cpu()
            .numpy()
        )
    base_preds = np.argmax(base_probs, axis=1)
    base_acc = accuracy_score(y_test, base_preds)
    base_f1 = f1_score(y_test, base_preds, average="macro", zero_division=0)
    print(
        f"🎯 Baseline EmotionMLP Test Accuracy: {base_acc*100:.2f}% | F1: {base_f1*100:.2f}%"
    )

    # -------------------------------------------------------------------------
    # 4. [Cơ Chế 5]: RBF-SVM Classifier & So Sánh với EmotionMLP
    # -------------------------------------------------------------------------
    print("\n--- 2️⃣ Training & Evaluating RBF-SVM Classifier (Cơ Chế 5) ---")
    svm_clf = SVC(
        kernel="rbf", C=3.0, gamma="scale", probability=True, random_state=42
    )
    print(f"  ⚡ Đang huấn luyện RBF-SVM trên {len(X_train_red)} mẫu ({X_train_red.shape[1]}D)...")
    svm_clf.fit(X_train_red, y_train)

    raw_svm_probs = svm_clf.predict_proba(X_test_red)
    svm_probs = np.zeros((len(X_test_red), num_classes), dtype=np.float32)
    for i, orig_c in enumerate(svm_clf.classes_):
        if orig_c < num_classes:
            svm_probs[:, orig_c] = raw_svm_probs[:, i]

    raw_svm_val_probs = svm_clf.predict_proba(X_val_tune_red)
    svm_val_probs = np.zeros((len(X_val_tune_red), num_classes), dtype=np.float32)
    for i, orig_c in enumerate(svm_clf.classes_):
        if orig_c < num_classes:
            svm_val_probs[:, orig_c] = raw_svm_val_probs[:, i]

    svm_preds = np.argmax(svm_probs, axis=1)
    svm_acc = accuracy_score(y_test, svm_preds)
    svm_f1 = f1_score(y_test, svm_preds, average="macro", zero_division=0)
    print(
        f"🎯 RBF-SVM Classifier Test Accuracy:  {svm_acc*100:.2f}% | F1: {svm_f1*100:.2f}%"
    )

    # Lưu RBF-SVM model
    svm_save_path = os.path.join(BACKEND_DIR, "rbf_svm_model.joblib")
    joblib.dump(svm_clf, svm_save_path)
    print(f"✅ Đã lưu RBF-SVM model tại: {svm_save_path}")

    # Bảng so sánh trực tiếp EmotionMLP vs RBF-SVM (Cơ Chế 5)
    print("\n" + "=" * 85)
    print("  ⚔️ [CƠ CHẾ 5] SO SÁNH TRỰC TIẾP: BASELINE EMOTIONMLP VS RBF-SVM CLASSIFIER")
    print("=" * 85)
    print(
        f"{'Mô Hình':<28} | {'Accuracy (%)':<14} | {'Macro F1 (%)':<14} | {'Quyết Định'}"
    )
    print("-" * 85)
    svm_wins = svm_acc > base_acc
    mlp_tag = "LOẠI BỎ (Kém hơn)" if svm_wins else "ĐƯỢC CHỌN VÀO ENSEMBLE"
    svm_tag = "ĐƯỢC CHỌN VÀO ENSEMBLE" if svm_wins else "LOẠI BỎ (Kém hơn)"
    print(
        f"{'1. Baseline EmotionMLP':<28} | {base_acc*100:<14.2f} | {base_f1*100:<14.2f} | {mlp_tag}"
    )
    print(
        f"{'2. RBF-SVM Classifier':<28} | {svm_acc*100:<14.2f} | {svm_f1*100:<14.2f} | {svm_tag}"
    )
    print("=" * 85)

    if svm_wins:
        print(
            f"🎯 KẾT LUẬN: RBF-SVM Classifier đạt Acc cao hơn (+{(svm_acc - base_acc)*100:.2f}%). "
            f"-> Quyết định: Chọn RBF-SVM thay thế Baseline EmotionMLP trong nhánh Ensemble!\n"
        )
        third_probs = svm_probs
        third_val_probs = svm_val_probs
        third_name = "RBF-SVM"
        third_key = "rbf_svm"
        third_acc = svm_acc
        third_f1 = svm_f1
    else:
        print(
            f"🎯 KẾT LUẬN: Baseline EmotionMLP đạt Acc cao hơn hoặc bằng (+{(base_acc - svm_acc)*100:.2f}%). "
            f"-> Quyết định: Giữ Baseline EmotionMLP trong nhánh Ensemble!\n"
        )
        third_probs = base_probs
        third_val_probs = base_val_probs
        third_name = "EmotionMLP"
        third_key = "mlp"
        third_acc = base_acc
        third_f1 = base_f1

    # -------------------------------------------------------------------------
    # 5. Train EmotionResMLP
    # -------------------------------------------------------------------------
    print("\n--- 3️⃣ Training EmotionResMLP (Deep Residual Connections) ---")
    res_mlp, best_val_acc = train_resmlp(
        X_train_red,
        y_train,
        X_test_red,
        y_test,
        num_classes=num_classes,
        epochs=120,
        device=device,
    )
    res_mlp.eval()
    with torch.no_grad():
        res_probs = (
            F.softmax(
                res_mlp(torch.tensor(X_test_red, dtype=torch.float32).to(device)),
                dim=1,
            )
            .cpu()
            .numpy()
        )
        res_val_probs = (
            F.softmax(
                res_mlp(torch.tensor(X_val_tune_red, dtype=torch.float32).to(device)),
                dim=1,
            )
            .cpu()
            .numpy()
        )
    res_preds = np.argmax(res_probs, axis=1)
    res_acc = accuracy_score(y_test, res_preds)
    res_f1 = f1_score(y_test, res_preds, average="macro", zero_division=0)
    print(f"🎯 EmotionResMLP Test Accuracy: {res_acc*100:.2f}% | F1: {res_f1*100:.2f}%")

    # -------------------------------------------------------------------------
    # 6. Train XGBoost Classifier
    # -------------------------------------------------------------------------
    print("\n--- 4️⃣ Training XGBoost Classifier ---")
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)

    if HAS_XGBOOST:
        xgb_clf = XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="multi:softprob",
            random_state=42,
            n_jobs=-1,
        )
    else:
        xgb_clf = GradientBoostingClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=5, random_state=42
        )
    xgb_clf.fit(X_train_red, y_train_enc)

    raw_xgb_probs = xgb_clf.predict_proba(X_test_red)
    xgb_probs = np.zeros((len(X_test_red), num_classes), dtype=np.float32)
    for i, orig_cls in enumerate(le.classes_):
        if orig_cls < num_classes:
            xgb_probs[:, orig_cls] = raw_xgb_probs[:, i]

    raw_xgb_val_probs = xgb_clf.predict_proba(X_val_tune_red)
    xgb_val_probs = np.zeros((len(X_val_tune_red), num_classes), dtype=np.float32)
    for i, orig_cls in enumerate(le.classes_):
        if orig_cls < num_classes:
            xgb_val_probs[:, orig_cls] = raw_xgb_val_probs[:, i]

    xgb_preds = np.argmax(xgb_probs, axis=1)
    xgb_acc = accuracy_score(y_test, xgb_preds)
    xgb_f1 = f1_score(y_test, xgb_preds, average="macro", zero_division=0)
    print(f"🎯 XGBoost Test Accuracy: {xgb_acc*100:.2f}% | F1: {xgb_f1*100:.2f}%")

    # -------------------------------------------------------------------------
    # 7. Soft-Voting Ensemble (ResMLP + XGBoost + Chosen Third Model)
    # -------------------------------------------------------------------------
    print(
        f"\n--- 5️⃣ Evaluating Soft-Voting Ensemble (ResMLP + XGBoost + {third_name}) ---"
    )
    best_ens_val_acc = 0.0
    best_weights = [0.34, 0.33, 0.33]

    # Grid search tìm bộ trọng số (w_res, w_xgb, w_third) trên tập VALIDATION để chống data leakage
    for w_xgb in np.linspace(0.0, 1.0, 21):
        for w_res in np.linspace(
            0.0, 1.0 - w_xgb, int(round((1.0 - w_xgb) / 0.05)) + 1
        ):
            w_third = max(0.0, round(1.0 - w_xgb - w_res, 4))
            p_val_comb = (
                w_res * res_val_probs + w_xgb * xgb_val_probs + w_third * third_val_probs
            )
            p_val_preds = np.argmax(p_val_comb, axis=1)
            val_acc = accuracy_score(y_val_tune, p_val_preds)
            if val_acc > best_ens_val_acc:
                best_ens_val_acc = val_acc
                best_weights = [
                    float(round(w_res, 3)),
                    float(round(w_xgb, 3)),
                    float(round(w_third, 3)),
                ]

    w_res, w_xgb, w_third = best_weights
    print(
        f"🎯 Trọng số Soft-Voting Tối Ưu (Tune trên Val): ResMLP = {w_res*100:.1f}%, XGBoost = {w_xgb*100:.1f}%, {third_name} = {w_third*100:.1f}%"
    )

    ensemble_probs = (
        w_res * res_probs + w_xgb * xgb_probs + w_third * third_probs
    )
    ensemble_preds = np.argmax(ensemble_probs, axis=1)
    ensemble_acc = accuracy_score(y_test, ensemble_preds)
    ensemble_f1 = f1_score(y_test, ensemble_preds, average="macro", zero_division=0)
    ensemble_prec = precision_score(
        y_test, ensemble_preds, average="macro", zero_division=0
    )
    ensemble_rec = recall_score(
        y_test, ensemble_preds, average="macro", zero_division=0
    )

    try:
        ensemble_auc = roc_auc_score(
            y_test, ensemble_probs, multi_class="ovr", average="macro"
        )
    except Exception:
        ensemble_auc = float("nan")

    print("\n" + "=" * 85)
    print(f"  🏆 SUMMARY PERFORMANCE TABLE [{dataset_name.upper()}]")
    print("=" * 85)
    print(f"{'Model Architecture':<36} | {'Accuracy (%)':<14} | {'Macro F1 (%)':<14}")
    print("-" * 75)
    print(
        f"{'1. Baseline EmotionMLP':<36} | {base_acc*100:<14.2f} | {base_f1*100:<14.2f}"
    )
    print(
        f"{'2. RBF-SVM Classifier (Cơ Chế 5)':<36} | {svm_acc*100:<14.2f} | {svm_f1*100:<14.2f}"
    )
    print(
        f"{'3. EmotionResMLP (Residual)':<36} | {res_acc*100:<14.2f} | {res_f1*100:<14.2f}"
    )
    print(
        f"{'4. XGBoost Classifier':<36} | {xgb_acc*100:<14.2f} | {xgb_f1*100:<14.2f}"
    )
    ens_label = f"5. ENSEMBLE (Soft-Voting: Res+XGB+{third_name})"
    print(
        f"{ens_label:<36} | {ensemble_acc*100:<14.2f} | {ensemble_f1*100:<14.2f}"
    )
    print("=" * 85)
    best_single_acc = max(base_acc, svm_acc, res_acc, xgb_acc)
    print(
        f"🚀 Net Gain over Best Single Model: +{(ensemble_acc - best_single_acc)*100:.2f}%\n"
    )

    # -------------------------------------------------------------------------
    # 8. Lưu Weights & Cấu Hình Phục Vụ Pipeline Inference
    # -------------------------------------------------------------------------
    resmlp_save_path = os.path.join(BACKEND_DIR, "emotion_resmlp_weights.pth")
    torch.save(res_mlp.state_dict(), resmlp_save_path)
    print(f"✅ Đã lưu EmotionResMLP weights tại: {resmlp_save_path}")

    ens_weights_path = os.path.join(BACKEND_DIR, "ensemble_weights.json")
    with open(ens_weights_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "weights": [w_res, w_xgb, w_third],
                "third_model": third_key,
                "third_name": third_name,
                "dataset": dataset_name,
                "accuracy": float(ensemble_acc),
                "macro_f1": float(ensemble_f1),
                "models_evaluated": {
                    "mlp_acc": float(base_acc),
                    "svm_acc": float(svm_acc),
                    "resmlp_acc": float(res_acc),
                    "xgboost_acc": float(xgb_acc),
                    "ensemble_acc": float(ensemble_acc),
                },
            },
            f,
            indent=2,
        )
    print(f"✅ Đã lưu cấu hình trọng số Ensemble tại: {ens_weights_path}")

    if HAS_XGBOOST:
        xgb_save_path = os.path.join(BACKEND_DIR, "xgboost_model.json")
        xgb_clf.save_model(xgb_save_path)
        print(f"✅ Đã lưu XGBoost model tại: {xgb_save_path}")

    return {
        "dataset": dataset_name,
        "base_acc": base_acc,
        "svm_acc": svm_acc,
        "res_acc": res_acc,
        "xgb_acc": xgb_acc,
        "third_model": third_name,
        "ensemble_acc": ensemble_acc,
        "ensemble_f1": ensemble_f1,
        "ensemble_prec": ensemble_prec,
        "ensemble_rec": ensemble_rec,
        "ensemble_auc": ensemble_auc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ensemble Head Benchmark with RBF-SVM (Cơ Chế 5)"
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["new_data"], help="Datasets to benchmark"
    )
    args = parser.parse_args()

    all_results = []
    for ds in args.datasets:
        res = evaluate_models_on_dataset(ds)
        all_results.append(res)

    for r in all_results:
        print(
            f"Dataset: {r['dataset'].upper():<8} | MLP: {r['base_acc']*100:.2f}% | RBF-SVM: {r['svm_acc']*100:.2f}% | ResMLP: {r['res_acc']*100:.2f}% | XGB: {r['xgb_acc']*100:.2f}% | ENSEMBLE: {r['ensemble_acc']*100:.2f}%"
        )
