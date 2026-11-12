import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader
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

# Thêm sys.path gốc backend
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from models.mlp import EmotionMLP
from models.resmlp import EmotionResMLP
from dataset.dataset_loader import DatasetRegistry, STANDARD_EMOTIONS, STANDARD_LABEL_TO_IDX
from dataset.dataset_preprocessor import load_dataset_cache


class MultiDatasetBenchmark:
    """
    Công cụ Đánh Giá Benchmark Đa Bộ Dữ Liệu (Multi-Dataset Benchmark Evaluator).
    Hỗ trợ đánh giá nội miền (In-domain) và đánh giá chéo (Cross-Dataset Evaluation).
    """

    def __init__(
        self,
        model_weights_path: str = None,
        input_dim: int = None,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Benchmark] Thiết bị kiểm thử: {self.device}")

        if model_weights_path is None:
            if os.path.exists(os.path.join(BACKEND_DIR, "emotion_mlp_weights.pth")):
                model_weights_path = "emotion_mlp_weights.pth"
            else:
                model_weights_path = "emotion_resmlp_weights.pth"

        full_weights_path = os.path.join(BACKEND_DIR, model_weights_path) if not os.path.isabs(model_weights_path) else model_weights_path
        if not os.path.exists(full_weights_path):
            raise FileNotFoundError(f"[Benchmark] Không tìm thấy weights mô hình tại: {full_weights_path}")

        self._weights_path = full_weights_path
        state_dict = torch.load(full_weights_path, map_location=self.device, weights_only=True)
        
        # Tự động nhận diện cấu trúc & số chiều đầu vào từ checkpoint (hỗ trợ cả 647D mới và 544D cũ)
        if "res1.fc1.weight" in state_dict:
            detected_dim = int(state_dict["input_layer.0.weight"].shape[1]) if "input_layer.0.weight" in state_dict else 647
            self.input_dim = input_dim or detected_dim
            self.model = EmotionResMLP(input_dim=self.input_dim, num_classes=len(STANDARD_EMOTIONS)).to(self.device)
            model_type = "EmotionResMLP"
        else:
            detected_dim = int(state_dict["fc1.weight"].shape[1]) if "fc1.weight" in state_dict else 647
            self.input_dim = input_dim or detected_dim
            self.model = EmotionMLP(input_dim=self.input_dim, num_classes=len(STANDARD_EMOTIONS)).to(self.device)
            model_type = "Baseline EmotionMLP"

        self.model.load_state_dict(state_dict)
        self.model.eval()
        print(f"[Benchmark] Đã nạp mô hình {model_type} ({self.input_dim} chiều)")
        print(f"  >> Weights: {os.path.basename(full_weights_path)}")

    @staticmethod
    def find_weights_for_dataset(dataset_name: str) -> str:
        """Tìm file trọng số riêng cho dataset, fallback sang trọng số chung."""
        candidates = [
            os.path.join(BACKEND_DIR, f"emotion_mlp_weights_{dataset_name.lower()}.pth"),
            os.path.join(BACKEND_DIR, f"emotion_resmlp_weights_{dataset_name.lower()}.pth"),
            os.path.join(BACKEND_DIR, "emotion_mlp_weights.pth"),
            os.path.join(BACKEND_DIR, "emotion_resmlp_weights.pth"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p
        return os.path.join(BACKEND_DIR, "emotion_mlp_weights.pth")

    def evaluate_dataset(self, dataset_name: str, cache_dir: str = None) -> dict:
        if cache_dir is None:
            cache_dir = BACKEND_DIR

        ds_data = load_dataset_cache(dataset_name, cache_dir=cache_dir)
        
        if "X_test" in ds_data and len(ds_data["X_test"]) > 0:
            X_test = ds_data["X_test"]
            y_test = ds_data["y_test"]
            split_used = "test"
        elif "X_val" in ds_data and len(ds_data["X_val"]) > 0:
            X_test = ds_data["X_val"]
            y_test = ds_data["y_val"]
            split_used = "validation"
        else:
            X_test = ds_data["X_train"]
            y_test = ds_data["y_train"]
            split_used = "train"

        X_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_tensor = torch.tensor(y_test, dtype=torch.long)
        loader = DataLoader(TensorDataset(X_tensor, y_tensor), batch_size=64, shuffle=False)

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for features, labels in loader:
                features = features.to(self.device)
                outputs = self.model(features)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_probs = np.array(all_probs)

        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, average="macro", zero_division=0)
        rec = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        try:
            roc_auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
        except Exception:
            roc_auc = float("nan")

        cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(STANDARD_EMOTIONS))))
        report_dict = classification_report(
            all_labels, all_preds, labels=list(range(len(STANDARD_EMOTIONS))), target_names=STANDARD_EMOTIONS, output_dict=True, zero_division=0
        )

        return {
            "dataset_name": dataset_name,
            "split_used": split_used,
            "sample_count": len(all_labels),
            "accuracy": float(acc),
            "precision_macro": float(prec),
            "recall_macro": float(rec),
            "f1_macro": float(f1),
            "roc_auc_macro": float(roc_auc) if not np.isnan(roc_auc) else None,
            "confusion_matrix": cm.tolist(),
            "classification_report": report_dict
        }

    def run_benchmark_suite(self, dataset_names: list = None, output_dir: str = None) -> pd.DataFrame:
        if dataset_names is None:
            dataset_names = DatasetRegistry.list_datasets()

        if output_dir is None:
            output_dir = os.path.join(BACKEND_DIR, "output", "benchmark_results")

        os.makedirs(output_dir, exist_ok=True)

        results = []
        raw_results = {}

        print("\n" + "=" * 70)
        print("     CHẠY BENCHMARK ĐÁNH GIÁ MÔ HÌNH TRÊN NHIỀU BỘ DỮ LIỆU")
        print("=" * 70)

        for ds_name in dataset_names:
            try:
                res = self.evaluate_dataset(ds_name)
                raw_results[ds_name] = res

                results.append({
                    "Dataset": ds_name.upper(),
                    "Split": res["split_used"],
                    "Samples": res["sample_count"],
                    "Accuracy (%)": round(res["accuracy"] * 100, 2),
                    "Precision (%)": round(res["precision_macro"] * 100, 2),
                    "Recall (%)": round(res["recall_macro"] * 100, 2),
                    "F1-Score (%)": round(res["f1_macro"] * 100, 2),
                    "ROC-AUC (%)": round(res["roc_auc_macro"] * 100, 2) if res["roc_auc_macro"] else "N/A"
                })

                cm_arr = np.array(res["confusion_matrix"])
                plt.figure(figsize=(9, 7))
                disp = ConfusionMatrixDisplay(confusion_matrix=cm_arr, display_labels=STANDARD_EMOTIONS)
                disp.plot(cmap=plt.cm.Blues, ax=plt.gca(), xticks_rotation=45)
                plt.title(f"Confusion Matrix Benchmark - {ds_name.upper()}")
                plt.tight_layout()
                cm_save_path = os.path.join(output_dir, f"confusion_matrix_{ds_name.lower()}.png")
                plt.savefig(cm_save_path, dpi=300)
                plt.close()

            except Exception as e:
                print(f"[Benchmark Lỗi] Bỏ qua dataset '{ds_name}': {e}")

        df_summary = pd.DataFrame(results)

        print("\n=== BẢNG KẾT QUẢ BENCHMARK SO SÁNH ĐA BỘ DỮ LIỆU ===")
        print(df_summary.to_string(index=False))

        summary_csv_path = os.path.join(output_dir, "benchmark_summary.csv")
        df_summary.to_csv(summary_csv_path, index=False)

        summary_json_path = os.path.join(output_dir, "benchmark_summary.json")
        with open(summary_json_path, "w", encoding="utf-8") as f:
            json.dump(raw_results, f, indent=4, ensure_ascii=False)

        report_txt_path = os.path.join(output_dir, "benchmark_report.txt")
        with open(report_txt_path, "w", encoding="utf-8") as f:
            f.write("=== BÁO CÁO BENCHMARK ĐÁNH GIÁ MÔ HÌNH ĐA BỘ DỮ LIỆU ===\n\n")
            f.write(df_summary.to_string(index=False))
            f.write("\n\n" + "=" * 50 + "\nCHI TIẾT TỪNG DATASET:\n" + "=" * 50 + "\n")
            for ds_name, res in raw_results.items():
                f.write(f"\n---> Dataset: {ds_name.upper()} ({res['sample_count']} mẫu, Split: {res['split_used']})\n")
                f.write(f"Accuracy: {res['accuracy']*100:.2f}%\n")
                f.write(f"F1 (Macro): {res['f1_macro']*100:.2f}%\n")
                f.write("Classification Report:\n")
                rep_df = pd.DataFrame(res["classification_report"]).transpose()
                f.write(rep_df.to_string() + "\n")

        print(f"\n[Thành Công] Đã xuất toàn bộ kết quả benchmark tại: {output_dir}")
        return df_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Dataset Benchmark Evaluator")
    parser.add_argument("--weights", type=str, default=None, help="File trọng số mô hình MLP (auto-detect nếu không chỉ định)")
    parser.add_argument("--datasets", nargs="+", default=None, help="Danh sách các datasets để đánh giá")
    args = parser.parse_args()

    # Nếu không chỉ định weights, tự động tìm theo dataset
    if args.datasets and len(args.datasets) == 1 and args.weights is None:
        ds = args.datasets[0]
        weights_path = MultiDatasetBenchmark.find_weights_for_dataset(ds)
        print(f"[Auto-detect] Dataset '{ds}' -> Weights: {os.path.basename(weights_path)}")
    else:
        weights_path = args.weights

    benchmark = MultiDatasetBenchmark(model_weights_path=weights_path, input_dim=None)
    benchmark.run_benchmark_suite(dataset_names=args.datasets)
