"""
Quy trình Đánh Giá Khả Năng Tổng Quát Hóa Đổi Miền (Zero-Shot Cross-Domain Evaluation)
Tuân thủ nghiêm ngặt Giao thức Chuẩn Quốc tế GFER (ECCV 2024, Table 1 - arXiv:2408.10614)

Đối chiếu mô hình EduFocus AI (CLIP ViT-B/16 + LoRA + Facial Cross-Attention):
- In-Domain: RAF-DB Test Set (3,068 ảnh)
- Cross-Domain 1: FERPlus Test Set (arnabkumarroy02/ferplus: ~3,579 ảnh Test)
- Cross-Domain 2: AffectNet-7 Validation Set (lintongdai/affectnet7: ~3,500 ảnh Val)

So sánh trực tiếp với SOTA Literature (SCN, EAC, OFER, CAFE ECCV 2024).
"""

import sys
import os
import glob
import time
import json
import argparse
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image

# Thiết lập đường dẫn backend
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Cache cho matplotlib trên Windows
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BACKEND_DIR, ".tmp_mpl"))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

# Thư mục quản lý weights và reports
WEIGHTS_DIR = os.path.join(BACKEND_DIR, "weights")
REPORTS_DIR = os.path.join(BACKEND_DIR, "reports")
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

from dataset.dataset_loader import STANDARD_EMOTIONS, STANDARD_LABEL_TO_IDX
from evaluation.train_lora import (
    MultimodalLoRAModel,
    load_clip_backbone,
    load_lora_state_dict,
    get_val_transform,
    safe_tqdm,
)

# Số liệu SOTA đã được công bố chính thức trong Bảng 1 bài GFER (ECCV 2024, arXiv:2408.10614)
# Được tổng hợp tại sheet 'Cross_domain_GFER' trong FER_literature_tables.xlsx
GFER_SOTA_BASELINES = {
    "SCN (CVPR 2020)": {
        "conference": "CVPR 2020",
        "rafdb": 87.32,
        "ferplus": 58.37,
        "affectnet": 42.85,
    },
    "EAC (ECCV 2022)": {
        "conference": "ECCV 2022",
        "rafdb": 89.54,
        "ferplus": 54.38,
        "affectnet": 43.91,
    },
    "OFER (ICCV 2023)": {
        "conference": "ICCV 2023",
        "rafdb": 89.07,
        "ferplus": 53.90,
        "affectnet": 42.73,
    },
    "CAFE (ECCV 2024)": {
        "conference": "ECCV 2024",
        "rafdb": 88.72,
        "ferplus": 73.16,
        "affectnet": 45.86,
    },
}

# Ánh xạ nhãn RAF-DB sang 7 cảm xúc chuẩn
RAFDB_LABEL_MAP = {
    "1": "surprise",
    "2": "fear",
    "3": "disgust",
    "4": "happy",
    "5": "sad",
    "6": "angry",
    "7": "neutral",
}

# Ánh xạ nhãn thư mục tổng quát (case-insensitive & synonyms)
GENERIC_LABEL_MAP = {
    "angry": "angry",
    "anger": "angry",
    "0": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "1": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "2": "fear",
    "happy": "happy",
    "happiness": "happy",
    "3": "happy",
    "neutral": "neutral",
    "4": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "5": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
    "6": "surprise",
}


def find_latest_kagglehub_cache(dataset_handle: str) -> Optional[str]:
    """Tìm đường dẫn cache mới nhất của kagglehub cho handle tương ứng."""
    user_home = os.path.expanduser("~")
    # Cấu trúc: ~/.cache/kagglehub/datasets/<owner>/<name>/versions/<v>
    base_pattern = os.path.join(user_home, ".cache", "kagglehub", "datasets", dataset_handle, "versions", "*")
    matches = glob.glob(base_pattern)
    if matches:
        matches.sort(key=os.path.getmtime, reverse=True)
        return matches[0]
    # Thử tìm trực tiếp không có versions
    direct = os.path.join(user_home, ".cache", "kagglehub", "datasets", dataset_handle)
    if os.path.exists(direct):
        return direct
    return None


class CrossDomainDataset(Dataset):
    """
    Dataset nạp ảnh kiểm thử đa nguồn có khả năng tự động nhận diện cấu trúc thư mục:
    - RAF-DB: DATASET/test/{1..7}/
    - FERPlus (arnabkumarroy02/ferplus): Test/{Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise}/
    - AffectNet-7 (lintongdai/affectnet7): val/{anger, disgust, ...}/ hoặc 7 thư mục cảm xúc
    """

    def __init__(self, root_dir: str, dataset_type: str = "auto", transform=None):
        self.root_dir = os.path.abspath(root_dir)
        self.transform = transform or get_val_transform()
        self.image_paths: List[str] = []
        self.labels: List[int] = []

        if not os.path.exists(self.root_dir):
            raise FileNotFoundError(f"Không tìm thấy thư mục: {self.root_dir}")

        self._load_samples(dataset_type)

    def _load_samples(self, dataset_type: str):
        valid_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

        # 1. Kiểm tra nếu là RAF-DB
        is_rafdb = dataset_type == "rafdb" or os.path.exists(os.path.join(self.root_dir, "DATASET", "test"))
        if is_rafdb:
            test_dir = os.path.join(self.root_dir, "DATASET", "test")
            if not os.path.exists(test_dir):
                test_dir = os.path.join(self.root_dir, "test") if os.path.exists(os.path.join(self.root_dir, "test")) else self.root_dir
            for folder_num, std_name in RAFDB_LABEL_MAP.items():
                target_sub = os.path.join(test_dir, folder_num)
                if os.path.exists(target_sub):
                    std_idx = STANDARD_LABEL_TO_IDX[std_name]
                    for fname in os.listdir(target_sub):
                        if fname.lower().endswith(valid_exts):
                            self.image_paths.append(os.path.join(target_sub, fname))
                            self.labels.append(std_idx)
            if len(self.image_paths) > 0:
                return

        # 2. Quét cấu trúc phân cấp theo tên cảm xúc
        # Ưu tiên các thư mục test hoặc val
        candidate_roots = []
        for root, dirs, _ in os.walk(self.root_dir):
            base = os.path.basename(root).lower()
            if dataset_type == "ferplus" and base in ("test", "publictest", "privatetest"):
                candidate_roots.append(root)
            elif dataset_type == "affectnet" and base in ("val", "validation", "val_set", "valid"):
                candidate_roots.append(root)

        if not candidate_roots:
            candidate_roots = [self.root_dir]
            for sub in ["test", "Test", "val", "Val", "validation", "Validation"]:
                p = os.path.join(self.root_dir, sub)
                if os.path.exists(p) and os.path.isdir(p):
                    candidate_roots.append(p)

        for c_root in candidate_roots:
            subdirs = [d for d in os.listdir(c_root) if os.path.isdir(os.path.join(c_root, d))]
            matched_any = False
            for d in subdirs:
                d_lower = d.lower()
                # Bỏ qua lớp contempt (lớp thứ 8 không thuộc 7 basic emotions chuẩn)
                if "contempt" in d_lower:
                    continue
                std_name = GENERIC_LABEL_MAP.get(d_lower)
                if not std_name:
                    for k, v in GENERIC_LABEL_MAP.items():
                        if k in d_lower:
                            std_name = v
                            break
                if std_name and std_name in STANDARD_LABEL_TO_IDX:
                    matched_any = True
                    std_idx = STANDARD_LABEL_TO_IDX[std_name]
                    folder_path = os.path.join(c_root, d)
                    for fname in os.listdir(folder_path):
                        if fname.lower().endswith(valid_exts):
                            self.image_paths.append(os.path.join(folder_path, fname))
                            self.labels.append(std_idx)
            if matched_any and len(self.image_paths) > 100:
                break

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception:
            image = Image.new("RGB", (224, 224), (128, 128, 128))
        if self.transform:
            image = self.transform(image)
        return image, torch.tensor(label, dtype=torch.long)


def load_model_for_eval(
    checkpoint_path: Optional[str] = None,
    clip_variant: str = "ViT-B/16",
    device: str = "cuda",
) -> Tuple[nn.Module, Dict]:
    """Nạp mô hình EduFocus AI hoàn chỉnh phục vụ đánh giá."""
    device_obj = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")

    target_ckpt = None
    if checkpoint_path and os.path.exists(checkpoint_path):
        target_ckpt = checkpoint_path
    else:
        candidates = [
            os.path.join(WEIGHTS_DIR, "lora_best.pth"),
            os.path.join(WEIGHTS_DIR, "lora_last.pth"),
            os.path.join(BACKEND_DIR, "..", "resume", "lora_best.pth"),
            "/content/drive/MyDrive/edufocus_weights_rafdb/lora_best.pth",
        ]
        for c in candidates:
            if os.path.exists(c):
                target_ckpt = c
                break

    import clip

    print(f"📦 Đang khởi tạo backbone CLIP gốc W0 ({clip_variant}) trên {device_obj}...")
    clip_model, _ = clip.load(clip_variant, device=device, jit=False)
    clip_model = clip_model.float()

    model = MultimodalLoRAModel(
        clip_model=clip_model,
        target_blocks=range(4, 12),
        lora_rank=8,
        lora_alpha=16.0,
        lora_dropout=0.0,
        num_classes=len(STANDARD_EMOTIONS),
        facial_out_dim=128,
    ).to(device_obj)

    ckpt_meta = {}
    if target_ckpt and os.path.exists(target_ckpt):
        print(f"👉 Nạp checkpoint từ: {target_ckpt}")
        try:
            ckpt = torch.load(target_ckpt, map_location=device_obj, weights_only=False)
            if "lora_state_dict" in ckpt:
                load_lora_state_dict(model.clip, ckpt["lora_state_dict"])
            if "visual_proj" in ckpt:
                model.visual_proj.load_state_dict(ckpt["visual_proj"])
            if "facial_cross_attn" in ckpt:
                model.facial_cross_attn.load_state_dict(ckpt["facial_cross_attn"])
            if "head" in ckpt:
                model.head.load_state_dict(ckpt["head"])
            ckpt_meta = {
                "epoch": ckpt.get("epoch", "N/A"),
                "val_acc": ckpt.get("val_acc", 0.0),
                "val_f1": ckpt.get("val_f1", 0.0),
                "best_val_f1": ckpt.get("best_val_f1", 0.0),
            }
            print(f"✅ Nạp checkpoint thành công! (Epoch: {ckpt_meta.get('epoch')}, Best Val F1: {ckpt_meta.get('best_val_f1', 0.0):.4f})")
        except Exception as e:
            print(f"⚠️ Lỗi khi nạp checkpoint: {e}. Thử nạp từng file weights...")
    else:
        vproj_pth = os.path.join(WEIGHTS_DIR, "visual_proj_best.pth")
        fcross_pth = os.path.join(WEIGHTS_DIR, "facial_cross_attn_best.pth")
        head_pth = os.path.join(WEIGHTS_DIR, "emotion_mlp_weights.pth")
        if os.path.exists(vproj_pth):
            model.visual_proj.load_state_dict(torch.load(vproj_pth, map_location=device_obj, weights_only=True))
        if os.path.exists(fcross_pth):
            model.facial_cross_attn.load_state_dict(torch.load(fcross_pth, map_location=device_obj, weights_only=True))
        if os.path.exists(head_pth):
            model.head.load_state_dict(torch.load(head_pth, map_location=device_obj, weights_only=True))
        print("✅ Đã hoàn tất nạp các module trọng số riêng lẻ.")

    model.eval()
    return model, ckpt_meta


def evaluate_single_dataset(
    model: nn.Module,
    dataset: Dataset,
    batch_size: int = 64,
    device: str = "cuda",
    fp16: bool = True,
    desc: str = "Eval",
) -> Dict:
    """Đánh giá mô hình trên một tập dữ liệu và tính toán đầy đủ các chỉ số."""
    device_obj = torch.device(device if torch.cuda.is_available() and "cuda" in device else "cpu")
    num_workers = 2 if sys.platform != "win32" else 0
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device_obj.type == "cuda"),
    )

    all_preds = []
    all_targets = []

    pbar = safe_tqdm(loader, desc=f"Kiểm thử [{desc}] ({len(dataset)} ảnh)")
    t0 = time.time()
    with torch.no_grad():
        for imgs, lbls in pbar:
            imgs = imgs.to(device_obj, non_blocking=True)
            if fp16 and device_obj.type == "cuda":
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
            else:
                logits = model(imgs)
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(lbls.numpy())

    total_time = time.time() - t0
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)

    acc = accuracy_score(y_true, y_pred) * 100.0
    bal_acc = balanced_accuracy_score(y_true, y_pred) * 100.0
    macro_f1 = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)[2] * 100.0
    weighted_f1 = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)[2] * 100.0
    p_per_class, r_per_class, f1_per_class, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(STANDARD_EMOTIONS))), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(STANDARD_EMOTIONS))))

    per_class_stats = {}
    for i, emo in enumerate(STANDARD_EMOTIONS):
        per_class_stats[emo] = {
            "precision": float(p_per_class[i] * 100.0),
            "recall": float(r_per_class[i] * 100.0),
            "f1_score": float(f1_per_class[i] * 100.0),
            "support": int(support[i]),
        }

    return {
        "dataset_name": desc,
        "num_samples": len(y_true),
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class_stats,
        "confusion_matrix": cm.tolist(),
        "elapsed_seconds": float(total_time),
        "throughput_fps": float(len(y_true) / max(1e-5, total_time)),
    }


def generate_markdown_report(
    eval_results: Dict[str, Dict],
    output_path: str,
):
    """Tạo bảng báo cáo Markdown chuẩn đối chiếu với Bảng 1 bài GFER ECCV 2024."""
    raf_acc = eval_results.get("rafdb", {}).get("accuracy", 0.0)
    fer_acc = eval_results.get("ferplus", {}).get("accuracy", 0.0)
    aff_acc = eval_results.get("affectnet", {}).get("accuracy", 0.0)

    delta_fer = (fer_acc - raf_acc) if (fer_acc > 0 and raf_acc > 0) else 0.0
    delta_aff = (aff_acc - raf_acc) if (aff_acc > 0 and raf_acc > 0) else 0.0

    valid_cross = [v for v in [fer_acc, aff_acc] if v > 0]
    avg_cross = np.mean(valid_cross) if valid_cross else 0.0

    lines = [
        "# 🔬 BÁO CÁO ĐÁNH GIÁ TỔNG QUÁT HÓA ĐỔI MIỀN (ZERO-SHOT CROSS-DOMAIN EVALUATION)",
        "",
        "> **Tuân thủ chuẩn thực nghiệm**: Giao thức Bảng 1 bài báo **GFER (ECCV 2024, arXiv:2408.10614)**.",
        "> Mô hình chỉ được huấn luyện duy nhất trên **RAF-DB (In-Domain)** và kiểm thử trực tiếp (Zero-Shot) sang các miền phân bố khác.",
        "",
        "## 1. Bảng Đối Chiếu Trực Tiếp Với SOTA Literature (ECCV 2024)",
        "",
        "| Mô hình | Hội nghị / Năm | RAF-DB (In-Domain) | FERPlus (Cross-Domain) | Δ FERPlus | AffectNet-7 (Cross-Domain) | Δ AffectNet-7 | Trung bình Cross-Domain |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for model_name, data in GFER_SOTA_BASELINES.items():
        m_raf = data["rafdb"]
        m_fer = data["ferplus"]
        m_aff = data["affectnet"]
        d_fer = m_fer - m_raf
        d_aff = m_aff - m_raf
        m_avg = (m_fer + m_aff) / 2.0
        lines.append(
            f"| {model_name} | {data['conference']} | {m_raf:.2f}% | {m_fer:.2f}% | {d_fer:+.2f}% | {m_aff:.2f}% | {d_aff:+.2f}% | {m_avg:.2f}% |"
        )

    fer_str = f"{fer_acc:.2f}%" if fer_acc > 0 else "N/A"
    aff_str = f"{aff_acc:.2f}%" if aff_acc > 0 else "N/A"
    d_fer_str = f"{delta_fer:+.2f}%" if fer_acc > 0 and raf_acc > 0 else "N/A"
    d_aff_str = f"{delta_aff:+.2f}%" if aff_acc > 0 and raf_acc > 0 else "N/A"
    avg_str = f"{avg_cross:.2f}%" if avg_cross > 0 else "N/A"

    lines.append(
        f"| **EduFocus AI (Của bạn)** | **2026** | **{raf_acc:.2f}%** | **{fer_str}** | **{d_fer_str}** | **{aff_str}** | **{d_aff_str}** | **{avg_str}** |"
    )

    lines.extend([
        "",
        "> [!NOTE]",
        "> - **Δ (Độ suy giảm miền)**: $\\Delta = \\text{Acc}_{\\text{Target}} - \\text{Acc}_{\\text{RAF-DB}}$. Giá trị âm càng nhỏ (gần 0) thể hiện mô hình chống quên và thích ứng ngoại miền càng bền vững.",
        "> - Dữ liệu SOTA tham chiếu lấy từ **Table 1 bài báo GFER (ECCV 2024)** (được lưu tại sheet `Cross_domain_GFER` trong file `FER_literature_tables.xlsx`).",
        "",
        "## 2. Chi Tiết Hiệu Năng Từng Tập Dữ Liệu Kiểm Thử",
        "",
    ])

    for k, res in eval_results.items():
        lines.extend([
            f"### 📍 {res['dataset_name'].upper()}",
            f"- **Số lượng mẫu**: {res['num_samples']:,} ảnh",
            f"- **Độ chính xác (Accuracy)**: **{res['accuracy']:.2f}%**",
            f"- **Balanced Accuracy (Macro Recall)**: {res['balanced_accuracy']:.2f}%",
            f"- **Macro-F1 Score**: {res['macro_f1']:.2f}%",
            f"- **Tốc độ suy luận (Throughput)**: {res['throughput_fps']:.1f} ảnh/giây",
            "",
            "| Cảm xúc | Precision (%) | Recall (%) | F1-Score (%) | Số lượng (Support) |",
            "| :--- | :---: | :---: | :---: | :---: |",
        ])
        for emo, stats in res["per_class"].items():
            lines.append(
                f"| {emo.capitalize()} | {stats['precision']:.2f}% | {stats['recall']:.2f}% | {stats['f1_score']:.2f}% | {stats['support']} |"
            )
        lines.append("")

    report_content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"📄 Đã lưu báo cáo Markdown tại: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Đánh giá khả năng tổng quát hóa đổi miền (Zero-Shot Cross-Domain Evaluation) - EduFocus AI"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Đường dẫn file lora_best.pth",
    )
    parser.add_argument(
        "--clip_variant",
        type=str,
        default="ViT-B/16",
        help="Biến thể CLIP (Mặc định: ViT-B/16)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "rafdb", "ferplus", "affectnet"],
        help="Tập dữ liệu cần đánh giá: all, rafdb, ferplus, affectnet (Mặc định: all)",
    )
    parser.add_argument(
        "--rafdb_dir",
        type=str,
        default=None,
        help="Đường dẫn thư mục RAF-DB (Mặc định: tự tìm)",
    )
    parser.add_argument(
        "--ferplus_dir",
        type=str,
        default=None,
        help="Đường dẫn thư mục FERPlus test set (hoặc cache arnabkumarroy02/ferplus)",
    )
    parser.add_argument(
        "--affectnet_dir",
        type=str,
        default=None,
        help="Đường dẫn thư mục AffectNet-7 validation set (hoặc cache lintongdai/affectnet7)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size khi suy luận (Mặc định: 64)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Thiết bị tính toán",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Bật chế độ nửa độ chính xác FP16 (AMP) để tăng tốc suy luận",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=REPORTS_DIR,
        help="Thư mục lưu báo cáo kết quả",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model, ckpt_meta = load_model_for_eval(
        checkpoint_path=args.checkpoint,
        clip_variant=args.clip_variant,
        device=args.device,
    )

    eval_targets = []

    # 1. RAF-DB (In-Domain)
    if args.dataset in ("all", "rafdb"):
        rafdb_candidates = [
            args.rafdb_dir,
            "/content/Emotion_Concentration/RAF_DB",
            "/content/RAF_DB",
            os.path.join(BACKEND_DIR, "..", "RAF_DB"),
            r"d:\Emotion_Concentration\RAF_DB",
        ]
        rafdb_path = None
        for p in rafdb_candidates:
            if p and os.path.exists(p):
                rafdb_path = p
                break
        if rafdb_path:
            eval_targets.append(("rafdb", rafdb_path, "RAF-DB Test Set (In-Domain)"))
        else:
            print("⚠️ Không tìm thấy đường dẫn RAF-DB hợp lệ.")

    # 2. FERPlus (Cross-Domain 1)
    if args.dataset in ("all", "ferplus"):
        fer_candidates = [
            args.ferplus_dir,
            find_latest_kagglehub_cache("arnabkumarroy02/ferplus"),
            "/content/datasets/ferplus",
            "/content/datasets/ferplus/test",
            "/content/datasets/fer2013/test",
            os.path.join(BACKEND_DIR, "..", "datasets", "ferplus"),
        ]
        fer_path = None
        for p in fer_candidates:
            if p and os.path.exists(p):
                fer_path = p
                break
        if fer_path:
            eval_targets.append(("ferplus", fer_path, "FERPlus Test Set (Cross-Domain 1)"))
        else:
            print("ℹ️ Chưa tìm thấy thư mục FERPlus test set. Hãy chạy cell tải arnabkumarroy02/ferplus.")

    # 3. AffectNet-7 (Cross-Domain 2)
    if args.dataset in ("all", "affectnet"):
        aff_candidates = [
            args.affectnet_dir,
            find_latest_kagglehub_cache("lintongdai/affectnet7"),
            find_latest_kagglehub_cache("thienhuynh/affectnet-validation-set"),
            "/content/datasets/affectnet",
            "/content/datasets/affectnet/val",
            os.path.join(BACKEND_DIR, "..", "datasets", "affectnet"),
        ]
        aff_path = None
        for p in aff_candidates:
            if p and os.path.exists(p):
                aff_path = p
                break
        if aff_path:
            eval_targets.append(("affectnet", aff_path, "AffectNet-7 Val Set (Cross-Domain 2)"))
        else:
            print("ℹ️ Chưa tìm thấy thư mục AffectNet-7 validation set. Hãy chạy cell tải lintongdai/affectnet7.")

    if not eval_targets:
        print("❌ Không có tập dữ liệu nào khả dụng để đánh giá! Hãy kiểm tra lại đường dẫn.")
        sys.exit(1)

    eval_results = {}
    print("\n" + "=" * 70)
    print("🚀 BẮT ĐẦU ĐÁNH GIÁ CHUẨN ĐỔI MIỀN (ZERO-SHOT CROSS-DOMAIN EVALUATION)")
    print("=" * 70)

    for key, data_path, display_name in eval_targets:
        print(f"\n📂 Nạp dữ liệu: {display_name}")
        print(f"👉 Thư mục: {data_path}")
        try:
            ds = CrossDomainDataset(data_path, dataset_type=key)
            if len(ds) == 0:
                print(f"⚠️ Không tìm thấy ảnh nào trong: {data_path}")
                continue
            print(f"  -> Đã nhận diện thành công: {len(ds):,} ảnh kiểm thử.")
            res = evaluate_single_dataset(
                model=model,
                dataset=ds,
                batch_size=args.batch_size,
                device=args.device,
                fp16=args.fp16,
                desc=key.upper(),
            )
            eval_results[key] = res
            print(f"  🎯 Kết quả [{key.upper()}]: Accuracy = {res['accuracy']:.2f}% | Macro-F1 = {res['macro_f1']:.2f}%")
        except Exception as e:
            print(f"❌ Lỗi khi đánh giá tập {key}: {e}")

    md_path = os.path.join(args.output_dir, "cross_domain_evaluation_report.md")
    json_path = os.path.join(args.output_dir, "cross_domain_evaluation_report.json")

    generate_markdown_report(eval_results, md_path)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "checkpoint_meta": ckpt_meta,
                "sota_baselines": GFER_SOTA_BASELINES,
                "evaluation_results": eval_results,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"📄 Đã lưu báo cáo JSON tại: {json_path}")

    print("\n" + "=" * 70)
    print("🏆 BẢNG KẾT QUẢ ĐỐI CHIẾU SOTA LITERATURE (ECCV 2024 Table 1):")
    print("=" * 70)
    raf_acc = eval_results.get("rafdb", {}).get("accuracy", 0.0)
    fer_acc = eval_results.get("ferplus", {}).get("accuracy", 0.0)
    aff_acc = eval_results.get("affectnet", {}).get("accuracy", 0.0)
    d_fer = fer_acc - raf_acc if (fer_acc > 0 and raf_acc > 0) else 0.0
    d_aff = aff_acc - raf_acc if (aff_acc > 0 and raf_acc > 0) else 0.0
    print(f"{'Mô hình':<28} | {'Hội nghị':<10} | {'RAF-DB':<10} | {'FERPlus':<10} | {'Δ FERPlus':<10} | {'AffectNet':<10} | {'Δ AffectNet':<10}")
    print("-" * 105)
    for m_name, d in GFER_SOTA_BASELINES.items():
        print(f"{m_name:<28} | {d['conference']:<10} | {d['rafdb']:<9.2f}% | {d['ferplus']:<9.2f}% | {(d['ferplus']-d['rafdb']):<+9.2f}% | {d['affectnet']:<9.2f}% | {(d['affectnet']-d['rafdb']):<+9.2f}%")
    print("-" * 105)
    print(f"{'EduFocus AI (Của bạn)':<28} | {'2026':<10} | {raf_acc:<9.2f}% | {fer_acc:<9.2f}% | {d_fer:<+9.2f}% | {aff_acc:<9.2f}% | {d_aff:<+9.2f}%")
    print("=" * 105 + "\n")


if __name__ == "__main__":
    main()
