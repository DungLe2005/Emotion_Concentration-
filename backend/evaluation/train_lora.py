"""
Huấn luyện CLIP Fine-tuning với LoRA (Low-Rank Adaptation) - EduFocus AI
Áp dụng LoRA vào các Transformer Blocks 4 -> 11 của CLIP ViT-B/32.
Đóng băng 100% Backbone gốc (W0) và Text Encoder.
Tối ưu hóa liên kết: LoRA (Visual) + VisualProjection (512D) + FacialQueryCrossAttention (25D) + EmotionResMLP Head.
Tự động merge LoRA weights vào clip_finetuned_best.pth đảm bảo tương thích 100% downstream pipeline.
"""

import sys
import os
import time
import json
import math
import argparse
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple, Optional, Iterable

# Cấu hình cache an toàn cho matplotlib trên Windows
os.environ.setdefault(
    "MPLCONFIGDIR",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".tmp_mpl"),
)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

import cv2
from torchvision import transforms
from tqdm import tqdm

# Thiết lập đường dẫn gốc backend
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import clip
from dataset.dataset_loader import (
    DatasetRegistry,
    STANDARD_EMOTIONS,
    STANDARD_LABEL_TO_IDX,
    get_emotion_centroid_embeddings,
)
from models.mlp import EmotionMLP
from models.facial_cross_attention import (
    FacialQueryCrossAttention,
    extract_clip_visual_and_patches,
)
from models.visual_projection import VisualProjection
from models.clip_lora import (
    apply_lora_to_clip_visual,
    merge_lora_weights,
    unmerge_lora_weights,
    get_lora_parameters,
    get_clean_clip_state_dict,
    extract_lora_state_dict,
    load_lora_state_dict,
)

# CLIP Chuẩn hóa hình ảnh (Normalizing constants)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def safe_tqdm(iterable, **kwargs):
    """Wrapper an toàn cho tqdm tránh lỗi [WinError 6] trên Windows PowerShell."""
    try:
        return tqdm(iterable, **kwargs, dynamic_ncols=False)
    except Exception:
        return iterable


def safe_torch_save(obj, fpath: str, max_retries: int = 10) -> bool:
    """Hàm lưu checkpoint an toàn chống lỗi File Lock trên Windows."""
    os.makedirs(os.path.dirname(os.path.abspath(fpath)), exist_ok=True)
    tmp_path = f"{fpath}.{os.getpid()}_{int(time.time() * 1000)}.tmp"
    try:
        torch.save(obj, tmp_path)
    except KeyboardInterrupt:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        raise
    except Exception as e:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        try:
            torch.save(obj, fpath)
            return True
        except Exception as e2:
            print(f"  ⚠️ Không thể lưu {os.path.basename(fpath)}: {e2}")
            return False

    for retry in range(max_retries):
        try:
            if os.path.exists(fpath) and sys.platform.startswith("win"):
                try:
                    os.remove(fpath)
                except Exception:
                    pass
            os.replace(tmp_path, fpath)
            return True
        except Exception:
            time.sleep(0.5 + 0.3 * retry)

    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    try:
        torch.save(obj, fpath)
        return True
    except Exception as final_e:
        print(f"  ⚠️ Chưa thể ghi đè {os.path.basename(fpath)} ({final_e})")
        return False


def get_train_transform():
    """Augmentation chuyên biệt cho vi biểu cảm khuôn mặt."""
    return transforms.Compose(
        [
            transforms.Resize(240, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.RandomResizedCrop(
                224,
                scale=(0.85, 1.0),
                ratio=(0.95, 1.05),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
            transforms.RandomErasing(
                p=0.25, scale=(0.02, 0.20), ratio=(0.3, 3.3), value=0
            ),
        ]
    )


def get_val_transform():
    """Transform tĩnh chuẩn của CLIP cho tập validation và test."""
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


class FocalLoss(nn.Module):
    """Focal Loss đa lớp kết hợp Class Weights và Label Smoothing."""

    def __init__(
        self,
        alpha: Optional[torch.Tensor] = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.05,
        reduction: str = "mean",
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = inputs.size(-1)
        with torch.no_grad():
            smooth_targets = torch.full_like(inputs, self.label_smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1.0 - self.label_smoothing)

        log_p = F.log_softmax(inputs, dim=-1)
        p = torch.exp(log_p)
        focal_weight = (1.0 - p) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(inputs.device)[targets].unsqueeze(1)
            focal_weight = focal_weight * alpha_t

        loss = -torch.sum(focal_weight * smooth_targets * log_p, dim=-1)
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        return loss


class MultimodalEmotionDataset(Dataset):
    """Dataset nạp ảnh khuôn mặt cho mô hình Multimodal CLIP LoRA."""

    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

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


class MultimodalLoRAModel(nn.Module):
    """
    Mô hình tích hợp End-to-End CLIP LoRA (Blocks 4 -> 11):
    1. CLIP Visual Transformer (W0 frozen, LoRA blocks 4..11 trainable) -> 512D Global + 49 Patch Tokens (768D)
    2. Visual Projection -> 512D (Trainable)
    3. CLIP Text Prompts -> 7D Cosine Similarity (Frozen & Cached)
    4. Facial Query Cross-Attention -> 32 Learnable Queries x 2 Decoders -> 25D (Trainable)
    5. Ghép nối thành 544D -> Classifier Head (EmotionResMLP) -> 7 Logits (Trainable)
    """

    def __init__(
        self,
        clip_model: nn.Module,
        target_blocks: Iterable[int] = range(4, 12),
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.05,
        num_classes: int = 7,
    ):
        super().__init__()
        self.clip = clip_model
        self.num_classes = num_classes

        # Tiêm LoRA vào nhánh Visual của CLIP
        self.lora_info = apply_lora_to_clip_visual(
            clip_model=self.clip,
            target_blocks=target_blocks,
            r=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
            enable_mlp_lora=True,
        )

        # Tiền mã hóa ngữ nghĩa 7 lớp cảm xúc qua Prompt Ensembling (Centroid Embedding)
        with torch.no_grad():
            dev = next(clip_model.parameters()).device
            t_norm = get_emotion_centroid_embeddings(clip_model, device=dev)
            self.register_buffer("cached_text_norm", t_norm)

        # Tầng chiếu tuyến tính riêng biệt cho nhánh Visual (512D -> 512D)
        self.visual_proj = VisualProjection(in_dim=512, out_dim=512, dropout=0.1)

        # Facial Query Cross-Attention (32 queries -> 25D)
        self.facial_cross_attn = FacialQueryCrossAttention(
            embed_dim=768,
            num_queries=32,
            num_heads=8,
            num_layers=2,
            out_dim=25,
            dropout=0.1,
        )

        # Classifier Head nhận vector 544D: [512D Visual + 7D Text + 25D Facial CA]
        total_dim = 512 + 7 + 25
        self.head = EmotionMLP(input_dim=total_dim, num_classes=num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # 1. Trích xuất đồng thời global visual (512D) và spatial patch tokens (49/196, 768D)
        img_feats, patch_tokens = extract_clip_visual_and_patches(
            self.clip.visual, images
        )
        img_norm = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-5)

        # 2. Chiếu tuyến tính nhánh Visual (512D)
        h_vis = self.visual_proj(img_feats)

        # 3. Tính tương đồng ngữ nghĩa với 7 Prompt Text (7D)
        prompt_sims = img_norm @ self.cached_text_norm.to(
            device=img_norm.device, dtype=img_norm.dtype
        ).T

        # 4. Facial Keypoint Queries Cross-Attention (25D)
        h_facial = self.facial_cross_attn(patch_tokens)

        # 5. Ghép nối thành vector 544D chuẩn
        combined = torch.cat([h_vis, prompt_sims, h_facial], dim=1)

        # 6. Dự đoán 7 logits qua Baseline EmotionMLP
        logits = self.head(combined)
        return logits


def make_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    eta_min_ratio: float = 0.02,
) -> optim.lr_scheduler.LambdaLR:
    """Tạo bộ lập lịch học kết hợp Warmup tuyến tính và Cosine Decay."""
    warmup_steps = max(1, warmup_steps)
    total_steps = max(warmup_steps + 1, total_steps)

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return max(0.05, float(current_step + 1) / float(warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        progress = min(1.0, max(0.0, progress))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_decay

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def load_clip_backbone(variant: str = "ViT-B/16", device: str = "cuda") -> nn.Module:
    """
    Nạp CLIP ViT an toàn:
    - Nếu có sẵn clip_finetuned_best.pth local khớp đúng kiến trúc: nạp từ state_dict.
    - Ngược lại: gọi clip.load (nạp từ cache hoặc tải về).
    """
    expected_patch_size = 16 if "16" in variant else 32
    local_pth = os.path.join(BACKEND_DIR, "models", "clip_finetuned_best.pth")
    if os.path.exists(local_pth):
        try:
            sd = torch.load(local_pth, map_location=device, weights_only=False)
            patch_w = sd.get("visual.conv1.weight", None)
            if patch_w is not None and patch_w.shape[-1] == expected_patch_size:
                print(f"  -> Nạp CLIP ViT ({variant}) từ checkpoint local có sẵn: {local_pth}")
                model = clip.model.build_model(sd).to(device)
                return model.float()
            else:
                curr_size = patch_w.shape[-1] if patch_w is not None else "Unknown"
                print(f"  ℹ️ Checkpoint local có patch size {curr_size}, khác với yêu cầu {expected_patch_size}. Chuyển sang clip.load('{variant}')...")
        except Exception as e:
            print(f"  ⚠️ Nạp checkpoint local thất bại ({e}), chuyển sang clip.load...")

    print(f"  -> Nạp CLIP ViT thông qua clip.load('{variant}')...")
    model, _ = clip.load(variant, device=device, jit=False)
    return model.float()


def run_training(args):
    print("=" * 80)
    print("🚀 BẮT ĐẦU HUẤN LUYỆN CLIP VỚI LoRA (LOW-RANK ADAPTATION) - EDUFOCUS AI")
    print("=" * 80)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"  - Thiết bị tính toán: {device}")
    print(f"  - Tập dữ liệu mục tiêu: {args.dataset}")
    print(f"  - LoRA Target Blocks: {args.lora_blocks}")
    print(f"  - LoRA Rank r: {args.lora_rank}, Alpha: {args.lora_alpha}")
    print(f"  - Tổng số Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - FP16 (AMP): {args.fp16}")

    # 1. Nạp Dataset
    reader = DatasetRegistry.get_reader(args.dataset)
    splits = reader.get_splits()
    if "train" not in splits or ("val" not in splits and "test" not in splits):
        raise ValueError(
            f"Dataset '{args.dataset}' không có đủ splits 'train' và 'val'/'test'!"
        )

    raw_train_paths, raw_train_labels = splits["train"]

    # Kiểm tra xem dataset có sẵn tập validation độc lập không
    if "val" in splits and len(splits["val"][0]) > 0:
        train_paths, train_labels = raw_train_paths, raw_train_labels
        val_paths, val_labels = splits["val"]
        test_paths, test_labels = splits.get("test", ([], []))
        print(f"  - Cơ chế Split: Sử dụng tập 'val' có sẵn của dataset.")
    else:
        # Nếu dataset chỉ có train và test (như RAF-DB, KMUFED):
        # Dùng Stratified Split để tách tập Validation sạch từ Train,
        # giữ tập TEST cách ly 100% để chống triệt để rò rỉ dữ liệu (Zero Data Leakage).
        test_paths, test_labels = splits.get("test", ([], []))
        train_paths, val_paths, train_labels, val_labels = train_test_split(
            raw_train_paths,
            raw_train_labels,
            test_size=args.val_ratio,
            stratify=raw_train_labels,
            random_state=42,
        )
        print(
            f"  - Cơ chế Split: Stratified Split {args.val_ratio * 100:.1f}% từ Train sang Validation (Chống rò rỉ 100%):"
        )
        print(
            f"    + Train sạch (Cập nhật Weights)       : {len(train_paths):,} ảnh"
        )
        print(
            f"    + Validation (Early Stopping & Best F1): {len(val_paths):,} ảnh"
        )
        print(
            f"    + Test Set (Cách ly 100% tuyệt đối)    : {len(test_paths):,} ảnh"
        )

    if args.dry_run:
        print("\n⚡ [DRY-RUN] Giới hạn dữ liệu để kiểm thử nhanh quy trình...")
        train_paths, train_labels = train_paths[:32], train_labels[:32]
        val_paths, val_labels = val_paths[:16], val_labels[:16]
        test_paths, test_labels = test_paths[:16], test_labels[:16]
        args.epochs = 2

    print(
        f"  - Tổng số mẫu nạp: Train={len(train_paths)} | Val={len(val_paths)} | Test={len(test_paths)}"
    )

    train_ds = MultimodalEmotionDataset(train_paths, train_labels, transform=get_train_transform())
    val_ds = MultimodalEmotionDataset(val_paths, val_labels, transform=get_val_transform())
    test_ds = MultimodalEmotionDataset(test_paths, test_labels, transform=get_val_transform()) if test_paths else None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True
    )
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        if test_ds
        else None
    )

    # 2. Phân tích blocks mục tiêu LoRA
    if args.lora_blocks == "4-11" or args.lora_blocks == "4..11":
        target_blocks = list(range(4, 12))
    elif args.lora_blocks == "top4":
        target_blocks = list(range(8, 12))
    elif args.lora_blocks == "top6":
        target_blocks = list(range(6, 12))
    elif args.lora_blocks == "all":
        target_blocks = list(range(0, 12))
    else:
        # Danh sách phân tách dấu phẩy (vd: "4,5,6,7,8,9,10,11")
        try:
            target_blocks = [int(b.strip()) for b in args.lora_blocks.split(",")]
        except Exception:
            target_blocks = list(range(4, 12))

    # 3. Khởi tạo mô hình
    print("\n[1/4] Khởi tạo OpenAI CLIP Backbone & Tiêm LoRA...")
    base_clip = load_clip_backbone(args.clip_variant, device=str(device))
    model = MultimodalLoRAModel(
        clip_model=base_clip,
        target_blocks=target_blocks,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_classes=len(STANDARD_EMOTIONS),
    ).to(device)

    print(f"  - Các Blocks đã tiêm LoRA: {model.lora_info['target_blocks']}")
    print(f"  - Tổng số tham số LoRA: {model.lora_info['total_lora_params']:,}")

    # Đếm số tham số trainable
    lora_params = get_lora_parameters(model.clip)
    adapter_params = (
        list(model.visual_proj.parameters())
        + list(model.facial_cross_attn.parameters())
        + list(model.head.parameters())
    )
    all_trainable_params = lora_params + adapter_params
    num_trainable_params = sum(p.numel() for p in all_trainable_params)
    total_params = sum(p.numel() for p in model.parameters())
    print(
        f"  - Tham số Trainable: {num_trainable_params:,} / {total_params:,} ({num_trainable_params / total_params * 100:.2f}%)"
    )

    # 4. Thiết lập Optimizer & Scheduler
    optimizer = optim.AdamW(
        [
            {
                "params": lora_params,
                "lr": args.lora_lr,
                "weight_decay": 1e-2,
                "name": "lora",
            },
            {
                "params": adapter_params,
                "lr": args.head_lr,
                "weight_decay": 1e-4,
                "name": "adapter",
            },
        ]
    )

    warmup_steps = max(1, args.warmup_epochs)
    scheduler = make_warmup_cosine_scheduler(
        optimizer=optimizer,
        total_steps=args.epochs,
        warmup_steps=warmup_steps,
        eta_min_ratio=0.02,
    )

    # 5. Class Weights & Loss
    class_counts = np.bincount(train_labels, minlength=len(STANDARD_EMOTIONS))
    total_samples = len(train_labels)
    class_weights = total_samples / (len(STANDARD_EMOTIONS) * np.maximum(class_counts, 1).astype(float))
    class_weights = torch.tensor(class_weights, dtype=torch.float, device=device)
    criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.05)

    scaler = torch.amp.GradScaler("cuda", init_scale=args.init_scale) if args.fp16 and device.type == "cuda" else None

    # Resume checkpoint nếu được yêu cầu
    start_epoch = 1
    best_val_f1 = 0.0
    if args.resume:
        resume_dir = args.resume_dir or os.path.join(BACKEND_DIR, "..", "resume")
        resume_pth = os.path.join(resume_dir, args.resume_file or "lora_best.pth")
        if os.path.exists(resume_pth):
            print(f"\n🔄 Tiến hành Resume từ Checkpoint: {resume_pth}")
            ckpt = torch.load(resume_pth, map_location=device, weights_only=False)
            if "lora_state_dict" in ckpt:
                load_lora_state_dict(model.clip, ckpt["lora_state_dict"])
            if "visual_proj" in ckpt:
                model.visual_proj.load_state_dict(ckpt["visual_proj"])
            if "facial_cross_attn" in ckpt:
                model.facial_cross_attn.load_state_dict(ckpt["facial_cross_attn"])
            if "head" in ckpt:
                model.head.load_state_dict(ckpt["head"])
            if "optimizer" in ckpt:
                optimizer.load_state_dict(ckpt["optimizer"])
            start_epoch = ckpt.get("epoch", 1) + 1
            best_val_f1 = ckpt.get("val_f1", 0.0)
            print(f"  ✅ Đã nạp thành công! Bắt đầu từ Epoch {start_epoch}, Best Val F1: {best_val_f1:.4f}")

    # 6. Vòng lặp huấn luyện
    print("\n[2/4] Bắt đầu quá trình Huấn Luyện LoRA...")
    patience_counter = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_train_loss = 0.0

        pbar = safe_tqdm(train_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Train]")
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
            optimizer.zero_grad()

            if scaler is not None:
                with torch.amp.autocast("cuda"):
                    logits = model(imgs)
                    loss = criterion(logits, lbls)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(imgs)
                loss = criterion(logits, lbls)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(all_trainable_params, max_norm=1.0)
                optimizer.step()

            total_train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = total_train_loss / max(1, len(train_loader))

        # Đánh giá trên Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, lbls in safe_tqdm(val_loader, desc=f"Epoch {epoch:02d}/{args.epochs:02d} [Val]"):
                imgs, lbls = imgs.to(device, non_blocking=True), lbls.to(device, non_blocking=True)
                if scaler is not None:
                    with torch.amp.autocast("cuda"):
                        logits = model(imgs)
                        loss = criterion(logits, lbls)
                else:
                    logits = model(imgs)
                    loss = criterion(logits, lbls)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=-1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(lbls.cpu().numpy())

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = f1_score(val_targets, val_preds, average="macro")
        elapsed = time.time() - t0

        curr_lora_lr = optimizer.param_groups[0]["lr"]
        curr_head_lr = optimizer.param_groups[1]["lr"]

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | "
            f"LR (LoRA/Head): {curr_lora_lr:.2e}/{curr_head_lr:.2e} | "
            f"Time: {elapsed:.1f}s"
        )

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        # Kiểm tra lưu checkpoint tốt nhất
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            print(f"  ⭐ [KỶ LỤC MỚI] Đạt Best Val F1: {best_val_f1:.4f}! Đang lưu toàn bộ checkpoints...")

            os.makedirs(args.save_dir, exist_ok=True)
            models_dir = os.path.join(BACKEND_DIR, "models")
            os.makedirs(models_dir, exist_ok=True)

            # 1. Lưu checkpoint nhẹ LoRA
            lora_sd = extract_lora_state_dict(model.clip)
            lora_ckpt = {
                "epoch": epoch,
                "val_f1": val_f1,
                "val_acc": val_acc,
                "lora_info": model.lora_info,
                "lora_state_dict": lora_sd,
                "visual_proj": model.visual_proj.state_dict(),
                "facial_cross_attn": model.facial_cross_attn.state_dict(),
                "head": model.head.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            safe_torch_save(lora_ckpt, os.path.join(args.save_dir, "lora_best.pth"))

            # 2. Lưu module con cho Downstream Pipeline (True End-to-End)
            safe_torch_save(model.visual_proj.state_dict(), os.path.join(models_dir, "visual_proj_best.pth"))
            safe_torch_save(model.facial_cross_attn.state_dict(), os.path.join(models_dir, "facial_cross_attn_best.pth"))
            safe_torch_save(model.head.state_dict(), os.path.join(BACKEND_DIR, "emotion_mlp_weights.pth"))
            safe_torch_save(model.head.state_dict(), os.path.join(BACKEND_DIR, f"emotion_mlp_weights_{args.dataset.lower()}.pth"))
            # Tương thích ngược: lưu thêm alias emotion_resmlp_weights.pth
            safe_torch_save(model.head.state_dict(), os.path.join(BACKEND_DIR, "emotion_resmlp_weights.pth"))
            print(f"  ✅ Đã lưu trọng số Baseline EmotionMLP (End-to-End) tại: {os.path.join(BACKEND_DIR, 'emotion_mlp_weights.pth')}")

            # 3. MERGE LORA WEIGHTS VÀO CLIP ĐỂ XUẤT CHUẨN OPENAI CLIP (302 keys)
            print("  🔄 Đang Merge LoRA weights vào bản sao CLIP ViT để xuất 'clip_finetuned_best.pth'...")
            merge_lora_weights(model.clip)
            clean_clip_sd = get_clean_clip_state_dict(model.clip)
            safe_torch_save(clean_clip_sd, os.path.join(models_dir, "clip_finetuned_best.pth"))
            unmerge_lora_weights(model.clip)
            print("  ✅ Đã xuất 'clip_finetuned_best.pth' chuẩn OpenAI CLIP (302 keys) tương thích 100% downstream!")
        else:
            patience_counter += 1
            if args.patience > 0 and patience_counter >= args.patience:
                print(f"\n🛑 Kích hoạt Early Stopping: Val F1 không cải thiện sau {args.patience} epochs.")
                break

    # 7. Đánh giá cuối cùng trên Test Set
    if test_loader:
        print("\n[3/4] Đánh giá mô hình tốt nhất trên Test Set...")
        # Nạp lại weights tốt nhất
        best_pth = os.path.join(args.save_dir, "lora_best.pth")
        if os.path.exists(best_pth):
            ckpt = torch.load(best_pth, map_location=device, weights_only=False)
            load_lora_state_dict(model.clip, ckpt["lora_state_dict"])
            model.visual_proj.load_state_dict(ckpt["visual_proj"])
            model.facial_cross_attn.load_state_dict(ckpt["facial_cross_attn"])
            model.head.load_state_dict(ckpt["head"])

        model.eval()
        test_preds, test_targets = [], []
        with torch.no_grad():
            for imgs, lbls in safe_tqdm(test_loader, desc="Testing"):
                imgs = imgs.to(device)
                logits = model(imgs)
                preds = torch.argmax(logits, dim=-1)
                test_preds.extend(preds.cpu().numpy())
                test_targets.extend(lbls.numpy())

        test_acc = accuracy_score(test_targets, test_preds)
        test_f1 = f1_score(test_targets, test_preds, average="macro")
        print("\n" + "=" * 60)
        print(f"📊 KẾT QUẢ ĐÁNH GIÁ TRÊN TẬP TEST ({args.dataset.upper()}):")
        print(f"  - Test Accuracy : {test_acc * 100:.2f}%")
        print(f"  - Test Macro-F1 : {test_f1 * 100:.2f}%")
        print("=" * 60)
        print("\nBáo cáo chi tiết theo từng lớp cảm xúc:")
        print(
            classification_report(
                test_targets,
                test_preds,
                labels=list(range(len(STANDARD_EMOTIONS))),
                target_names=STANDARD_EMOTIONS,
                digits=4,
                zero_division=0,
            )
        )

    # 8. Lưu lịch sử
    print("\n[4/4] Lưu lịch sử huấn luyện...")
    hist_path = os.path.join(args.save_dir, "training_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"  ✅ Lịch sử huấn luyện đã ghi vào: {hist_path}")
    print("\n🎉 HUẤN LUYỆN LoRA HOÀN TẤT THÀNH CÔNG!")


def main():
    parser = argparse.ArgumentParser(
        description="Huấn Luyện CLIP Fine-Tuning với LoRA (EduFocus AI)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="new_data",
        help="Tên dataset: new_data, rafdb, kmufed",
    )
    parser.add_argument(
        "--epochs", type=int, default=70, help="Tổng số epoch tối đa (Mặc định: 70)"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Số epochs không cải thiện trước khi Early Stopping (Mặc định: 20)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (Mặc định: 16)",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.10,
        help="Tỷ lệ tách tập Validation từ Train khi dataset chỉ có Train/Test (Mặc định: 0.10 tức 10%%)",
    )
    parser.add_argument(
        "--clip_variant",
        type=str,
        default="ViT-B/16",
        help="Biến thể CLIP: ViT-B/16 (Khuyến nghị, 196 tokens) hoặc ViT-B/32 (49 tokens)",
    )
    parser.add_argument(
        "--lora_blocks",
        type=str,
        default="4-11",
        help="Khối blocks tiêm LoRA: 4-11, top4, top6, all hoặc danh sách phân tách dấu phẩy (Mặc định: 4-11)",
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help="Hạng LoRA Rank r (Mặc định: 8)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=float,
        default=16.0,
        help="Hệ số scale LoRA alpha (Mặc định: 16.0)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.05,
        help="Dropout cho LoRA (Mặc định: 0.05)",
    )
    parser.add_argument(
        "--lora_lr",
        type=float,
        default=2e-4,
        help="Learning rate cho các tham số LoRA (Mặc định: 2e-4)",
    )
    parser.add_argument(
        "--head_lr",
        type=float,
        default=1e-3,
        help="Learning rate cho Projections + Facial Cross-Attention + Head (Mặc định: 1e-3)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="Thiết bị: cuda:0 hoặc cpu"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Số worker nạp dữ liệu (Mặc định: 0 cho Windows)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=os.path.join(BACKEND_DIR, "reports", "lora"),
        help="Thư mục lưu weights và reports",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Chế độ test nhanh pipeline"
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Kích hoạt chế độ huấn luyện nửa độ chính xác FP16 (AMP)",
    )
    parser.add_argument(
        "--init_scale",
        type=float,
        default=1024.0,
        help="Hệ số scale ban đầu cho GradScaler FP16 (Mặc định: 1024.0)",
    )
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=2,
        help="Số epoch Warmup học tăng dần ở đầu (Mặc định: 2)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Bật chế độ tiếp tục huấn luyện (resume)",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="Đường dẫn thư mục chứa checkpoint resume",
    )
    parser.add_argument(
        "--resume_file",
        type=str,
        default=None,
        help="Tên file checkpoint cụ thể trong folder resume",
    )
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
