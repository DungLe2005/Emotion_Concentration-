import sys
import os
import time
import json
import argparse
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple

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

import cv2
from torchvision import transforms
from tqdm import tqdm


def safe_tqdm(iterable, **kwargs):
    """Wrapper an toàn cho tqdm tránh lỗi [WinError 6] The handle is invalid trên Windows PowerShell."""
    try:
        return tqdm(iterable, **kwargs, dynamic_ncols=False)
    except Exception:
        return iterable


def safe_torch_save(obj, fpath: str, max_retries: int = 10) -> bool:
    """
    Hàm lưu checkpoint an toàn tuyệt đối trên Windows chống lỗi Sharing Violation / File Lock:
    - Ghi trước ra file tạm .tmp với PID và timestamp ngẫu nhiên (chống Windows Defender xung đột).
    - Retry tối đa 10 lần với thời gian chờ tăng dần (tổng cộng ~15 giây).
    - Tuyệt đối không throw Exception làm crash toàn bộ tiến trình huấn luyện.
    """
    os.makedirs(os.path.dirname(os.path.abspath(fpath)), exist_ok=True)
    tmp_path = f"{fpath}.{os.getpid()}_{int(time.time() * 1000)}.tmp"
    try:
        torch.save(obj, tmp_path)
    except Exception as e:
        # Nếu không lưu được file tạm, thử lưu trực tiếp vào fpath
        try:
            torch.save(obj, fpath)
            return True
        except Exception as e2:
            print(f"  ⚠️ Không thể lưu {os.path.basename(fpath)}: {e2}")
            return False

    for retry in range(max_retries):
        try:
            if os.path.exists(fpath):
                try:
                    os.remove(fpath)
                except Exception:
                    pass
            os.replace(tmp_path, fpath)
            return True
        except Exception:
            time.sleep(0.5 + 0.3 * retry)

    # Nếu sau max_retries vẫn bị khóa, dọn file tạm và fallback ghi đè lần cuối
    if os.path.exists(tmp_path):
        try:
            os.remove(tmp_path)
        except Exception:
            pass
    try:
        torch.save(obj, fpath)
        return True
    except Exception as final_e:
        print(
            f"  ⚠️ [LƯU Ý] Chưa thể ghi đè {os.path.basename(fpath)} do Windows Defender đang quét file ({final_e}). Huấn luyện vẫn tiếp tục an toàn..."
        )
        return False


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
from features.landmark_features import LandmarkFeatureExtractor, NUM_LANDMARK_FEATURES
from models.resmlp import EmotionResMLP
from models.facial_cross_attention import (
    FacialQueryCrossAttention,
    extract_clip_visual_and_patches,
)

EMOTION_PROMPTS = [
    "a face with an angry expression, furrowed eyebrows and intense glare",  # 0: angry
    "a face showing disgust, wrinkled nose and raised upper lip",  # 1: disgust
    "a fearful frightened face with wide eyes and open mouth",  # 2: fear
    "a happy smiling cheerful face with open smile",  # 3: happy
    "a neutral calm relaxed face with a plain expression",  # 4: neutral
    "a sad sorrowful face with downturned lips and gloomy eyes",  # 5: sad
    "a shocked surprised face with raised eyebrows and gasping mouth",  # 6: surprise
]

# CLIP Chuẩn hóa hình ảnh (Normalizing constants)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def get_train_transform():
    """
    Data Augmentation chuyên biệt cho vi biểu cảm khuôn mặt (Facial Emotion Recognition):
    - RandomResizedCrop (scale 0.85-1.0): Mô phỏng các góc chụp cận mặt hoặc lệch tâm.
    - RandomHorizontalFlip: Biểu cảm khuôn mặt đa số có tính đối xứng tự nhiên.
    - RandomRotation (±10°): Mô phỏng độ nghiêng đầu tự nhiên khi học bài.
    - ColorJitter (độ sáng, độ tương phản nhẹ): Mô phỏng biến thiên ánh sáng phòng học/camera.
    - RandomErasing (scale 0.02-0.20, p=0.25): Che khuất ngẫu nhiên một phần khuôn mặt
      (mô phỏng tay che mặt, kính, tóc), ép mô hình học phân bổ toàn diện thay vì chỉ nhìn miệng.
    """
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
    """Transform tĩnh chuẩn của CLIP cho tập kiểm định (Validation) và kiểm thử (Test)."""
    return transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_MEAN, CLIP_STD),
        ]
    )


def prepare_landmarks_for_split(
    image_paths: List[str],
    split_name: str,
    dataset_name: str,
    cache_dir: str = BACKEND_DIR,
    is_dry_run: bool = False,
) -> np.ndarray:
    """
    [Hướng A: Learnable Facial Keypoint Queries Cross-Attention (DETR-Style)]
    MediaPipe được thay thế hoàn toàn bằng cơ chế Cross-Attention trực tiếp trên GPU.
    Không cần trích xuất hay tính toán mốc hình học CPU. Trả về None ngay lập tức.
    """
    print(
        f"  -> [Facial Cross-Attention {split_name.upper()}] Bỏ qua MediaPipe CPU. Toàn bộ đặc trưng cấu trúc cơ mặt được học trực tiếp trên GPU qua 32 Learnable Queries & 2-Layer Decoder."
    )
    return None


class FocalLoss(nn.Module):
    """
    Focal Loss đa lớp chuẩn (Lin et al.) kết hợp Label Smoothing và Class Weighting.
    Khắc phục triệt để mất cân bằng mẫu: dập tắt gradient của lớp chiếm đa số (Happy, Neutral),
    tập trung lực gradient vào các lớp thiểu số khó phân biệt (Fear, Disgust).
    """

    def __init__(self, weight=None, gamma=2.0, label_smoothing=0.05):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, label_smoothing=self.label_smoothing, reduction="none"
        )
        pt = torch.exp(-ce_loss.clamp(max=20.0))
        focal_weight = (1.0 - pt) ** self.gamma
        loss = focal_weight * ce_loss
        if self.weight is not None:
            alpha = self.weight[targets]
            loss = loss * alpha
        return loss.mean()


class MultimodalEmotionDataset(Dataset):
    """
    Dataset đa thức: Ảnh khuôn mặt + Mốc hình học MediaPipe (25D) + Nhãn cảm xúc.
    Hỗ trợ caching mốc hình học để tránh tính toán lại nhiều lần qua 100 epochs.
    """

    def __init__(
        self,
        image_paths: List[str],
        labels: List[int],
        preprocess_fn,
        landmark_cache: np.ndarray = None,
    ):
        self.image_paths = image_paths
        self.labels = labels
        self.preprocess = preprocess_fn
        self.landmark_cache = landmark_cache

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            pil_img = Image.open(img_path).convert("RGB")
            img_tensor = self.preprocess(pil_img)
        except Exception:
            img_tensor = torch.zeros(3, 224, 224, dtype=torch.float32)

        if self.landmark_cache is not None and idx < len(self.landmark_cache):
            lm_feat = torch.tensor(self.landmark_cache[idx], dtype=torch.float32)
        else:
            lm_feat = torch.zeros(NUM_LANDMARK_FEATURES, dtype=torch.float32)

        return img_tensor, lm_feat, torch.tensor(label, dtype=torch.long)


class VisualProjection(nn.Module):
    """
    Tầng chiếu tuyến tính riêng biệt cho nhánh Visual 512D với Residual Connection,
    giúp ánh xạ đặc trưng thị giác từ CLIP sang không gian thích hợp cho vi biểu cảm khuôn mặt.
    """

    def __init__(self, in_dim=512, out_dim=512, dropout=0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ln_out = nn.LayerNorm(out_dim)

    def forward(self, x):
        return self.ln_out(x + self.proj(x))


class MultimodalSequentialModel(nn.Module):
    """
    Mô hình tích hợp End-to-End với Learnable Facial Keypoint Queries Cross-Attention (Hướng A):
    1. CLIP Visual Encoder -> 512D Global Visual + 49 Spatial Patch Tokens (768D)
    2. Visual Projection -> 512D
    3. CLIP Text Prompts -> 7D Cosine Similarity (Cached & Frozen)
    4. Facial Query Cross-Attention -> 32 Queries x 2-Layer Decoder -> 25D (Deep Facial Deformation)
    5. Ghép nối thành 544D -> Classifier Head (ResMLP) -> 7 Logits
    """

    def __init__(self, clip_model, num_classes=7, head_dropout=0.3):
        super().__init__()
        self.clip = clip_model
        self.num_classes = num_classes

        # Tiền mã hóa ngữ nghĩa 7 lớp cảm xúc qua Prompt Ensembling (Centroid Embedding từ fer_descriptions.json)
        with torch.no_grad():
            dev = next(clip_model.parameters()).device
            t_norm = get_emotion_centroid_embeddings(clip_model, device=dev)
            self.register_buffer("cached_text_norm", t_norm)

        # Tầng chiếu tuyến tính độc lập cho nhánh Visual và Facial Query Cross-Attention
        self.visual_proj = VisualProjection(in_dim=512, out_dim=512, dropout=0.1)
        self.facial_cross_attn = FacialQueryCrossAttention(
            embed_dim=768,
            num_queries=32,
            num_heads=8,
            num_layers=2,
            out_dim=25,
            dropout=0.1,
        )

        # Classifier Head tiếp nhận vector 544D chuẩn: [512D Visual Proj + 7D Text + 25D Facial CA]
        total_dim = 512 + 7 + 25
        self.head = EmotionResMLP(input_dim=total_dim, num_classes=num_classes)

    def forward(
        self, images: torch.Tensor, landmarks: torch.Tensor = None
    ) -> torch.Tensor:
        # 1. Trích xuất đồng thời global visual (512D) và spatial patch tokens (49, 768D) trong 1 forward pass
        img_feats, patch_tokens = extract_clip_visual_and_patches(
            self.clip.visual, images
        )
        img_norm = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-7)

        # 2. Chiếu tuyến tính nhánh Visual qua adapter riêng (512D -> 512D)
        h_vis = self.visual_proj(img_feats)

        # 3. Tính tương đồng ngữ nghĩa với 7 Prompt Text (7D)
        prompt_sims = img_norm @ self.cached_text_norm.to(img_norm.device).T  # (B, 7)

        # 4. Facial Keypoint Queries Cross-Attention (25D)
        h_facial = self.facial_cross_attn(patch_tokens)  # (B, 25)

        # 5. Ghép nối thành vector 544D chuẩn: [512D h_vis + 7D prompt_sims + 25D h_facial]
        combined = torch.cat([h_vis, prompt_sims, h_facial], dim=1)

        # 6. Dự đoán qua Classifier Head (ResMLP)
        logits = self.head(combined)
        return logits


class SequentialUnfreezeTrainer:
    """
    Bộ điều phối rã băng tuần tự 3 giai đoạn tối ưu hóa lũy tiến (Curriculum Progressive Unfreezing).
    - Phase 0 (1..20% epochs): Warm-up Head + Projections + 32 Facial Queries (Backbone 100% Khóa).
    - Phase 1 (21%..50% epochs): Visual Top-2 Adaptation (Blocks 10-11, Text Khóa).
    - Phase 2 (51%..100% epochs): Deep Visual Joint Optimization (Blocks 8-11, LLRD, Text Khóa).
    Text Encoder được ĐÓNG BĂNG 100% xuyên suốt mọi phase để bảo toàn không gian nhúng văn bản.
    """

    def __init__(
        self,
        model: MultimodalSequentialModel,
        total_epochs: int = 100,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.total_epochs = total_epochs
        self.device = device
        self.current_phase = -1
        self.optimizer = None
        self.scheduler = None

    def get_phase_limits(self) -> Tuple[int, int]:
        """
        Tính toán ranh giới epoch cho 3 Phase tối ưu khoa học:
        - Phase 0: 8-10% epochs (tối đa 8 epochs) để Warm-up Head + Projections (Backbone khóa).
        - Phase 1: 35% tiếp theo để rã băng Visual Top-2 (Blocks 10-11, thích nghi biểu cảm).
        - Phase 2: ~55% còn lại để tối ưu hóa sâu Blocks 8-11 kết hợp LLRD.
        """
        p0_limit = min(8, max(4, int(0.30 * self.total_epochs)))
        p1_limit = max(p0_limit + 15, int(0.60 * self.total_epochs))
        return p0_limit, p1_limit

    def get_phase_info(self, epoch: int) -> Tuple[int, str]:
        p0_limit, p1_limit = self.get_phase_limits()

        if epoch <= p0_limit:
            return (
                0,
                f"Phase 0: Head Warm-up (Epoch 1..{p0_limit}) [Backbone 100% Khóa]",
            )
        elif epoch <= p1_limit:
            return (
                1,
                f"Phase 1: Visual Top-2 Adaptation (Epoch {p0_limit + 1}..{p1_limit}) [Blocks 10-11, Text Khóa]",
            )
        else:
            return (
                2,
                f"Phase 2: Deep Visual Joint Optimization (Epoch {p1_limit + 1}..{self.total_epochs}) [Blocks 8-11, LLRD, Text Khóa]",
            )

    def update_phase(self, epoch: int):
        new_phase, phase_desc = self.get_phase_info(epoch)
        if new_phase == self.current_phase and self.optimizer is not None:
            return

        self.current_phase = new_phase
        print(f"\n{'='*70}")
        print(f"🔄 CHUYỂN GIAI ĐOẠN [Epoch {epoch}/{self.total_epochs}]: {phase_desc}")
        print(f"{'='*70}")

        clip_m = self.model.clip
        head_m = self.model.head

        # -----------------------------------------------------------------
        # QUY TẮC CỐT LÕI: KHÓA 100% TEXT ENCODER Ở TẤT CẢ CÁC PHASE
        # Tránh hiện tượng 'Sụp đổ không gian nhúng Văn bản' do chỉ có 7 câu prompt
        # -----------------------------------------------------------------
        for p in clip_m.transformer.parameters():
            p.requires_grad = False
        clip_m.token_embedding.requires_grad = False
        clip_m.positional_embedding.requires_grad = False
        clip_m.ln_final.requires_grad = False
        if hasattr(clip_m, "text_projection") and clip_m.text_projection is not None:
            clip_m.text_projection.requires_grad = False

        # Luôn mở trainable cho các tầng chiếu tuyến tính (Visual & Facial Cross-Attention) và Head
        for p in self.model.visual_proj.parameters():
            p.requires_grad = True
        for p in self.model.facial_cross_attn.parameters():
            p.requires_grad = True
        for p in head_m.parameters():
            p.requires_grad = True

        p0_limit, p1_limit = self.get_phase_limits()

        if self.current_phase == 0:
            # Phase 0: Khóa toàn bộ CLIP Visual + Text, chỉ mở Head + Projections + Facial Cross-Attention
            for p in clip_m.parameters():
                p.requires_grad = False

            adapter_params = (
                list(head_m.parameters())
                + list(self.model.visual_proj.parameters())
                + list(self.model.facial_cross_attn.parameters())
            )
            self.optimizer = optim.AdamW(
                [
                    {
                        "params": adapter_params,
                        "lr": 1e-3,
                        "weight_decay": 1e-4,
                        "name": "adapter",
                    }
                ]
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=p0_limit, eta_min=1e-5
            )

        elif self.current_phase == 1:
            # Phase 1: Mở Visual Blocks 10, 11 + ln_post + proj (Bảo toàn 100% Momentum của Adapter)
            vis_top2 = []
            for name, param in clip_m.visual.named_parameters():
                if any(
                    name.startswith(f"transformer.resblocks.{b}.") for b in [10, 11]
                ) or name in ["proj", "ln_post.weight", "ln_post.bias"]:
                    param.requires_grad = True
                    vis_top2.append(param)

            # Cập nhật LR của Adapter hiện có mà KHÔNG xóa trạng thái momentum
            for g in self.optimizer.param_groups:
                if g.get("name") == "adapter":
                    g["lr"] = 2e-4

            # Thêm nhóm tham số Visual Top-2 vào optimizer hiện hành
            self.optimizer.add_param_group(
                {
                    "params": vis_top2,
                    "lr": 3e-5,
                    "weight_decay": 1e-2,
                    "name": "vis_top2",
                }
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=p1_limit - p0_limit, eta_min=2e-6
            )

        else:
            # Phase 2: Mở rộng Visual Blocks 8, 9 với Layer-wise LR Decay (LLRD) và duy trì LR tối ưu
            vis_mid = []
            for name, param in clip_m.visual.named_parameters():
                if any(name.startswith(f"transformer.resblocks.{b}.") for b in [8, 9]):
                    param.requires_grad = True
                    vis_mid.append(param)

            # Điều chỉnh LR: Duy trì mức học hiệu quả, không dìm quá sâu để tránh gradient chết
            for g in self.optimizer.param_groups:
                if g.get("name") == "adapter":
                    g["lr"] = 8e-5
                elif g.get("name") == "vis_top2":
                    g["lr"] = 1.5e-5

            # Thêm nhóm Blocks 8-9 với LR cân bằng và eta_min hợp lý (1e-6)
            self.optimizer.add_param_group(
                {"params": vis_mid, "lr": 5e-6, "weight_decay": 1e-2, "name": "vis_mid"}
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.total_epochs - p1_limit, eta_min=1e-6
            )

        # In thông tin kiểm tra số tham số trainable
        trainable_cnt = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        total_cnt = sum(p.numel() for p in self.model.parameters())
        print(
            f"  -> Tham số Trainable: {trainable_cnt:,} / {total_cnt:,} ({100.0 * trainable_cnt / total_cnt:.2f}%)"
        )
        print(
            f"  -> Số nhóm tham số trong Optimizer: {len(self.optimizer.param_groups)} (Bảo toàn 100% Momentum)"
        )


def run_training(args):
    device = torch.device(
        args.device if torch.cuda.is_available() and "cuda" in args.device else "cpu"
    )
    print(
        f"\n🚀 [EduFocus AI] Bắt đầu Huấn Luyện Rã Băng Tuần Tự trên thiết bị: {device}"
    )
    print(f"  - Dataset: {args.dataset}")
    print(f"  - Tổng Epochs: {args.epochs}")
    print(f"  - Batch Size: {args.batch_size}")
    print(f"  - Save Dir: {args.save_dir}")
    os.makedirs(args.save_dir, exist_ok=True)

    # 1. Đọc dữ liệu từ DatasetRegistry
    print(f"\n[1/5] Đang nạp dữ liệu từ DatasetRegistry: '{args.dataset}'...")
    reader = DatasetRegistry.get_reader(args.dataset)
    splits = reader.get_splits()

    if "train" not in splits or ("val" not in splits and "test" not in splits):
        raise ValueError(
            f"Dataset '{args.dataset}' không có đủ splits 'train' và 'val'/'test'!"
        )

    train_paths, train_labels = splits["train"]
    val_key = "val" if "val" in splits else "test"
    val_paths, val_labels = splits[val_key]

    full_train_labels = list(train_labels)

    if args.dry_run:
        print("[Dry-run Mode] Cắt giảm dữ liệu để test nhanh toàn bộ 4 phase...")
        train_paths, train_labels = train_paths[:64], train_labels[:64]
        val_paths, val_labels = val_paths[:32], val_labels[:32]

    print(f"  - Tập Train: {len(train_paths)} ảnh")
    print(f"  - Tập Val ({val_key}): {len(val_paths)} ảnh")

    # 2. Tự động chuẩn bị và đồng bộ Landmark Features (25D) cùng lúc với huấn luyện
    print(
        f"\n[2/5] Đang xử lý & đồng bộ dữ liệu MediaPipe Landmark (25D) cùng lúc với huấn luyện..."
    )
    train_lm_cache = prepare_landmarks_for_split(
        train_paths,
        split_name="train",
        dataset_name=args.dataset,
        cache_dir=BACKEND_DIR,
        is_dry_run=args.dry_run,
    )
    val_lm_cache = prepare_landmarks_for_split(
        val_paths,
        split_name=val_key,
        dataset_name=args.dataset,
        cache_dir=BACKEND_DIR,
        is_dry_run=args.dry_run,
    )

    # 3. Khởi tạo CLIP ViT Model & Data Augmentation Transforms
    print(
        f"\n[3/5] Đang nạp OpenAI CLIP ({args.clip_variant}) & thiết lập Data Augmentation..."
    )
    clip_model, _ = clip.load(args.clip_variant, device=device)
    clip_model = clip_model.float()

    model = MultimodalSequentialModel(
        clip_model, num_classes=len(STANDARD_EMOTIONS)
    ).to(device)

    # Dataloaders: Train với Data Augmentation, Val với Transform tĩnh
    train_transform = get_train_transform()
    val_transform = get_val_transform()

    train_ds = MultimodalEmotionDataset(
        train_paths, train_labels, train_transform, train_lm_cache
    )
    val_ds = MultimodalEmotionDataset(
        val_paths, val_labels, val_transform, val_lm_cache
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
    )

    # 4. Tính Class Weights & Khởi tạo Class-Weighted Cross-Entropy Loss
    unique_cls, counts = np.unique(full_train_labels, return_counts=True)
    weights = np.ones(len(STANDARD_EMOTIONS), dtype=np.float32)
    tot_samples = len(full_train_labels)
    for c, cnt in zip(unique_cls, counts):
        if c < len(STANDARD_EMOTIONS):
            weights[c] = tot_samples / (len(unique_cls) * cnt)
    class_weights_t = torch.tensor(np.sqrt(weights), dtype=torch.float32).to(device)

    print("\n[Trọng số lớp Class-Weighted Focal Loss (Tính từ toàn bộ tập Train)]:")
    for idx, (emo, w) in enumerate(
        zip(STANDARD_EMOTIONS, class_weights_t.cpu().numpy())
    ):
        print(f"  - Lớp {idx} ({emo:10s}): w = {w:.3f}")

    # KÍCH HOẠT FOCAL LOSS: gamma=2.0, class_weights, label_smoothing=0.05
    # Dập tắt gradient lớp chiếm đa số (Happy/Neutral), tập trung học các lớp khó (Fear/Disgust)
    criterion = FocalLoss(weight=class_weights_t, gamma=2.0, label_smoothing=0.05)
    print("  -> ✅ Đã kích hoạt FocalLoss(gamma=2.0, label_smoothing=0.05) thành công!")
    # AMP đã được tắt hoàn toàn (FP32 thuần) để tránh NaN overflow ở blocks Visual sâu

    trainer = SequentialUnfreezeTrainer(
        model, total_epochs=args.epochs, device=str(device)
    )

    best_val_f1 = 0.0
    best_val_acc = 0.0
    no_improve_epochs = 0
    patience = args.patience
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}
    nan_streak = 0  # Đếm số epoch NaN liên tiếp cho cơ chế rollback
    best_ckpt_path = os.path.join(args.save_dir, "sequential_unfreeze_best.pth")

    start_epoch = 1
    if getattr(args, "resume", False) and os.path.exists(best_ckpt_path):
        try:
            print(f"  🔄 [RESUME] Đang tải checkpoint tối ưu từ: {best_ckpt_path}...")
            ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"])
            start_epoch = ckpt.get("epoch", 0) + 1
            best_val_f1 = ckpt.get("best_val_f1", 0.0)
            best_val_acc = ckpt.get("best_val_acc", 0.0)
            history = ckpt.get("history", history)
            print(
                f"  ✅ Đã tải checkpoint thành công! Tiếp tục từ Epoch {start_epoch} (Kỷ lục Best Val Macro F1: {best_val_f1:.2f}%)"
            )
        except Exception as e:
            print(
                f"  ⚠️ Không thể nạp checkpoint resume ({e}), bắt đầu huấn luyện từ đầu."
            )

    print(
        f"\n[4/5] Bắt đầu Vòng lặp Huấn Luyện (Epoch {start_epoch}..{args.epochs})..."
    )

    for epoch in range(start_epoch, args.epochs + 1):
        trainer.update_phase(epoch)
        model.train()

        running_loss = 0.0
        correct_train = 0
        total_train = 0
        t0 = time.time()

        for batch_idx, (b_imgs, b_lms, b_labels) in enumerate(train_loader):
            b_imgs = b_imgs.to(device)
            b_lms = b_lms.to(device)
            b_labels = b_labels.to(device)

            trainer.optimizer.zero_grad()

            # FP32 thuần - không dùng autocast để tránh NaN overflow
            logits = model(b_imgs, b_lms)
            loss = criterion(logits, b_labels)

            # NaN Detection: Bỏ qua batch có loss NaN/Inf để ngăn lan truyền
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"  ⚠️ NaN/Inf detected at batch {batch_idx}, skipping...")
                trainer.optimizer.zero_grad()
                continue

            loss.backward()
            total_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], max_norm=0.5
            )
            if torch.isnan(total_norm) or torch.isinf(total_norm):
                print(
                    f"  ⚠️ NaN/Inf gradient norm ({total_norm}) detected at batch {batch_idx}, skipping step..."
                )
                trainer.optimizer.zero_grad()
                continue
            trainer.optimizer.step()

            running_loss += loss.item() * len(b_labels)
            preds = torch.argmax(logits, dim=1)
            correct_train += (preds == b_labels).sum().item()
            total_train += len(b_labels)

        if trainer.scheduler is not None:
            trainer.scheduler.step()

        train_loss = running_loss / max(1, total_train)
        train_acc = 100.0 * correct_train / max(1, total_train)
        epoch_time_sec = time.time() - t0

        # Đánh giá trên tập Validation
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for v_imgs, v_lms, v_labels in val_loader:
                v_imgs = v_imgs.to(device)
                v_lms = v_lms.to(device)
                v_labels = v_labels.to(device)

                v_logits = model(v_imgs, v_lms)
                v_l = criterion(v_logits, v_labels)

                val_loss += v_l.item() * len(v_labels)
                v_preds = torch.argmax(v_logits, dim=1).cpu().numpy()
                all_preds.extend(v_preds)
                all_targets.extend(v_labels.cpu().numpy())

        val_loss = val_loss / max(1, len(all_targets))
        val_acc = 100.0 * accuracy_score(all_targets, all_preds)
        val_f1 = 100.0 * f1_score(all_targets, all_preds, average="macro")

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)

        print(
            f"Epoch [{epoch:03d}/{args.epochs:03d}] (P{trainer.current_phase}) | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.2f}% F1: {val_f1:.2f}% | "
            f"Time: {epoch_time_sec:.1f}s"
        )

        # NaN Rollback: Nếu loss NaN hoặc toàn bộ batch bị skip → rollback
        epoch_is_nan = np.isnan(train_loss) or np.isnan(val_loss) or total_train == 0
        if epoch_is_nan:
            nan_streak += 1
            print(f"  ⚠️ Epoch {epoch} NaN streak: {nan_streak}/3")
            if nan_streak >= 3 and os.path.exists(best_ckpt_path):
                print(f"  🔄 NaN Rollback: Load lại best checkpoint + giảm LR 50%...")
                ckpt = torch.load(
                    best_ckpt_path, map_location=device, weights_only=False
                )
                model.load_state_dict(ckpt["model_state_dict"])
                for g in trainer.optimizer.param_groups:
                    g["lr"] *= 0.5
                    print(f"     - {g.get('name', 'unknown')}: LR → {g['lr']:.2e}")
                nan_streak = 0
            continue  # Bỏ qua checkpoint save cho epoch NaN
        else:
            nan_streak = 0  # Reset streak nếu epoch hợp lệ

        # Lưu checkpoint nếu đạt Best Val Macro F1
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_acc = val_acc
            no_improve_epochs = 0
            print(
                f"======[KỶ LỤC MỚI] Đạt Best Val Macro F1: {best_val_f1:.2f}% (Acc: {best_val_acc:.2f}%) ➔ Đang lưu Checkpoint======"
            )

            try:
                # 1. Lưu weights tổng hợp (huấn luyện / resume)
                ckpt_path = os.path.join(args.save_dir, "sequential_unfreeze_best.pth")
                safe_torch_save(
                    {
                        "epoch": epoch,
                        "phase": trainer.current_phase,
                        "model_state_dict": model.state_dict(),
                        "head_state_dict": model.head.state_dict(),
                        "best_val_f1": best_val_f1,
                        "best_val_acc": best_val_acc,
                        "history": history,
                    },
                    ckpt_path,
                )

                # 2. Lưu riêng weights của CLIP để tái sử dụng trong trích xuất đặc trưng
                clip_save_path = os.path.join(
                    BACKEND_DIR, "models", "clip_finetuned_best.pth"
                )
                os.makedirs(os.path.dirname(clip_save_path), exist_ok=True)
                safe_torch_save(model.clip.state_dict(), clip_save_path)

                # 3. Lưu Facial Cross-Attention, Visual Proj và ResMLP Head cho inference & pipeline
                safe_torch_save(
                    model.facial_cross_attn.state_dict(),
                    os.path.join(BACKEND_DIR, "models", "facial_cross_attn_best.pth"),
                )
                safe_torch_save(
                    model.visual_proj.state_dict(),
                    os.path.join(BACKEND_DIR, "models", "visual_proj_best.pth"),
                )
                safe_torch_save(
                    model.head.state_dict(),
                    os.path.join(BACKEND_DIR, "emotion_resmlp_weights.pth"),
                )
            except Exception as save_err:
                print(
                    f"Lưu checkpoint gặp lỗi nhẹ ({save_err}), tiếp tục huấn luyện..."
                )
        else:
            if trainer.current_phase >= 1:
                no_improve_epochs += 1
                if patience > 0 and no_improve_epochs >= patience:
                    print(
                        f"\n[EARLY STOPPING] Val Macro F1 không cải thiện trong {patience} epochs liên tiếp ({no_improve_epochs}/{patience})."
                    )
                    print(
                        f"   Dừng huấn luyện sớm tại Epoch {epoch}/{args.epochs} để chống quá khớp (Overfitting)."
                    )
                    print(
                        f"   Kỷ lục tối ưu đã lưu: Best Val Macro F1: {best_val_f1:.2f}% (Acc: {best_val_acc:.2f}%)"
                    )
                    break

    print(f"\n[5/5] Huấn Luyện Hoàn Tất!")
    print(f"  - Kỷ lục Best Val Macro F1: {best_val_f1:.2f}%")
    print(f"  - Kỷ lục Best Val Accuracy: {best_val_acc:.2f}%")
    print(
        f"  - Checkpoint lưu tại: {os.path.join(args.save_dir, 'sequential_unfreeze_best.pth')}"
    )
    print(
        f"  - CLIP Backbone lưu tại: {os.path.join(BACKEND_DIR, 'models', 'clip_finetuned_best.pth')}"
    )

    # Ghi log lịch sử
    log_file = os.path.join(args.save_dir, "training_history.json")
    with open(log_file, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"  - Lịch sử huấn luyện đã ghi vào: {log_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Huấn Luyện Rã Băng Tuần Tự (EduFocus AI)"
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
        default=15,
        help="Số epochs không cải thiện trước khi Early Stopping (Mặc định: 15, 0 để tắt)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size (Mặc định: 16 cho RTX 3050 Laptop)",
    )
    parser.add_argument(
        "--clip_variant",
        type=str,
        default="ViT-B/32",
        help="Biến thể CLIP: ViT-B/32 hoặc ViT-B/16",
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
        default=os.path.join(BACKEND_DIR, "reports", "sequential_unfreeze"),
        help="Thư mục lưu weights và reports",
    )
    parser.add_argument(
        "--dry_run", action="store_true", help="Chế độ test nhanh pipeline"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Tiếp tục huấn luyện từ checkpoint gần nhất nếu có",
    )
    args = parser.parse_args()

    run_training(args)


if __name__ == "__main__":
    main()
