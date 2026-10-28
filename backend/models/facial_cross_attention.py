"""
Facial Query Cross-Attention Module for Emotion Recognition.

Nâng cấp Hướng A (DETR-style Learnable Facial Keypoint Queries Cross-Attention):
- 32 Learnable Queries (tăng từ 16) bao phủ chi tiết toàn bộ các vùng cơ mặt.
- Stack 2 tầng Transformer Decoder:
  + Self-Attention giữa các Queries: Học quan hệ không gian/hình học tương quan giữa các vùng biểu cảm.
  + Cross-Attention vào CLIP Patch Tokens: Truy vấn thông tin biến dạng biểu cảm từ 49 patches của CLIP ViT-B/32.
  + Feed-Forward Network (FFN) với GELU activation.
- Kiến trúc Pre-LN (LayerNorm trước Attention/FFN) đảm bảo gradient flow cực kỳ ổn định, chống triệt để NaN overflow.
- Attentive Query Pooling: Học trọng số quan trọng động của từng query thay vì mean pooling đơn thuần.
- Chiếu ra 25D biểu diễn cấu trúc vi biểu cảm sâu (Deep Facial Deformation Representation).
- Tổng kích thước đặc trưng giữ nguyên chuẩn 544D (512D Visual + 7D Text + 25D Facial CA), tương thích 100% với EPSO và Ensemble.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def extract_clip_visual_and_patches(visual_encoder: nn.Module, images: torch.Tensor):
    """
    Trích xuất đồng thời Global Visual Feature (512D) và 49 Spatial Patch Tokens (768D)
    từ CLIP VisionTransformer (ViT-B/32) trong 1 forward pass duy nhất.

    Args:
        visual_encoder: clip_model.visual (clip.model.VisionTransformer)
        images: Tensor ảnh [B, 3, 224, 224]

    Returns:
        global_feature: Tensor [B, 512] đã qua ln_post và projection
        patch_tokens: Tensor [B, 49, 768] đặc trưng không gian của 49 patches đã qua ln_post
    """
    v = visual_encoder
    x = images.type(v.conv1.weight.dtype)

    # 1. Patch embedding qua Conv2d (stride 32x32)
    x = v.conv1(x)  # [B, width, 7, 7]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, width, 49]
    x = x.permute(0, 2, 1)  # [B, 49, width]

    # 2. Thêm CLS token
    cls_token = v.class_embedding.to(x.dtype) + torch.zeros(
        x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device
    )
    x = torch.cat([cls_token, x], dim=1)  # [B, 50, width]

    # 3. Positional embedding & ln_pre
    x = x + v.positional_embedding.to(x.dtype)
    x = v.ln_pre(x)

    # 4. ViT Transformer Blocks
    x = x.permute(1, 0, 2)  # [50, B, width]
    x = v.transformer(x)
    x = x.permute(1, 0, 2)  # [B, 50, width]

    # 5. Global CLS vector (512D)
    cls_out = v.ln_post(x[:, 0, :])
    if v.proj is not None:
        global_feature = cls_out @ v.proj
    else:
        global_feature = cls_out

    # 6. Spatial Patch Tokens (49 tokens, width=768)
    patch_tokens = v.ln_post(x[:, 1:, :])

    return global_feature.float(), patch_tokens.float()


class FacialDecoderLayer(nn.Module):
    """
    Một tầng Transformer Decoder hoàn chỉnh cho Facial Queries (Pre-LN):
    1. Multi-Head Self-Attention: Các facial queries tương tác với nhau để học
       quan hệ tương quan giữa các nhóm cơ mặt (khóe miệng, chân mày, gò má).
    2. Multi-Head Cross-Attention: Facial queries truy vấn vào 49 Patch Tokens
       để trích xuất thông tin biến dạng biểu cảm từ ảnh khuôn mặt.
    3. Feed-Forward Network: Chiếu phi tuyến 2 tầng với GELU và Dropout.
    """
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 8,
        dim_feedforward: int = 1536,
        dropout: float = 0.1,
    ):
        super().__init__()
        # Self-Attention giữa các queries
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_self = nn.LayerNorm(embed_dim)

        # Cross-Attention truy vấn vào patch tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_kv = nn.LayerNorm(embed_dim)

        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm_ffn = nn.LayerNorm(embed_dim)

    def forward(self, queries: torch.Tensor, patch_tokens: torch.Tensor) -> torch.Tensor:
        # 1. Pre-LN Self-Attention + Residual
        q_norm = self.norm_self(queries)
        sa_out, _ = self.self_attn(query=q_norm, key=q_norm, value=q_norm)
        queries = queries + sa_out

        # 2. Pre-LN Cross-Attention + Residual
        q_norm = self.norm_q(queries)
        kv_norm = self.norm_kv(patch_tokens)
        ca_out, _ = self.cross_attn(query=q_norm, key=kv_norm, value=kv_norm)
        queries = queries + ca_out

        # 3. Pre-LN FFN + Residual
        queries = queries + self.ffn(self.norm_ffn(queries))
        return queries


class FacialQueryCrossAttention(nn.Module):
    """
    Learnable Facial Keypoint Queries Cross-Attention Module (DETR-Style Decoder Stack).

    32 Learnable Queries truy vấn vào 49 Patch Tokens của CLIP ViT để học
    sự biến dạng của các vùng cơ mặt (chân mày, mắt, rãnh cười, môi, cằm)
    mà không cần bất kỳ điểm landmark thủ công nào từ MediaPipe.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_queries: int = 32,
        num_heads: int = 8,
        num_layers: int = 2,
        out_dim: int = 25,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.out_dim = out_dim

        # 32 Learnable Queries đại diện tiềm ẩn cho các vùng giải phẫu cơ mặt
        self.queries = nn.Parameter(torch.randn(1, num_queries, embed_dim) * 0.02)

        # 2 tầng Transformer Decoder Layers
        self.layers = nn.ModuleList([
            FacialDecoderLayer(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dim_feedforward=embed_dim * 2,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        self.norm_out = nn.LayerNorm(embed_dim)

        # Attentive Query Pooling: Học trọng số quan trọng của từng query
        self.pool_attn = nn.Linear(embed_dim, 1)

        # Chiếu xuống 25D để tương thích hoàn hảo chuẩn 544D multimodal
        self.proj_head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_tokens: Tensor [B, 49, 768] từ CLIP ViT

        Returns:
            facial_features: Tensor [B, out_dim] (mặc định 25D)
        """
        B = patch_tokens.shape[0]

        # 1. Expand Learnable Queries cho toàn bộ batch
        q = self.queries.expand(B, -1, -1)  # [B, 32, 768]

        # 2. Chuyển tiếp qua 2 tầng Decoder (Self-Attention + Cross-Attention + FFN)
        for layer in self.layers:
            q = layer(q, patch_tokens)

        q = self.norm_out(q)  # [B, 32, 768]

        # 3. Attentive Pooling qua 32 queries
        attn_weights = F.softmax(self.pool_attn(q), dim=1)  # [B, 32, 1]
        x_pooled = torch.sum(q * attn_weights, dim=1)        # [B, 768]

        # 4. Chiếu về 25D
        facial_features = self.proj_head(x_pooled)  # [B, 25]
        return facial_features
