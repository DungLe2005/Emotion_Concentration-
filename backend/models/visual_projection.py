"""
Visual Projection Module for EduFocus AI.
Ánh xạ đặc trưng thị giác 512D từ CLIP sang không gian thích hợp cho vi biểu cảm khuôn mặt.
"""

import torch
import torch.nn as nn


class VisualProjection(nn.Module):
    """
    Tầng chiếu tuyến tính riêng biệt cho nhánh Visual 512D với Residual Connection,
    giúp ánh xạ đặc trưng thị giác từ CLIP sang không gian thích hợp cho vi biểu cảm khuôn mặt.
    """

    def __init__(self, in_dim: int = 512, out_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.ln_out = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln_out(x + self.proj(x))

