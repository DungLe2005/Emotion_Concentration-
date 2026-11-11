"""
LoRA (Low-Rank Adaptation) Module for OpenAI CLIP Vision Transformer.
Hỗ trợ tiêm LoRA vào các khối Attention và MLP từ blocks 4 đến 11 của ViT-B/32,
bảo toàn 100% trọng số gốc W0, chống Catastrophic Forgetting,
đồng thời cung cấp cơ chế Merge/Unmerge trực tiếp tương thích 100% với hạ tầng CLIP.
"""

import math
import os
from typing import Dict, Iterable, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """
    Tầng Linear tích hợp LoRA:
    h = W0(x) + (Dropout(x) @ A^T) @ B^T * (alpha / r)
    """

    def __init__(
        self,
        original_linear: nn.Linear,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original_linear = original_linear
        # Đóng băng 100% trọng số gốc W0
        for p in self.original_linear.parameters():
            p.requires_grad = False

        in_dim = original_linear.in_features
        out_dim = original_linear.out_features
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        device = original_linear.weight.device
        dtype = original_linear.weight.dtype

        # Ma trận LoRA khởi tạo cùng device & dtype với layer gốc
        self.lora_A = nn.Parameter(torch.empty(r, in_dim, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r, device=device, dtype=dtype))

        # Khởi tạo chuẩn He/Kaiming cho A và zeros cho B
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.merged = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.original_linear(x)

        base = self.original_linear(x)
        # delta = (Dropout(x) @ A.T) @ B.T * scaling
        delta = (self.dropout(x) @ self.lora_A.T) @ self.lora_B.T * self.scaling
        return base + delta

    def merge(self):
        if not self.merged:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.original_linear.weight.data += delta.to(
                device=self.original_linear.weight.device,
                dtype=self.original_linear.weight.dtype,
            )
            self.merged = True

    def unmerge(self):
        if self.merged:
            delta = (self.lora_B @ self.lora_A) * self.scaling
            self.original_linear.weight.data -= delta.to(
                device=self.original_linear.weight.device,
                dtype=self.original_linear.weight.dtype,
            )
            self.merged = False


class LoRAMultiheadAttention(nn.Module):
    """
    Bọc MultiheadAttention của OpenAI CLIP với LoRA cho cả QKV projection và Out projection.
    Đảm bảo 100% tương thích với chữ ký gọi: attn(x, x, x, need_weights=False, attn_mask=...)
    """

    def __init__(
        self,
        original_mha: nn.MultiheadAttention,
        r: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.original_mha = original_mha
        # Đóng băng 100% trọng số gốc
        for p in self.original_mha.parameters():
            p.requires_grad = False

        self.embed_dim = original_mha.embed_dim
        self.num_heads = original_mha.num_heads
        self.r = r
        self.alpha = alpha
        self.scaling = alpha / r

        device = original_mha.in_proj_weight.device
        dtype = original_mha.in_proj_weight.dtype

        # LoRA cho in_proj (Q, K, V gộp kích thước 3 * embed_dim)
        self.lora_A_qkv = nn.Parameter(torch.empty(r, self.embed_dim, device=device, dtype=dtype))
        self.lora_B_qkv = nn.Parameter(torch.zeros(3 * self.embed_dim, r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A_qkv, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_qkv)

        # LoRA cho out_proj
        self.lora_A_out = nn.Parameter(torch.empty(r, self.embed_dim, device=device, dtype=dtype))
        self.lora_B_out = nn.Parameter(torch.zeros(self.embed_dim, r, device=device, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A_out, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B_out)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()
        self.merged = False

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ):
        mha = self.original_mha

        if self.merged:
            return mha(
                query,
                key,
                value,
                key_padding_mask=key_padding_mask,
                need_weights=need_weights,
                attn_mask=attn_mask,
            )

        # Tính toán ma trận trọng số hiệu dụng (Dynamic Effective Weights)
        delta_in = (self.lora_B_qkv @ self.lora_A_qkv) * self.scaling
        effective_in_weight = mha.in_proj_weight + delta_in.to(
            device=mha.in_proj_weight.device, dtype=mha.in_proj_weight.dtype
        )

        delta_out = (self.lora_B_out @ self.lora_A_out) * self.scaling
        effective_out_weight = mha.out_proj.weight + delta_out.to(
            device=mha.out_proj.weight.device, dtype=mha.out_proj.weight.dtype
        )

        return F.multi_head_attention_forward(
            query=query,
            key=key,
            value=value,
            embed_dim_to_check=self.embed_dim,
            num_heads=self.num_heads,
            in_proj_weight=effective_in_weight,
            in_proj_bias=mha.in_proj_bias,
            bias_k=mha.bias_k,
            bias_v=mha.bias_v,
            add_zero_attn=mha.add_zero_attn,
            dropout_p=mha.dropout if self.training else 0.0,
            out_proj_weight=effective_out_weight,
            out_proj_bias=mha.out_proj.bias,
            training=self.training,
            key_padding_mask=key_padding_mask,
            need_weights=need_weights,
            attn_mask=attn_mask,
        )

    def merge(self):
        if not self.merged:
            delta_in = (self.lora_B_qkv @ self.lora_A_qkv) * self.scaling
            self.original_mha.in_proj_weight.data += delta_in.to(
                device=self.original_mha.in_proj_weight.device,
                dtype=self.original_mha.in_proj_weight.dtype,
            )
            delta_out = (self.lora_B_out @ self.lora_A_out) * self.scaling
            self.original_mha.out_proj.weight.data += delta_out.to(
                device=self.original_mha.out_proj.weight.device,
                dtype=self.original_mha.out_proj.weight.dtype,
            )
            self.merged = True

    def unmerge(self):
        if self.merged:
            delta_in = (self.lora_B_qkv @ self.lora_A_qkv) * self.scaling
            self.original_mha.in_proj_weight.data -= delta_in.to(
                device=self.original_mha.in_proj_weight.device,
                dtype=self.original_mha.in_proj_weight.dtype,
            )
            delta_out = (self.lora_B_out @ self.lora_A_out) * self.scaling
            self.original_mha.out_proj.weight.data -= delta_out.to(
                device=self.original_mha.out_proj.weight.device,
                dtype=self.original_mha.out_proj.weight.dtype,
            )
            self.merged = False


def freeze_model_parameters(model: nn.Module):
    """Đóng băng toàn bộ tham số của mô hình."""
    for param in model.parameters():
        param.requires_grad = False


def apply_lora_to_clip_visual(
    clip_model: nn.Module,
    target_blocks: Iterable[int] = range(4, 12),
    r: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.05,
    enable_mlp_lora: bool = True,
) -> Dict[str, Union[int, List[int]]]:
    """
    Tiêm LoRA vào nhánh Visual Transformer của OpenAI CLIP:
    - Đóng băng 100% Backbone gốc (cả Visual và Text).
    - Thay thế Attention và MLP của các blocks chỉ định (Mặc định: blocks 4..11).

    Args:
        clip_model: OpenAI CLIP model
        target_blocks: Danh sách chỉ số blocks cần tiêm LoRA (Mặc định: 4..11)
        r: Hạng rank của LoRA (Mặc định: 8)
        alpha: Hệ số scale LoRA (Mặc định: 16.0)
        dropout: Tỷ lệ Dropout của LoRA (Mặc định: 0.05)
        enable_mlp_lora: Có tiêm LoRA vào cả MLP (c_fc, c_proj) hay không

    Returns:
        Dict thông tin cấu hình LoRA đã tiêm
    """
    # 1. Đóng băng 100% Text Encoder và Visual Encoder gốc
    freeze_model_parameters(clip_model)

    visual = clip_model.visual
    resblocks = visual.transformer.resblocks
    target_blocks = sorted(list(target_blocks))

    injected_blocks = []
    total_lora_params = 0

    for idx in target_blocks:
        if idx < 0 or idx >= len(resblocks):
            continue

        block = resblocks[idx]

        # Tiêm LoRA vào Attention nếu chưa tiêm
        if not isinstance(block.attn, LoRAMultiheadAttention):
            block.attn = LoRAMultiheadAttention(
                original_mha=block.attn, r=r, alpha=alpha, dropout=dropout
            )

        # Tiêm LoRA vào MLP (c_fc và c_proj) nếu enable
        if enable_mlp_lora:
            if hasattr(block.mlp, "c_fc") and not isinstance(
                block.mlp.c_fc, LoRALinear
            ):
                block.mlp.c_fc = LoRALinear(
                    original_linear=block.mlp.c_fc, r=r, alpha=alpha, dropout=dropout
                )
            if hasattr(block.mlp, "c_proj") and not isinstance(
                block.mlp.c_proj, LoRALinear
            ):
                block.mlp.c_proj = LoRALinear(
                    original_linear=block.mlp.c_proj,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                )

        injected_blocks.append(idx)

    # Đếm số lượng tham số LoRA
    for p in get_lora_parameters(clip_model):
        total_lora_params += p.numel()

    info = {
        "target_blocks": injected_blocks,
        "rank": r,
        "alpha": alpha,
        "total_lora_params": total_lora_params,
    }
    return info


def get_lora_parameters(clip_model: nn.Module) -> List[nn.Parameter]:
    """Lấy danh sách tất cả các tham số LoRA cần huấn luyện trong CLIP."""
    lora_params = []
    for module in clip_model.visual.modules():
        if isinstance(module, (LoRALinear, LoRAMultiheadAttention)):
            for name, param in module.named_parameters(recurse=False):
                if param.requires_grad:
                    lora_params.append(param)
    return lora_params


def merge_lora_weights(clip_model: nn.Module):
    """
    Merge vĩnh viễn trọng số LoRA vào trọng số gốc W0 của CLIP Visual Transformer.
    Sau khi merge, mô hình trở thành OpenAI CLIP chuẩn, không tốn thêm chi phí tính toán.
    """
    for module in clip_model.visual.modules():
        if isinstance(module, (LoRALinear, LoRAMultiheadAttention)):
            module.merge()


def unmerge_lora_weights(clip_model: nn.Module):
    """Khôi phục lại trọng số gốc W0 bằng cách trừ đi phần delta LoRA."""
    for module in clip_model.visual.modules():
        if isinstance(module, (LoRALinear, LoRAMultiheadAttention)):
            module.unmerge()


def get_clean_clip_state_dict(clip_model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Trích xuất state_dict chuẩn 100% của OpenAI CLIP (302 keys, không chứa key LoRA hay wrapper).
    Cần gọi merge_lora_weights(clip_model) trước khi gọi hàm này.
    """
    clean_sd = {}
    visual = clip_model.visual
    resblocks = visual.transformer.resblocks

    # 1. Sao chép các tham số Text Encoder và Visual không bị bọc
    for k, v in clip_model.state_dict().items():
        # Bỏ qua các key sinh ra từ LoRA wrapper
        if (
            "lora_A" in k
            or "lora_B" in k
            or "original_mha" in k
            or "original_linear" in k
        ):
            continue
        clean_sd[k] = v.clone()

    # 2. Bổ sung các ma trận trọng số chuẩn từ các khối đã tiêm LoRA (đã merge)
    for idx, block in enumerate(resblocks):
        prefix = f"visual.transformer.resblocks.{idx}"
        # Attention
        if isinstance(block.attn, LoRAMultiheadAttention):
            orig_attn = block.attn.original_mha
            clean_sd[f"{prefix}.attn.in_proj_weight"] = orig_attn.in_proj_weight.data.clone()
            clean_sd[f"{prefix}.attn.in_proj_bias"] = orig_attn.in_proj_bias.data.clone()
            clean_sd[f"{prefix}.attn.out_proj.weight"] = orig_attn.out_proj.weight.data.clone()
            clean_sd[f"{prefix}.attn.out_proj.bias"] = orig_attn.out_proj.bias.data.clone()
        # MLP
        if hasattr(block.mlp, "c_fc") and isinstance(block.mlp.c_fc, LoRALinear):
            orig_fc = block.mlp.c_fc.original_linear
            clean_sd[f"{prefix}.mlp.c_fc.weight"] = orig_fc.weight.data.clone()
            clean_sd[f"{prefix}.mlp.c_fc.bias"] = orig_fc.bias.data.clone()
        if hasattr(block.mlp, "c_proj") and isinstance(block.mlp.c_proj, LoRALinear):
            orig_proj = block.mlp.c_proj.original_linear
            clean_sd[f"{prefix}.mlp.c_proj.weight"] = orig_proj.weight.data.clone()
            clean_sd[f"{prefix}.mlp.c_proj.bias"] = orig_proj.bias.data.clone()

    return clean_sd


def extract_lora_state_dict(clip_model: nn.Module) -> Dict[str, torch.Tensor]:
    """Trích xuất state_dict chỉ chứa các trọng số LoRA (dung lượng siêu nhẹ ~2-5 MB)."""
    lora_state = {}
    for name, module in clip_model.visual.named_modules():
        if isinstance(module, LoRALinear):
            lora_state[f"{name}.lora_A"] = module.lora_A.data.cpu()
            lora_state[f"{name}.lora_B"] = module.lora_B.data.cpu()
        elif isinstance(module, LoRAMultiheadAttention):
            lora_state[f"{name}.lora_A_qkv"] = module.lora_A_qkv.data.cpu()
            lora_state[f"{name}.lora_B_qkv"] = module.lora_B_qkv.data.cpu()
            lora_state[f"{name}.lora_A_out"] = module.lora_A_out.data.cpu()
            lora_state[f"{name}.lora_B_out"] = module.lora_B_out.data.cpu()
    return lora_state


def load_lora_state_dict(clip_model: nn.Module, state_dict: Dict[str, torch.Tensor]):
    """Nạp trọng số LoRA vào mô hình đã được tiêm LoRA."""
    for name, module in clip_model.visual.named_modules():
        if isinstance(module, LoRALinear):
            k_a = f"{name}.lora_A"
            k_b = f"{name}.lora_B"
            if k_a in state_dict and k_b in state_dict:
                module.lora_A.data.copy_(state_dict[k_a].to(module.lora_A.device))
                module.lora_B.data.copy_(state_dict[k_b].to(module.lora_B.device))
        elif isinstance(module, LoRAMultiheadAttention):
            k_aq = f"{name}.lora_A_qkv"
            k_bq = f"{name}.lora_B_qkv"
            k_ao = f"{name}.lora_A_out"
            k_bo = f"{name}.lora_B_out"
            if (
                k_aq in state_dict
                and k_bq in state_dict
                and k_ao in state_dict
                and k_bo in state_dict
            ):
                module.lora_A_qkv.data.copy_(
                    state_dict[k_aq].to(module.lora_A_qkv.device)
                )
                module.lora_B_qkv.data.copy_(
                    state_dict[k_bq].to(module.lora_B_qkv.device)
                )
                module.lora_A_out.data.copy_(
                    state_dict[k_ao].to(module.lora_A_out.device)
                )
                module.lora_B_out.data.copy_(
                    state_dict[k_bo].to(module.lora_B_out.device)
                )
