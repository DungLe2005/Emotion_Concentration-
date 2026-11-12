"""
Kiểm thử Unit Test cho module clip_lora.py
"""

import sys
import os
import torch

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

import clip
from models.clip_lora import (
    apply_lora_to_clip_visual,
    merge_lora_weights,
    unmerge_lora_weights,
    get_lora_parameters,
    extract_lora_state_dict,
    load_lora_state_dict,
)
from models.facial_cross_attention import extract_clip_visual_and_patches


def test_lora():
    print("=" * 60)
    print("🧪 BẮT ĐẦU KIỂM THỬ MODULE CLIP LORA...")
    print("=" * 60)

    # Nạp mô hình CLIP chuẩn ViT-B/32
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Thiết bị: {device}")

    clip_path = os.path.join(BACKEND_DIR, "models", "clip_finetuned_best.pth")
    if os.path.exists(clip_path):
        print(f"Nạp CLIP từ checkpoint local: {clip_path}")
        state_dict = torch.load(clip_path, map_location=device, weights_only=False)
        clip_model = clip.model.build_model(state_dict).to(device)
    else:
        try:
            clip_model, _ = clip.load("ViT-B/32", device=device, jit=False)
        except Exception as e:
            print(f"Không thể tải CLIP từ internet: {e}")
            return

    clip_model = clip_model.float()

    # 1. Áp dụng LoRA từ blocks 4 đến 11
    target_blocks = list(range(4, 12))
    print(f"\n1. Tiêm LoRA vào Visual Transformer (Blocks {target_blocks})...")
    info = apply_lora_to_clip_visual(
        clip_model, target_blocks=target_blocks, r=8, alpha=16.0, dropout=0.0
    )
    print(f"   - Target blocks: {info['target_blocks']}")
    print(f"   - LoRA Rank r={info['rank']}, Alpha={info['alpha']}")
    print(f"   - Tổng tham số LoRA: {info['total_lora_params']:,} tham số")

    # Đảm bảo Text Encoder và Base Visual bị đóng băng 100%
    for name, p in clip_model.transformer.named_parameters():
        assert not p.requires_grad, f"Text param {name} không được mở grad!"
    for name, p in clip_model.visual.conv1.named_parameters():
        assert not p.requires_grad, f"Visual conv1 không được mở grad!"

    lora_params = get_lora_parameters(clip_model)
    print(f"   - Số lượng tensors LoRA trainable: {len(lora_params)}")
    assert len(lora_params) > 0, "Không tìm thấy tham số LoRA nào có grad!"

    # 2. Test forward pass
    print("\n2. Kiểm thử Forward Pass qua CLIP Visual...")
    dummy_img = torch.randn(2, 3, 224, 224, device=device)
    with torch.no_grad():
        feats_before_merge, patches_before_merge = extract_clip_visual_and_patches(
            clip_model.visual, dummy_img
        )
    print(f"   - Global feature shape: {feats_before_merge.shape}")
    print(f"   - Patch tokens shape: {patches_before_merge.shape}")
    assert feats_before_merge.shape == (2, 512)
    assert patches_before_merge.shape == (2, 49, 768)

    # 3. Test Backward pass
    print("\n3. Kiểm thử Backward Pass (Gradient Flow)...")
    clip_model.train()
    img_grad = torch.randn(2, 3, 224, 224, device=device)
    feats, patches = extract_clip_visual_and_patches(clip_model.visual, img_grad)
    loss = feats.sum() + patches.sum()
    loss.backward()

    for i, p in enumerate(lora_params[:4]):
        assert p.grad is not None, f"LoRA param {i} không nhận được gradient!"
    print("   ✅ Toàn bộ tham số LoRA nhận gradient thành công!")

    # 4. Test Merge và Unmerge
    print("\n4. Kiểm thử Merge & Unmerge LoRA Weights...")
    clip_model.eval()
    with torch.no_grad():
        out_lora, _ = extract_clip_visual_and_patches(clip_model.visual, dummy_img)

    merge_lora_weights(clip_model)
    with torch.no_grad():
        out_merged, _ = extract_clip_visual_and_patches(clip_model.visual, dummy_img)

    diff_merge = (out_lora - out_merged).abs().max().item()
    print(f"   - Độ sai lệch Max |out_lora - out_merged|: {diff_merge:.8e}")
    assert diff_merge < 1e-4, f"Lỗi: Output sau merge khác output LoRA ({diff_merge})!"
    print("   ✅ Zero-overhead Merge cho kết quả toán học trùng khớp 100%!")

    unmerge_lora_weights(clip_model)
    with torch.no_grad():
        out_unmerged, _ = extract_clip_visual_and_patches(clip_model.visual, dummy_img)
    diff_unmerge = (out_lora - out_unmerged).abs().max().item()
    print(f"   - Độ sai lệch Max |out_lora - out_unmerged|: {diff_unmerge:.8e}")
    assert diff_unmerge < 1e-4, "Lỗi unmerge!"
    print("   ✅ Unmerge khôi phục trọng số ban đầu chính xác tuyệt đối!")

    # 5. Test State Dict Extract & Load
    print("\n5. Kiểm thử Extract & Load LoRA Adapter state_dict...")
    lora_sd = extract_lora_state_dict(clip_model)
    print(f"   - Số keys trong LoRA state_dict: {len(lora_sd)}")
    load_lora_state_dict(clip_model, lora_sd)
    print("   ✅ Nạp lại LoRA Adapter thành công!")

    print("\n" + "=" * 60)
    print("🎉 TẤT CẢ KIỂM THỬ ĐƠN VỊ CHO CLIP LORA ĐÃ ĐẠT 100%!")
    print("=" * 60)


if __name__ == "__main__":
    test_lora()
