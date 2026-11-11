"""
Legacy Wrapper: Sequential Unfreezing -> Tự động chuyển đổi sang LoRA Fine-Tuning.
Theo kế hoạch nâng cấp EduFocus AI, phương pháp Sequential Unfreezing đã được thay thế
toàn diện bằng LoRA (Low-Rank Adaptation) trên các Blocks 4..11 của CLIP ViT-B/32.
"""

import sys
import os

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Re-export VisualProjection để đảm bảo 100% tương thích ngược
from models.visual_projection import VisualProjection
from evaluation.train_lora import main as lora_main

if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("📢 [THÔNG BÁO] Phương pháp Sequential Unfreeze đã được nâng cấp sang LoRA!")
    print("👉 Toàn bộ quá trình huấn luyện tự động chuyển tiếp sang 'train_lora.py' (Blocks 4-11).")
    print("=" * 80 + "\n")
    lora_main()
