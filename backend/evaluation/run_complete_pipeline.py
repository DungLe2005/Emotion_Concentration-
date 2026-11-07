import sys
import os
import time
import argparse
import numpy as np

# Thiết lập đường dẫn backend
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Cấu hình cache an toàn cho matplotlib trên Windows
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BACKEND_DIR, ".tmp_mpl"))

import torch

from dataset.dataset_loader import DatasetRegistry
from dataset.dataset_preprocessor import MultiDatasetPreprocessor
import subprocess


def run_step(step_name: str, cmd_list: list):
    try:
        print("\n" + "=" * 80)
        print(f"🚀 BẮT ĐẦU: {step_name}")
        print(f"👉 Lệnh: {' '.join(cmd_list)}")
        print("=" * 80)
        t0 = time.time()
        env = os.environ.copy()
        env["PYTHONPATH"] = BACKEND_DIR + os.pathsep + env.get("PYTHONPATH", "")
        res = subprocess.run(cmd_list, cwd=BACKEND_DIR, env=env)
        elapsed = time.time() - t0
        if res.returncode != 0:
            print(f"\n❌ [LỖI] Bước '{step_name}' thất bại với mã lỗi {res.returncode}")
            sys.exit(res.returncode)
        print(f"\n✅ [HOÀN TẤT] {step_name} (Thời gian thực thi: {elapsed:.2f}s)")
    except Exception as e:
        print(f"\n❌ [LỖI] Bước '{step_name}' thất bại với lỗi: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Quy trình Hoàn Thiện Multimodal End-to-End (EduFocus AI)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="new_data",
        help="Tên dataset (Mặc định: new_data)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=70,
        help="Số epochs rã băng tuần tự (Mặc định: 70)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        help="Số epochs không cải thiện trước khi Early Stopping (Mặc định: 20)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size (Mặc định: 16)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Thiết bị tính toán",
    )
    parser.add_argument(
        "--skip_unfreeze",
        action="store_true",
        help="Bỏ qua bước huấn luyện rã băng nếu đã có weights",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Chạy chế độ kiểm thử nhanh toàn bộ pipeline",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Kích hoạt chế độ huấn luyện nửa độ chính xác FP16 (AMP) giúp tăng tốc 2-3x và tiết kiệm 50%% VRAM",
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
        help="Số epoch Warmup khi rã băng (Mặc định: 2)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Bật chế độ tiếp tục huấn luyện (resume) từ checkpoint trong folder resume",
    )
    parser.add_argument(
        "--resume_dir",
        type=str,
        default=None,
        help="Đường dẫn thư mục chứa checkpoint resume (Mặc định: folder 'resume' ở root)",
    )
    parser.add_argument(
        "--resume_file",
        type=str,
        default=None,
        help="Tên file checkpoint cụ thể trong folder resume để nạp",
    )
    args = parser.parse_args()

    python_exe = sys.executable

    print("      EDUFOCUS AI - QUY TRÌNH NÂNG CẤP ĐỘ CHÍNH XÁC TOÀN DIỆN")
    print("      (Sequential Unfreezing -> Feature Extraction -> EPSO -> Ensemble)")
    print(f"  - Dataset Mục Tiêu: {args.dataset}")
    print(f"  - Tổng Epochs Rã Băng: {args.epochs}")
    print(f"  - Thiết Bị: {args.device}")
    print(f"  - Chế Độ FP16 (AMP): {args.fp16}")
    if args.fp16:
        print(f"  - GradScaler init_scale: {args.init_scale}")
    print(f"  - LR Warmup: {args.warmup_epochs} epochs")
    print(f"  - Chế Độ Dry-run: {args.dry_run}")
    print(f"  - Bỏ Qua Unfreeze: {args.skip_unfreeze}")
    print(f"  - Tiếp Tục (Resume): {args.resume}")
    if args.resume_dir:
        print(f"  - Thư Mục Resume: {args.resume_dir}")
    if args.resume_file:
        print(f"  - Checkpoint File: {args.resume_file}")

    # BƯỚC 1: HUẤN LUYỆN RÃ BĂNG TUẦN TỰ (100 EPOCHS)
    if not args.skip_unfreeze:
        unfreeze_script = os.path.join(
            BACKEND_DIR, "evaluation", "train_sequential_unfreeze.py"
        )
        cmd_unfreeze = [
            python_exe,
            "-u",
            unfreeze_script,
            "--dataset",
            args.dataset,
            "--epochs",
            str(args.epochs),
            "--patience",
            str(args.patience),
            "--batch_size",
            str(args.batch_size),
            "--device",
            args.device,
            "--init_scale",
            str(args.init_scale),
            "--warmup_epochs",
            str(args.warmup_epochs),
        ]
        if args.dry_run:
            cmd_unfreeze.append("--dry_run")
        if args.fp16:
            cmd_unfreeze.append("--fp16")
        if args.resume:
            cmd_unfreeze.append("--resume")
        if args.resume_dir:
            cmd_unfreeze.extend(["--resume_dir", args.resume_dir])
        if args.resume_file:
            cmd_unfreeze.extend(["--resume_file", args.resume_file])
        run_step(
            "1. Huấn Luyện Rã Băng Tuần Tự (CLIP Visual Top-4 + Projections + ResMLP Head)",
            cmd_unfreeze,
        )
    else:
        print("\n⏩ Đã bỏ qua Bước 1 (Sử dụng checkpoint fine-tuned có sẵn)...")

    # BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG VỚI BACKBONE ĐÃ FINE-TUNE
    print("\n" + "=" * 80)
    print(
        "🚀 BƯỚC 2: TRÍCH XUẤT ĐẶC TRƯNG 544D VỚI CLIP FINE-TUNED VÀ FACIAL CROSS-ATTENTION (THUẦN GPU)"
    )
    print("=" * 80)
    cache_file = os.path.join(BACKEND_DIR, f"cache_{args.dataset.lower()}.npz")
    need_reextract = (not args.skip_unfreeze) or (not os.path.exists(cache_file))
    if need_reextract:
        prep = MultiDatasetPreprocessor(device=torch.device(args.device))
        cache_out = prep.process_and_cache_dataset(args.dataset, force_reextract=True)
        prep.close()
        print(f"✅ Đã tạo/cập nhật file cache đặc trưng 544D tại: {cache_out}")
    else:
        print(f"✅ Đã có sẵn file cache đặc trưng 544D tại: {cache_file}")

    # BƯỚC 3: TỐI ƯU HÓA ĐẶC TRƯNG BẰNG EPSO
    cmd_epso = [
        python_exe,
        "-u",
        "-c",
        f"from evaluation.train_epso import train_pipeline; train_pipeline(dataset_names=['{args.dataset}'], run_benchmark_after=False)",
    ]
    run_step("3. Chọn Lọc Không Gian Đặc Trưng Bằng Thuật Toán EPSO", cmd_epso)

    # BƯỚC 4: HUẤN LUYỆN ENSEMBLE HEAD (ResMLP + XGBoost + Baseline)
    ensemble_script = os.path.join(BACKEND_DIR, "evaluation", "train_ensemble.py")
    cmd_ensemble = [python_exe, "-u", ensemble_script, "--datasets", args.dataset]
    run_step(
        "4. Huấn Luyện & Đánh Giá Mô Hình Ensemble Soft-Voting (ResMLP + XGBoost)",
        cmd_ensemble,
    )


if __name__ == "__main__":
    main()
