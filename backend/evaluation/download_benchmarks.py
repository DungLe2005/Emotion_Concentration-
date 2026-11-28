"""
Công cụ Tự Động Tải & Chuẩn Hóa Benchmark FER Quốc Tế (FERPlus & AffectNet-7)
Sử dụng KaggleHub (không yêu cầu cấu hình thủ công API key) trực tiếp trên Google Colab.
Chuẩn hóa dữ liệu về đúng 7 lớp cảm xúc chuẩn:
['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
"""

import sys
import os
import shutil
import argparse
from typing import Dict, List, Optional, Tuple

# Danh sách 7 cảm xúc chuẩn quốc tế
STANDARD_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Ánh xạ tên gọi cảm xúc (case-insensitive & synonyms) — chỉ dùng cho tên chữ
SYNONYM_MAP = {
    "angry": "angry",
    "anger": "angry",
    "disgust": "disgust",
    "disgusted": "disgust",
    "fear": "fear",
    "fearful": "fear",
    "happy": "happy",
    "happiness": "happy",
    "neutral": "neutral",
    "sad": "sad",
    "sadness": "sad",
    "surprise": "surprise",
    "surprised": "surprise",
}

# Ánh xạ nhãn thư mục số cho AffectNet-7 (chuẩn AffectNet official ordering)
# 0=Neutral, 1=Happy, 2=Sad, 3=Surprise, 4=Fear, 5=Disgust, 6=Angry
AFFECTNET_NUMERIC_MAP = {
    "0": "neutral",
    "1": "happy",
    "2": "sad",
    "3": "surprise",
    "4": "fear",
    "5": "disgust",
    "6": "angry",
}

# Ánh xạ nhãn thư mục số cho FER2013/FERPlus (nếu dùng thư mục số)
FER2013_NUMERIC_MAP = {
    "0": "angry",
    "1": "disgust",
    "2": "fear",
    "3": "happy",
    "4": "sad",
    "5": "surprise",
    "6": "neutral",
}

# Các handle KaggleHub uy tín cho từng benchmark (ưu tiên arnabkumarroy02/ferplus & lintongdai/affectnet7)
CANDIDATE_HANDLES = {
    "ferplus": [
        "arnabkumarroy02/ferplus",
        "msambare/fer2013",
        "debanga/ferplus",
        "davidadrian/ferplus",
        "deadskull7/fer2013",
    ],
    "affectnet": [
        "lintongdai/affectnet7",
        "thienhuynh/affectnet-validation-set",
        "noamsegal/affectnet-training-data",
        "adityashrm21/affectnet-val",
        "sujaykapadnis/affectnet-7-classes",
    ],
}


def ensure_kagglehub():
    """Kiểm tra và tự động cài đặt thư viện kagglehub nếu chưa có."""
    try:
        import kagglehub
        return kagglehub
    except ImportError:
        print("📦 Chưa tìm thấy thư viện 'kagglehub'. Đang tự động cài đặt...")
        import subprocess
        res = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "kagglehub"])
        if res.returncode == 0:
            import kagglehub
            print("✅ Đã cài đặt 'kagglehub' thành công!")
            return kagglehub
        else:
            raise RuntimeError(
                "Không thể tự động cài đặt kagglehub. Hãy chạy lệnh: pip install kagglehub"
            )


def safe_link_or_copy(src_path: str, dst_path: str):
    """Tạo symlink nếu được phép, nếu lỗi (Windows không có dev mode) thì copy file."""
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        return
    try:
        os.symlink(src_path, dst_path)
    except Exception:
        shutil.copy2(src_path, dst_path)


def discover_emotion_dirs(root_search_dir: str, target_split: str = "test", dataset_type: str = "ferplus") -> Dict[str, List[str]]:
    """
    Quét thông minh thư mục đã tải để gom các file ảnh theo 7 lớp cảm xúc chuẩn.
    Hỗ trợ cấu trúc đa dạng:
    - root/Test/Angry/img.jpg
    - root/test/angry/img.jpg
    - root/val/happy/img.jpg
    - root/Validation/Happy/img.jpg
    - root/angry/img.jpg
    - root/val/0/img.jpg (AffectNet numeric folders)
    """
    valid_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    found_images: Dict[str, List[str]] = {emo: [] for emo in STANDARD_EMOTIONS}

    # Chọn bảng ánh xạ số phù hợp theo loại dataset
    if dataset_type == "affectnet":
        numeric_map = AFFECTNET_NUMERIC_MAP
    else:
        numeric_map = FER2013_NUMERIC_MAP

    candidate_subdirs = []
    for root, dirs, _ in os.walk(root_search_dir):
        base = os.path.basename(root).lower()
        if target_split in ("test", "val", "validation"):
            if target_split == "test" and base in ("test", "publictest", "privatetest"):
                candidate_subdirs.append(root)
            elif target_split in ("val", "validation") and base in ("val", "validation", "val_set", "valid"):
                candidate_subdirs.append(root)

    search_roots = candidate_subdirs if candidate_subdirs else [root_search_dir]

    for s_root in search_roots:
        for root, dirs, files in os.walk(s_root):
            base_folder = os.path.basename(root).lower()
            matched_emo = None
            # 1) Nếu tên thư mục là số, dùng bảng ánh xạ riêng theo dataset
            if base_folder.isdigit() and base_folder in numeric_map:
                matched_emo = numeric_map[base_folder]
            else:
                # 2) Nếu tên thư mục là chữ, dùng bảng synonyms
                for key, std_name in SYNONYM_MAP.items():
                    if key in base_folder or base_folder == key:
                        matched_emo = std_name
                        break

            if matched_emo:
                for f in files:
                    if f.lower().endswith(valid_extensions):
                        found_images[matched_emo].append(os.path.join(root, f))

    total_found = sum(len(v) for v in found_images.values())
    if total_found == 0:
        for root, dirs, files in os.walk(root_search_dir):
            base_folder = os.path.basename(root).lower()
            matched_emo = None
            if base_folder.isdigit() and base_folder in numeric_map:
                matched_emo = numeric_map[base_folder]
            else:
                for key, std_name in SYNONYM_MAP.items():
                    if key in base_folder or base_folder == key:
                        matched_emo = std_name
                        break
            if matched_emo:
                for f in files:
                    if f.lower().endswith(valid_extensions):
                        found_images[matched_emo].append(os.path.join(root, f))

    return found_images


def download_and_organize_benchmark(
    benchmark_name: str,
    dest_dir: str,
    handles: List[str],
    target_split: str,
    force: bool = False,
) -> Optional[str]:
    """Tải và chuẩn hóa một benchmark cụ thể vào dest_dir."""
    target_folder_name = "ferplus" if "fer" in benchmark_name.lower() else "affectnet"
    split_name = "test" if target_folder_name == "ferplus" else "val"
    final_output_path = os.path.join(dest_dir, target_folder_name, split_name)

    if os.path.exists(final_output_path) and not force:
        existing_counts = {}
        for emo in STANDARD_EMOTIONS:
            emo_dir = os.path.join(final_output_path, emo)
            if os.path.exists(emo_dir):
                cnt = len([f for f in os.listdir(emo_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
                existing_counts[emo] = cnt
        if sum(existing_counts.values()) > 500:
            print(f"\n✅ Benchmark '{benchmark_name}' đã tồn tại sẵn tại: {final_output_path}")
            print("  Thống kê số ảnh hiện có:")
            for emo, cnt in existing_counts.items():
                print(f"    - {emo:10s}: {cnt:5d} ảnh")
            print(f"  -> Tổng cộng: {sum(existing_counts.values())} ảnh.")
            return final_output_path

    kagglehub = ensure_kagglehub()

    print(f"\n{'='*70}")
    print(f"🚀 BẮT ĐẦU TẢI BENCHMARK: {benchmark_name.upper()} QUA KAGGLEHUB")
    print(f"{'='*70}")

    download_dir = None
    last_error = None
    for handle in handles:
        try:
            print(f"  ⏳ Đang kết nối KaggleHub tải handle: '{handle}'...")
            download_dir = kagglehub.dataset_download(handle)
            if download_dir and os.path.exists(download_dir):
                print(f"  ✅ Tải thành công handle '{handle}' về cache: {download_dir}")
                break
        except Exception as e:
            print(f"  ⚠️ Handle '{handle}' không thành công: {e}")
            last_error = e

    if not download_dir or not os.path.exists(download_dir):
        print(f"❌ Không thể tải benchmark '{benchmark_name}'. Lỗi cuối: {last_error}")
        return None

    print(f"  🔍 Đang quét cấu trúc file và nhận diện 7 lớp cảm xúc...")
    found_images = discover_emotion_dirs(download_dir, target_split=target_split, dataset_type=target_folder_name)

    total_imgs = sum(len(v) for v in found_images.values())
    if total_imgs == 0:
        print(f"❌ Không tìm thấy ảnh cảm xúc nào hợp lệ trong: {download_dir}")
        return None

    print(f"  📁 Đang tổ chức và liên kết dữ liệu vào: {final_output_path}...")
    os.makedirs(final_output_path, exist_ok=True)

    summary_counts = {}
    for emo in STANDARD_EMOTIONS:
        dest_emo_dir = os.path.join(final_output_path, emo)
        os.makedirs(dest_emo_dir, exist_ok=True)
        img_list = found_images.get(emo, [])
        for src_img in img_list:
            fname = os.path.basename(src_img)
            dst_img = os.path.join(dest_emo_dir, fname)
            safe_link_or_copy(src_img, dst_img)
        summary_counts[emo] = len(img_list)

    print(f"\n🎉 [HOÀN TẤT] Benchmark '{benchmark_name}' đã sẵn sàng tại:")
    print(f"👉 Đường dẫn: {final_output_path}")
    print("  Chi tiết phân bố 7 lớp cảm xúc:")
    for emo in STANDARD_EMOTIONS:
        print(f"    - {emo:10s}: {summary_counts[emo]:5d} ảnh")
    print(f"  -> Tổng số ảnh chuẩn hóa: {total_imgs:,} ảnh\n")

    return final_output_path


def main():
    parser = argparse.ArgumentParser(
        description="Tải và chuẩn hóa bộ dữ liệu kiểm thử ngoài miền (FERPlus / AffectNet-7) bằng KaggleHub"
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        choices=["all", "ferplus", "affectnet", "fer2013"],
        help="Benchmark cần tải: all, ferplus, affectnet (Mặc định: all)",
    )
    parser.add_argument(
        "--dest_dir",
        type=str,
        default="/content/datasets" if os.path.exists("/content") else "datasets",
        help="Thư mục lưu trữ đích (Mặc định: /content/datasets trên Colab hoặc datasets cục bộ)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bắt buộc tải lại dù dữ liệu đã có sẵn",
    )

    args = parser.parse_args()
    dest_dir = os.path.abspath(args.dest_dir)
    os.makedirs(dest_dir, exist_ok=True)

    print(f"📌 Thư mục lưu dữ liệu đích: {dest_dir}")

    targets = []
    if args.datasets in ("all", "ferplus", "fer2013"):
        targets.append(("ferplus", CANDIDATE_HANDLES["ferplus"], "test"))
    if args.datasets in ("all", "affectnet"):
        targets.append(("affectnet", CANDIDATE_HANDLES["affectnet"], "val"))

    results = {}
    for name, handles, split in targets:
        path = download_and_organize_benchmark(
            benchmark_name=name,
            dest_dir=dest_dir,
            handles=handles,
            target_split=split,
            force=args.force,
        )
        results[name] = path

    print("=" * 70)
    print("📋 TỔNG KẾT QUÁ TRÌNH TẢI DỮ LIỆU BENCHMARK:")
    for name, p in results.items():
        status = "✅ SẴN SÀNG" if p and os.path.exists(p) else "❌ THẤT BÀI"
        print(f"  - {name.upper():12s}: {status} -> {p}")
    print("=" * 70)


if __name__ == "__main__":
    main()
