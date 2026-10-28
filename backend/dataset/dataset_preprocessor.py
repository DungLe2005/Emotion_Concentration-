import sys
import os
import argparse
import numpy as np
import torch
from PIL import Image
import clip
import cv2
from tqdm import tqdm

def safe_tqdm(iterable, **kwargs):
    """Wrapper an toàn cho tqdm tránh lỗi [WinError 6] The handle is invalid trên Windows PowerShell."""
    try:
        return tqdm(iterable, **kwargs, dynamic_ncols=False)
    except Exception:
        return iterable


# Thêm sys.path gốc backend
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

from dataset.dataset_loader import (
    DatasetRegistry,
    STANDARD_EMOTIONS,
    STANDARD_LABEL_TO_IDX,
    get_emotion_centroid_embeddings,
)
from features.landmark_features import LandmarkFeatureExtractor, NUM_LANDMARK_FEATURES
from models.facial_cross_attention import FacialQueryCrossAttention, extract_clip_visual_and_patches


EMOTION_PROMPTS = [
    "a face with an angry expression, furrowed eyebrows and intense glare",  # 0: angry
    "a face showing disgust, wrinkled nose and raised upper lip",             # 1: disgust
    "a fearful frightened face with wide eyes and open mouth",                # 2: fear
    "a happy smiling cheerful face with open smile",                         # 3: happy
    "a neutral calm relaxed face with a plain expression",                   # 4: neutral
    "a sad sorrowful face with downturned lips and gloomy eyes",             # 5: sad
    "a shocked surprised face with raised eyebrows and gasping mouth"        # 6: surprise
]

CLIP_DIM = 512
PROMPT_DIM = 7
TOTAL_FEATURE_DIM = CLIP_DIM + PROMPT_DIM + NUM_LANDMARK_FEATURES  # 512 + 7 + 25 = 544


class MultiDatasetPreprocessor:
    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Preprocessor] Thiết bị tính toán: {self.device}")

        print("[Preprocessor] Đang nạp mô hình CLIP ViT-B/32...")
        self.clip_model, self.clip_preprocess = clip.load("ViT-B/32", device=self.device)

        finetuned_clip_path = os.path.join(BACKEND_DIR, "models", "clip_finetuned_best.pth")
        if os.path.exists(finetuned_clip_path):
            try:
                print(f"[Preprocessor] Tìm thấy weights CLIP Fine-tuned tại '{finetuned_clip_path}'. Đang nạp...")
                state_dict = torch.load(finetuned_clip_path, map_location=self.device, weights_only=True)
                self.clip_model.load_state_dict(state_dict)
                print("[Preprocessor] ✅ Đã nạp thành công weights CLIP Fine-tuned!")
            except Exception as e:
                print(f"[Preprocessor] Cảnh báo không thể nạp fine-tuned weights ({e}), sử dụng weights mặc định.")

        self.clip_model = self.clip_model.float().eval()
        self.text_features = get_emotion_centroid_embeddings(self.clip_model, device=self.device)

        # Nạp Facial Query Cross-Attention & Visual Projection nếu có (Hướng B GPU)
        self.facial_cross_attn = None
        self.visual_proj = None
        fca_path = os.path.join(BACKEND_DIR, "models", "facial_cross_attn_best.pth")
        vproj_path = os.path.join(BACKEND_DIR, "models", "visual_proj_best.pth")
        if os.path.exists(fca_path):
            try:
                self.facial_cross_attn = FacialQueryCrossAttention(
                    embed_dim=768, num_queries=32, num_heads=8, num_layers=2, out_dim=25
                ).to(self.device).eval()
                self.facial_cross_attn.load_state_dict(torch.load(fca_path, map_location=self.device, weights_only=True))
                print("[Preprocessor] ✅ Đã nạp thành công FacialQueryCrossAttention (Hướng B GPU)!")
            except Exception as e:
                print(f"[Preprocessor] Cảnh báo không thể nạp FacialQueryCrossAttention ({e})")

        if os.path.exists(vproj_path):
            try:
                from evaluation.train_sequential_unfreeze import VisualProjection
                self.visual_proj = VisualProjection(in_dim=512, out_dim=512).to(self.device).eval()
                self.visual_proj.load_state_dict(torch.load(vproj_path, map_location=self.device, weights_only=True))
                print("[Preprocessor] ✅ Đã nạp thành công VisualProjection!")
            except Exception as e:
                print(f"[Preprocessor] Cảnh báo không thể nạp VisualProjection ({e})")

        if self.facial_cross_attn is None:
            print("[Preprocessor] Đang khởi tạo MediaPipe Landmark Feature Extractor (Fallback)...")
            self.landmark_extractor = LandmarkFeatureExtractor()
        else:
            self.landmark_extractor = None
            print("[Preprocessor] 🚀 100% GPU Mode: Đã vô hiệu hóa MediaPipe CPU!")

    def process_and_cache_dataset(
        self,
        dataset_name: str,
        output_cache_path: str = None,
        force_reextract: bool = False,
        batch_size: int = 64
    ) -> str:
        reader = DatasetRegistry.get_reader(dataset_name)
        splits = reader.get_splits()

        if not splits:
            raise ValueError(f"Không tìm thấy dữ liệu ảnh cho dataset '{dataset_name}'")

        if output_cache_path is None:
            output_cache_path = os.path.join(
                BACKEND_DIR, f"cache_{dataset_name.lower()}.npz"
            )

        if not force_reextract and os.path.exists(output_cache_path):
            print(f"[Cache Found] File cache đã tồn tại tại: {output_cache_path}")
            return output_cache_path

        print(f"\n========================================================")
        print(f"   BẮT ĐẦU TRÍCH XUẤT ĐẶC TRƯNG [Dataset: {dataset_name}]")
        print(f"========================================================")

        processed_splits = {}

        for split_name, (image_paths, labels) in splits.items():
            if len(image_paths) == 0:
                continue

            print(f"\n---> [Split: {split_name.upper()}] Số lượng ảnh: {len(image_paths)}")

            if self.facial_cross_attn is not None:
                # -------------------------------------------------------------
                # HƯỚNG B: Trích xuất 100% trên GPU với Facial Query Cross-Attention
                # Không cần bất kỳ mốc hình học nào từ MediaPipe!
                # -------------------------------------------------------------
                combined_list = []
                for i in safe_tqdm(range(0, len(image_paths), batch_size), desc=f"GPU Cross-Attention {split_name}"):
                    batch_paths = image_paths[i : i + batch_size]
                    pil_images = []
                    for p in batch_paths:
                        try:
                            img = Image.open(p).convert("RGB")
                            pil_images.append(self.clip_preprocess(img))
                        except Exception:
                            pil_images.append(torch.zeros(3, 224, 224))

                    if pil_images:
                        image_tensors = torch.stack(pil_images).to(self.device)
                        with torch.no_grad():
                            img_feats, patch_tokens = extract_clip_visual_and_patches(self.clip_model.visual, image_tensors)
                            if self.visual_proj is not None:
                                h_vis = self.visual_proj(img_feats)
                            else:
                                h_vis = img_feats
                            img_norm = img_feats / (img_feats.norm(dim=-1, keepdim=True) + 1e-7)
                            prompt_sims = (img_norm @ self.text_features.T).cpu().numpy()
                            h_facial = self.facial_cross_attn(patch_tokens).cpu().numpy()
                            h_vis_np = h_vis.cpu().numpy()

                            batch_544 = np.concatenate([h_vis_np, prompt_sims, h_facial], axis=1)
                            combined_list.append(batch_544)

                combined_features = (
                    np.concatenate(combined_list, axis=0)
                    if combined_list
                    else np.zeros((len(image_paths), TOTAL_FEATURE_DIM), dtype=np.float32)
                )
                processed_splits[f"X_{split_name}"] = combined_features
                processed_splits[f"y_{split_name}"] = np.array(labels, dtype=np.int64)
                continue

            clip_features_list = []
            prompt_sims_list = []

            for i in safe_tqdm(range(0, len(image_paths), batch_size), desc=f"CLIP {split_name}"):
                batch_paths = image_paths[i : i + batch_size]
                pil_images = []

                for p in batch_paths:
                    try:
                        img = Image.open(p).convert("RGB")
                        pil_images.append(self.clip_preprocess(img))
                    except Exception:
                        pil_images.append(torch.zeros(3, 224, 224))

                if pil_images:
                    image_tensors = torch.stack(pil_images).to(self.device)
                    with torch.no_grad():
                        img_feats = self.clip_model.encode_image(image_tensors).float()
                        img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)

                        prompt_sims = (img_feats @ self.text_features.T).cpu().numpy()

                        clip_features_list.append(img_feats.cpu().numpy())
                        prompt_sims_list.append(prompt_sims)

            clip_feats_arr = (
                np.concatenate(clip_features_list, axis=0)
                if clip_features_list
                else np.zeros((len(image_paths), CLIP_DIM), dtype=np.float32)
            )
            prompt_sims_arr = (
                np.concatenate(prompt_sims_list, axis=0)
                if prompt_sims_list
                else np.zeros((len(image_paths), PROMPT_DIM), dtype=np.float32)
            )

            # Kiểm tra xem có sẵn landmark cache cho split này không
            lm_cache_file = os.path.join(BACKEND_DIR, f"cache_{dataset_name.lower()}_{split_name.lower()}_landmarks.npz")
            landmark_features = None
            if os.path.exists(lm_cache_file):
                try:
                    cdata = np.load(lm_cache_file, allow_pickle=True)
                    if "landmarks" in cdata and len(cdata["landmarks"]) == len(image_paths):
                        landmark_features = cdata["landmarks"]
                        print(f"  [Landmark {split_name}] ✅ Đã nạp trực tiếp {len(landmark_features)} mốc hình học từ cache: {os.path.basename(lm_cache_file)}")
                except Exception:
                    landmark_features = None

            if landmark_features is None:
                landmark_features = np.zeros((len(image_paths), NUM_LANDMARK_FEATURES), dtype=np.float32)
                failed_count = 0
                for idx, img_path in enumerate(safe_tqdm(image_paths, desc=f"Landmark {split_name}")):
                    try:
                        img_bgr = cv2.imread(img_path)
                        if img_bgr is not None:
                            feat = self.landmark_extractor.extract(img_bgr)
                            landmark_features[idx] = feat
                            if np.all(feat == 0):
                                failed_count += 1
                        else:
                            failed_count += 1
                    except Exception:
                        failed_count += 1

                print(
                    f"  [Landmark {split_name}] Phát hiện thành công: "
                    f"{len(image_paths) - failed_count}/{len(image_paths)} ảnh"
                )

            combined_features = np.concatenate(
                [clip_feats_arr, prompt_sims_arr, landmark_features], axis=1
            )
            processed_splits[f"X_{split_name}"] = combined_features
            processed_splits[f"y_{split_name}"] = np.array(labels, dtype=np.int64)

        print(f"\n[Lưu Cache] Đang lưu file nén {output_cache_path}...")
        np.savez_compressed(
            output_cache_path,
            **processed_splits,
            label_to_idx=STANDARD_LABEL_TO_IDX,
            emotions=STANDARD_EMOTIONS
        )
        print(f"[Thành Công] Đã tạo cache cho dataset '{dataset_name}' ({TOTAL_FEATURE_DIM} chiều)!")
        return output_cache_path

    def close(self):
        if hasattr(self, "landmark_extractor") and self.landmark_extractor:
            self.landmark_extractor.close()


def load_dataset_cache(dataset_name: str, cache_dir: str = None) -> dict:
    if cache_dir is None:
        cache_dir = BACKEND_DIR

    possible_paths = [
        os.path.join(cache_dir, f"cache_{dataset_name.lower()}.npz"),
        os.path.join(cache_dir, "dataset_cache.npz") if dataset_name.lower() == "default" else ""
    ]

    cache_path = None
    for p in possible_paths:
        if p and os.path.exists(p):
            cache_path = p
            break

    if not cache_path:
        raise FileNotFoundError(
            f"Không tìm thấy file cache cho dataset '{dataset_name}'. "
            f"Hãy chạy `python dataset/dataset_preprocessor.py --dataset {dataset_name}` trước để trích xuất đặc trưng."
        )

    data = np.load(cache_path, allow_pickle=True)
    res = {k: data[k] for k in data.files}
    return res


def combine_dataset_caches(dataset_names: list, cache_dir: str = None) -> dict:
    combined_X_train = []
    combined_y_train = []
    combined_X_test = []
    combined_y_test = []

    for name in dataset_names:
        ds_data = load_dataset_cache(name, cache_dir=cache_dir)
        
        if "X_train" in ds_data:
            combined_X_train.append(ds_data["X_train"])
            combined_y_train.append(ds_data["y_train"])
        if "X_val" in ds_data:
            combined_X_train.append(ds_data["X_val"])
            combined_y_train.append(ds_data["y_val"])

        if "X_test" in ds_data:
            combined_X_test.append(ds_data["X_test"])
            combined_y_test.append(ds_data["y_test"])

    X_train_full = np.concatenate(combined_X_train, axis=0) if combined_X_train else np.array([])
    y_train_full = np.concatenate(combined_y_train, axis=0) if combined_y_train else np.array([])
    X_test_full = np.concatenate(combined_X_test, axis=0) if combined_X_test else np.array([])
    y_test_full = np.concatenate(combined_y_test, axis=0) if combined_y_test else np.array([])

    return {
        "X_train": X_train_full,
        "y_train": y_train_full,
        "X_test": X_test_full,
        "y_test": y_test_full,
        "label_to_idx": STANDARD_LABEL_TO_IDX,
        "emotions": STANDARD_EMOTIONS
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Dataset Feature Extractor & Cache Generator")
    parser.add_argument("--dataset", type=str, default="all", help="Tên dataset để trích xuất")
    parser.add_argument("--force", action="store_true", help="Trích xuất lại ngay cả khi đã có cache")
    args = parser.parse_args()

    preprocessor = MultiDatasetPreprocessor()

    if args.dataset.lower() == "all":
        datasets_to_process = DatasetRegistry.list_datasets()
    else:
        datasets_to_process = [args.dataset]

    for ds in datasets_to_process:
        try:
            preprocessor.process_and_cache_dataset(ds, force_reextract=args.force)
        except Exception as e:
            print(f"[ERROR] Không thể xử lý dataset '{ds}': {e}")

    preprocessor.close()
