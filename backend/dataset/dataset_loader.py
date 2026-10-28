import sys
import os
import json
import pandas as pd
import torch
from typing import Dict, List, Tuple, Optional

# Thêm sys.path gốc backend
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

STANDARD_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
STANDARD_LABEL_TO_IDX = {emotion: idx for idx, emotion in enumerate(STANDARD_EMOTIONS)}

DEFAULT_EMOTION_PROMPTS = [
    "a face with an angry expression, furrowed eyebrows and intense glare",  # 0: angry
    "a face showing disgust, wrinkled nose and raised upper lip",             # 1: disgust
    "a fearful frightened face with wide eyes and open mouth",                # 2: fear
    "a happy smiling cheerful face with open smile",                         # 3: happy
    "a neutral calm relaxed face with a plain expression",                   # 4: neutral
    "a sad sorrowful face with downturned lips and gloomy eyes",             # 5: sad
    "a shocked surprised face with raised eyebrows and gasping mouth"        # 6: surprise
]


def get_emotion_centroid_embeddings(
    clip_model,
    device: Optional[torch.device] = None,
    desc_path: Optional[str] = None,
) -> torch.Tensor:
    """
    Tính toán vector biểu diễn ngữ nghĩa chuẩn hóa (7, 512) cho 7 lớp cảm xúc chuẩn:
    Nếu có file fer_descriptions.json (mô tả Action Units chi tiết sinh bởi LLM):
      -> Sử dụng kỹ thuật Prompt Ensembling (Centroid Embedding - CuPL CVPR 2023):
         Mã hóa toàn bộ các câu mô tả vi biểu cảm cho từng cảm xúc, tính trung bình cộng (Centroid)
         và L2-normalize lại. Cho ra mỏ neo ngữ nghĩa cực kỳ vững chắc, bao phủ đầy đủ các Action Units.
    Nếu không có -> Fallback về 7 câu prompt đơn lẻ mặc định.

    Returns:
        torch.Tensor hình dạng (7, 512) kiểu float32 trên thiết bị `device`.
    """
    import clip

    if device is None:
        device = next(clip_model.parameters()).device

    if desc_path is None:
        desc_path = os.path.join(BACKEND_DIR, "fer_descriptions.json")

    if os.path.exists(desc_path):
        try:
            with open(desc_path, "r", encoding="utf-8") as f:
                raw_data = json.load(f)
            data_lower = {k.lower(): v for k, v in raw_data.items()}

            centroids = []
            with torch.no_grad():
                for idx, emo in enumerate(STANDARD_EMOTIONS):
                    prompts = data_lower.get(emo, [])
                    if not prompts:
                        prompts = [DEFAULT_EMOTION_PROMPTS[idx]]
                    toks = clip.tokenize(prompts, truncate=True).to(device)
                    feats = clip_model.encode_text(toks).float()
                    feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-7)
                    centroid = feats.mean(dim=0, keepdim=True)
                    centroid = centroid / (centroid.norm(dim=-1, keepdim=True) + 1e-7)
                    centroids.append(centroid)

            text_features = torch.cat(centroids, dim=0).float()
            print(f"[AI Text] 🎯 Đã nạp & tính Centroid Embedding từ '{os.path.basename(desc_path)}' (7 cảm xúc x Action Units, shape {list(text_features.shape)})!")
            return text_features
        except Exception as e:
            print(f"[AI Text Warning] Lỗi nạp fer_descriptions.json ({e}), chuyển sang prompt mặc định.")

    # Fallback mặc định
    toks = clip.tokenize(DEFAULT_EMOTION_PROMPTS).to(device)
    with torch.no_grad():
        feats = clip_model.encode_text(toks).float()
        text_features = (feats / (feats.norm(dim=-1, keepdim=True) + 1e-7)).float()
    print("[AI Text] Đã tạo text features chuẩn từ 7 câu prompt mặc định (7, 512).")
    return text_features


class BaseDatasetReader:
    """
    Interface cơ sở cho tất cả các Bộ đọc Dữ liệu (Dataset Readers).
    Chuẩn hóa đầu ra thành danh sách đường dẫn ảnh và nhãn số integer (0..6).
    """

    def __init__(self, dataset_name: str, root_dir: str):
        self.dataset_name = dataset_name
        self.root_dir = root_dir

    def get_splits(self) -> Dict[str, Tuple[List[str], List[int]]]:
        raise NotImplementedError


class FolderDatasetReader(BaseDatasetReader):
    def __init__(
        self,
        dataset_name: str,
        root_dir: str,
        split_folders: Optional[Dict[str, str]] = None,
        emotion_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(dataset_name, root_dir)
        self.split_folders = split_folders or {
            "train": "train",
            "val": "validation",
            "test": "test",
        }
        self.emotion_map = emotion_map or {e: e for e in STANDARD_EMOTIONS}

    def get_splits(self) -> Dict[str, Tuple[List[str], List[int]]]:
        splits_data = {}

        for split_key, folder_name in self.split_folders.items():
            split_dir = os.path.join(self.root_dir, folder_name)
            if not os.path.exists(split_dir):
                alt_split_dir = os.path.join(self.root_dir, split_key)
                if os.path.exists(alt_split_dir):
                    split_dir = alt_split_dir
                else:
                    continue

            image_paths = []
            labels = []

            for local_emo, std_emo in self.emotion_map.items():
                if std_emo not in STANDARD_LABEL_TO_IDX:
                    continue
                std_idx = STANDARD_LABEL_TO_IDX[std_emo]

                emo_dir = os.path.join(split_dir, local_emo)
                if not os.path.exists(emo_dir):
                    continue

                for fname in os.listdir(emo_dir):
                    if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
                        image_paths.append(os.path.join(emo_dir, fname))
                        labels.append(std_idx)

            splits_data[split_key] = (image_paths, labels)

        return splits_data


class KMUFEDDatasetReader(BaseDatasetReader):
    """
    Bộ đọc dữ liệu KMU-FED (Korean Multi-University Facial Expression Database).
    Dữ liệu gồm 1106 ảnh khuôn mặt từ 12 người, 6 cảm xúc (AN, DI, FE, HA, SA, SU).
    Tên file theo format: {subject}_{emotion}_{session}_{frame}.jpg
    Chia train/test theo 10-fold cross-validation (giống CLIVP-FER).
    """

    # Map mã cảm xúc KMU-FED → tên chuẩn 7 lớp (không có neutral)
    KMUFED_EMOTION_MAP = {
        "AN": "angry",
        "DI": "disgust",
        "FE": "fear",
        "HA": "happy",
        "SA": "sad",
        "SU": "surprise",
    }

    # Thứ tự sắp xếp cảm xúc và số lượng ảnh tích lũy (khớp CLIVP-FER)
    EMOTION_ORDER = ["AN", "DI", "FE", "HA", "SA", "SU"]
    SUM_NUMBER = [0, 196, 316, 516, 725, 905, 1104]  # Vị trí bắt đầu tích lũy mỗi cảm xúc
    TEST_NUMBER = [19, 12, 20, 21, 18, 20]  # Số ảnh test mỗi fold cho mỗi cảm xúc

    def __init__(
        self,
        dataset_name: str = "kmufed",
        root_dir: str = r"d:\Emotion_Concentration\KMU-FED",
        split_by_subject: bool = True,
        test_subjects: List[int] = [10, 11, 12],
        fold: int = 1,
    ):
        super().__init__(dataset_name, root_dir)
        self.split_by_subject = split_by_subject
        self.test_subjects = set(test_subjects)
        self.fold = fold

    def _collect_all_images_sorted(self) -> List[Tuple[str, int]]:
        """
        Thu thập và sắp xếp tất cả ảnh theo thứ tự cảm xúc (AN, DI, FE, HA, SA, SU).
        """
        all_items = []  # List[(path, std_label_idx)]

        for emo_code in self.EMOTION_ORDER:
            std_emo = self.KMUFED_EMOTION_MAP.get(emo_code)
            if std_emo not in STANDARD_LABEL_TO_IDX:
                continue
            std_idx = STANDARD_LABEL_TO_IDX[std_emo]

            emo_files = []
            for fname in os.listdir(self.root_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                parts = fname.split("_")
                if len(parts) >= 2 and parts[1] == emo_code:
                    emo_files.append(fname)

            emo_files.sort()
            for fname in emo_files:
                all_items.append((os.path.join(self.root_dir, fname), std_idx))

        return all_items

    def get_splits(self) -> Dict[str, Tuple[List[str], List[int]]]:
        """
        Chia train/test theo Person-Independent Split (Subjects 01-09 Train, Subjects 10-12 Test)
        để đảm bảo không bị rò rỉ dữ liệu khuôn mặt (Identity/Temporal Leakage).
        """
        train_paths, train_labels = [], []
        test_paths, test_labels = [], []

        if self.split_by_subject:
            for fname in os.listdir(self.root_dir):
                if not fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                    continue
                parts = fname.split("_")
                if len(parts) < 2:
                    continue
                emo_code = parts[1]
                std_emo = self.KMUFED_EMOTION_MAP.get(emo_code)
                if not std_emo or std_emo not in STANDARD_LABEL_TO_IDX:
                    continue

                std_idx = STANDARD_LABEL_TO_IDX[std_emo]
                img_path = os.path.join(self.root_dir, fname)

                try:
                    subj_num = int(parts[0])
                except ValueError:
                    subj_num = 1

                if subj_num in self.test_subjects:
                    test_paths.append(img_path)
                    test_labels.append(std_idx)
                else:
                    train_paths.append(img_path)
                    train_labels.append(std_idx)
        else:
            all_items = self._collect_all_images_sorted()
            if not all_items:
                return {}

            total = len(all_items)
            test_indices = set()
            for j in range(len(self.TEST_NUMBER)):
                for k in range(self.TEST_NUMBER[j]):
                    if self.fold != 10:
                        idx = self.SUM_NUMBER[j] + (self.fold - 1) * self.TEST_NUMBER[j] + k
                    else:
                        idx = self.SUM_NUMBER[j + 1] - 1 - k
                    if idx < total:
                        test_indices.add(idx)

            for i, (path, label) in enumerate(all_items):
                if i in test_indices:
                    test_paths.append(path)
                    test_labels.append(label)
                else:
                    train_paths.append(path)
                    train_labels.append(label)

        splits_data = {}
        if train_paths:
            splits_data["train"] = (train_paths, train_labels)
        if test_paths:
            splits_data["test"] = (test_paths, test_labels)

        return splits_data


class CSVDatasetReader(BaseDatasetReader):
    def __init__(
        self,
        dataset_name: str,
        csv_file: str,
        image_dir: str,
        path_col: str = "path",
        label_col: str = "label",
        label_map: Optional[Dict[str, str]] = None,
    ):
        super().__init__(dataset_name, image_dir)
        self.csv_file = csv_file
        self.image_dir = image_dir
        self.path_col = path_col
        self.label_col = label_col
        self.label_map = label_map or {}

    def get_splits(self) -> Dict[str, Tuple[List[str], List[int]]]:
        if not os.path.exists(self.csv_file):
            return {}

        df = pd.read_csv(self.csv_file)
        train_paths, train_labels = [], []
        test_paths, test_labels = [], []

        for _, row in df.iterrows():
            rel_path = str(row[self.path_col])
            raw_label = str(row[self.label_col]).lower()

            std_emo = self.label_map.get(raw_label, raw_label)
            if std_emo not in STANDARD_LABEL_TO_IDX:
                continue

            std_idx = STANDARD_LABEL_TO_IDX[std_emo]
            full_path = os.path.join(self.image_dir, rel_path)

            if os.path.exists(full_path):
                if "split" in df.columns and str(row["split"]).lower() == "test":
                    test_paths.append(full_path)
                    test_labels.append(std_idx)
                else:
                    train_paths.append(full_path)
                    train_labels.append(std_idx)

        res = {}
        if train_paths:
            res["train"] = (train_paths, train_labels)
        if test_paths:
            res["test"] = (test_paths, test_labels)

        return res


class RAFDBDatasetReader(FolderDatasetReader):
    """
    Bộ đọc dữ liệu RAF-DB (Real-world Affective Faces Database).
    Cấu trúc: DATASET/train/{1..7}/, DATASET/test/{1..7}/
    Label mapping: 1=surprise, 2=fear, 3=disgust, 4=happy, 5=sad, 6=angry, 7=neutral
    """

    RAFDB_EMOTION_MAP = {
        "1": "surprise",
        "2": "fear",
        "3": "disgust",
        "4": "happy",
        "5": "sad",
        "6": "angry",
        "7": "neutral",
    }

    def __init__(self, dataset_name: str = "rafdb", root_dir: str = r"d:\Emotion_Concentration\RAF_DB"):
        dataset_dir = os.path.join(root_dir, "DATASET")
        super().__init__(
            dataset_name=dataset_name,
            root_dir=dataset_dir,
            split_folders={"train": "train", "test": "test"},
            emotion_map=self.RAFDB_EMOTION_MAP,
        )


class DatasetRegistry:
    _registry: Dict[str, BaseDatasetReader] = {}

    @classmethod
    def register(cls, name: str, reader: BaseDatasetReader):
        cls._registry[name.lower()] = reader

    @classmethod
    def get_reader(cls, name: str) -> BaseDatasetReader:
        name_clean = name.lower()
        if name_clean in cls._registry:
            return cls._registry[name_clean]
        raise ValueError(f"Bộ dữ liệu '{name}' chưa được đăng ký trong DatasetRegistry!")

    @classmethod
    def list_datasets(cls) -> List[str]:
        return list(cls._registry.keys())

    @classmethod
    def auto_discover_defaults(cls, project_root: str = r"d:\Emotion_Concentration"):
        new_data_path = os.path.join(project_root, "new_data")
        if os.path.exists(new_data_path):
            cls.register("new_data", FolderDatasetReader("new_data", new_data_path))

        kmufed_path = os.path.join(project_root, "KMU-FED")
        if os.path.exists(kmufed_path):
            cls.register("kmufed", KMUFEDDatasetReader("kmufed", kmufed_path))

        rafdb_path = os.path.join(project_root, "RAF_DB")
        if os.path.exists(rafdb_path):
            cls.register("rafdb", RAFDBDatasetReader("rafdb", rafdb_path))

        config_path = os.path.join(BACKEND_DIR, "dataset_config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as f:
                    configs = json.load(f)
                    for ds_name, cfg in configs.items():
                        ds_type = cfg.get("type", "folder")
                        root_dir = cfg.get("root_dir", "")
                        if ds_type == "folder":
                            cls.register(
                                ds_name,
                                FolderDatasetReader(
                                    ds_name,
                                    root_dir,
                                    split_folders=cfg.get("splits"),
                                    emotion_map=cfg.get("emotion_map"),
                                ),
                            )
                        elif ds_type == "csv":
                            cls.register(
                                ds_name,
                                CSVDatasetReader(
                                    ds_name,
                                    csv_file=cfg.get("csv_file"),
                                    image_dir=root_dir,
                                    path_col=cfg.get("path_col", "path"),
                                    label_col=cfg.get("label_col", "label"),
                                    label_map=cfg.get("label_map"),
                                ),
                            )
            except Exception as e:
                print(f"[DatasetRegistry] Lỗi đọc dataset_config.json: {e}")


DatasetRegistry.auto_discover_defaults()


if __name__ == "__main__":
    print(f"Danh sách dataset đã đăng ký: {DatasetRegistry.list_datasets()}")
