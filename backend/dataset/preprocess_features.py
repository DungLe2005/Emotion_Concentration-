import sys
import os
import cv2
import pandas as pd
import numpy as np
import torch
import math
from tqdm import tqdm
from PIL import Image
import clip
import torch.nn as nn
from retinaface import RetinaFace
import tensorflow as tf

BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)


class RealCLIVP_FER:
    """
    Tích hợp trực tiếp mã nguồn trích xuất đặc trưng của CLIVP-FER (Dựa trên CLIP ViT-B/32).
    """

    def __init__(self, device):
        self.device = device
        print("Đang tải trọng số mô hình CLIP ViT-B/32...")
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.clip_model.eval()

        self.avg_pool = nn.AvgPool1d(kernel_size=2)
        self.pooling_layer = nn.AvgPool1d(kernel_size=2, stride=2)

    def extract(self, face_img):
        img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        images = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.clip_model.encode_image(images)
            features = image_features.view(-1).float()

        return features.cpu()


def random_rotate(image, max_angle=10):
    angle = np.random.uniform(-max_angle, max_angle)
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def random_brightness(image, max_delta=30):
    delta = np.random.uniform(-max_delta, max_delta)
    image = image.astype(np.float32) + delta
    return np.clip(image, 0, 255).astype(np.uint8)


def preprocess_and_save_features(force_reextract=True):
    project_root = os.path.abspath(os.path.join(BACKEND_DIR, ".."))
    csv_path = os.path.join(project_root, "data", "labels.csv")
    train_dir = os.path.join(project_root, "data", "Train")
    test_dir = os.path.join(project_root, "data", "Test")
    output_dir = os.path.join(project_root, "data", "extracted_features")

    os.makedirs(output_dir, exist_ok=True)
    if not os.path.exists(csv_path):
        print(f"Không tìm thấy csv_path: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clivp_model = RealCLIVP_FER(device)

    print("Đang tải mạng cắt khuôn mặt RetinaFace...")
    detector_model = RetinaFace.build_model()

    print(f"Bắt đầu trích xuất siêu tốc cho {len(df)} ảnh...")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        is_train = False
        test_img_path = os.path.join(test_dir, row["pth"])
        train_img_path = os.path.join(train_dir, row["pth"])

        if os.path.exists(test_img_path):
            is_train = False
            img_path = test_img_path
        elif os.path.exists(train_img_path):
            is_train = True
            img_path = train_img_path
        else:
            continue

        rel_dir = os.path.dirname(row["pth"])
        save_dir = os.path.join(output_dir, rel_dir)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(
            output_dir, row["pth"].replace(".jpg", ".pt").replace(".png", ".pt")
        )

        if not force_reextract and os.path.exists(save_path):
            try:
                MIN_VALID_SIZE = 1000
                if os.path.getsize(save_path) > MIN_VALID_SIZE:
                    if not is_train:
                        continue
                    else:
                        base_feat_path = save_path.replace(".pt", "")
                        all_aug_exist = all(
                            os.path.exists(f"{base_feat_path}_aug_{i}.pt")
                            and os.path.getsize(f"{base_feat_path}_aug_{i}.pt") > MIN_VALID_SIZE
                            for i in range(3)
                        )
                        if all_aug_exist:
                            continue
            except Exception:
                pass

        img = cv2.imread(img_path)
        if img is None:
            continue

        try:
            detections = RetinaFace.detect_faces(img, model=detector_model)
        except Exception as e:
            detections = None
            print(f"Lỗi RetinaFace: {e}")

        faces_data = []
        if isinstance(detections, dict):
            for key in detections:
                face = detections[key]
                if "facial_area" in face and "landmarks" in face:
                    faces_data.append(
                        {
                            "facial_area": face["facial_area"],
                            "landmarks": face["landmarks"],
                        }
                    )

        feature_vector = None
        if len(faces_data) > 0:
            primary_face = max(
                faces_data,
                key=lambda f: (f["facial_area"][2] - f["facial_area"][0])
                * (f["facial_area"][3] - f["facial_area"][1]),
            )

            box = primary_face["facial_area"]
            landmarks = primary_face["landmarks"]
            x1, y1, x2, y2 = [int(b) for b in box]

            w = x2 - x1
            h = y2 - y1

            top_pad = int(h * 0.20)
            y1_new = max(0, y1 - top_pad)
            h_new = y2 - y1_new

            center_x = x1 + w / 2.0
            center_y = y1_new + h_new / 2.0

            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]

            if left_eye[0] > right_eye[0]:
                left_eye, right_eye = right_eye, left_eye

            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = math.degrees(math.atan2(dY, dX))

            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            rotated_img = cv2.warpAffine(
                img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR
            )

            pad_w = w * 0.15
            pad_h = h_new * 0.15
            new_w = w + 2 * pad_w
            new_h = h_new + 2 * pad_h
            square_size = int(max(new_w, new_h))

            face_img = cv2.getRectSubPix(
                rotated_img, (square_size, square_size), (center_x, center_y)
            )

            if face_img is not None and face_img.size > 0:
                h_crop, w_crop = face_img.shape[:2]

                if h_crop < 100 or w_crop < 100:
                    face_img_processed = cv2.resize(
                        face_img, (224, 224), interpolation=cv2.INTER_LANCZOS4
                    )
                    ycrcb = cv2.cvtColor(face_img_processed, cv2.COLOR_BGR2YCrCb)
                    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(4, 4))
                    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
                    enhanced_face = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
                else:
                    face_img_processed = cv2.bilateralFilter(
                        face_img, d=7, sigmaColor=50, sigmaSpace=50
                    )
                    ycrcb = cv2.cvtColor(face_img_processed, cv2.COLOR_BGR2YCrCb)
                    clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(8, 8))
                    ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
                    face_img_processed = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

                    blurred = cv2.GaussianBlur(face_img_processed, (5, 5), 1.0)
                    face_img_processed = cv2.addWeighted(
                        face_img_processed, 1.2, blurred, -0.2, 0
                    )
                    enhanced_face = cv2.resize(
                        face_img_processed, (224, 224), interpolation=cv2.INTER_LANCZOS4
                    )

                cropped_dir = os.path.join(project_root, "data", "cropped_faces")
                save_cropped_dir = os.path.join(cropped_dir, rel_dir)
                os.makedirs(save_cropped_dir, exist_ok=True)

                cropped_path = os.path.join(
                    save_cropped_dir, os.path.basename(row["pth"])
                )
                cv2.imwrite(cropped_path, enhanced_face)

                feature_vector = clivp_model.extract(enhanced_face)

                if is_train:
                    flip_img = cv2.flip(enhanced_face, 1)
                    feat_flip = clivp_model.extract(flip_img)
                    torch.save(feat_flip, save_path.replace(".pt", "_aug_0.pt"))

                    rot_img = random_rotate(enhanced_face, max_angle=10)
                    feat_rot = clivp_model.extract(rot_img)
                    torch.save(feat_rot, save_path.replace(".pt", "_aug_1.pt"))

                    bright_img = random_brightness(enhanced_face, max_delta=30)
                    feat_bright = clivp_model.extract(bright_img)
                    torch.save(feat_bright, save_path.replace(".pt", "_aug_2.pt"))

        if feature_vector is None:
            feature_vector = torch.zeros(512)
            if is_train:
                for i in range(3):
                    torch.save(
                        torch.zeros(512), save_path.replace(".pt", f"_aug_{i}.pt")
                    )

        torch.save(feature_vector, save_path)


if __name__ == "__main__":
    preprocess_and_save_features(force_reextract=False)
