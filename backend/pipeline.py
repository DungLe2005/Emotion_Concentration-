import time
import numpy as np
import cv2
import torch
import pandas as pd
import base64
import sys
import os
import math
from retinaface import RetinaFace
from preprocess_features import RealCLIVP_FER
from models.mlp import EmotionMLP


class EmotionPipeline:
    """
    Class quản lý toàn bộ luồng xử lý AI trong thời gian thực.
    Được thiết kế để khởi tạo 1 lần (Singleton) nhằm tiết kiệm RAM và thời gian.
    """

    def __init__(self):
        print("\n[AI] Đang khởi tạo toàn bộ Pipeline Models...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # 2. Khởi tạo CLIVP-FER
        self.clivp_model = RealCLIVP_FER(self.device)

        # 2.5 Khởi tạo RetinaFace Model (Chỉ load weights 1 lần)
        print("[AI] Đang nạp RetinaFace Model...")
        self.retinaface_model = RetinaFace.build_model()

        # 3. Load EPSO Mask (BẮT BUỘC phải tồn tại - chạy train_epso.py để tạo)
        epso_mask_path = os.path.join(os.path.dirname(__file__), "epso_mask_real.npy")
        if not os.path.exists(epso_mask_path):
            raise FileNotFoundError(
                f"[AI] LỖI NGHIÊM TRỌNG: Không tìm thấy file '{epso_mask_path}'. "
                "Cần chạy train_epso.py trước để tạo file này."
            )
        self.epso_mask = np.load(epso_mask_path)
        reduced_dim = len(self.epso_mask)

        # 4. Tìm nhãn (labels mapping)
        df = pd.read_csv(r"d:\Emotion_Concentration\data\labels.csv")
        labels_list = df["label"].unique()
        self.idx_to_label = {idx: label for idx, label in enumerate(labels_list)}

        # 5. Load MLP (EPSO) - KHÔNG catch lỗi để phát hiện ngay nếu weights sai
        self.mlp = EmotionMLP(input_dim=reduced_dim, num_classes=len(labels_list)).to(
            self.device
        )
        self.mlp.load_state_dict(
            torch.load(
                "emotion_mlp_epso_weights.pth",
                map_location=self.device,
                weights_only=True,
            )
        )
        self.mlp.eval()

        print(f"[AI] Đã nạp thành công MLP với đầu vào {len(self.epso_mask)} chiều.")
        print("[AI] Pipeline đã sẵn sàng nhận ảnh!\n")

    def run(self, image_bytes):
        logs = []
        start_time = time.time()

        # 1. Decode Image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        stage1_time = time.time()
        logs.append(
            {
                "stage": "Image Decoding",
                "latency_ms": round((stage1_time - start_time) * 1000, 2),
                "message": "Giải mã ảnh thành công",
            }
        )

        if img is None:
            return {
                "emotion": "Error",
                "concentration": 0,
                "faces": [],
                "logs": logs,
                "error": "Ảnh không hợp lệ",
            }

        # 2. Face Detection với RetinaFace
        stage2_start = time.time()
        try:
            detections = RetinaFace.detect_faces(img, model=self.retinaface_model)
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

        if len(faces_data) == 0:
            stage2_time = time.time()
            logs.append(
                {
                    "stage": "Face Detection (RetinaFace)",
                    "latency_ms": round((stage2_time - stage2_start) * 1000, 2),
                    "message": "Không tìm thấy khuôn mặt",
                }
            )
            return {"emotion": "No Face", "concentration": 0, "faces": [], "logs": logs}

        stage2_time = time.time()
        logs.append(
            {
                "stage": "Face Detection (RetinaFace)",
                "latency_ms": round((stage2_time - stage2_start) * 1000, 2),
                "message": f"Tìm thấy {len(faces_data)} khuôn mặt",
            }
        )

        faces_list = []
        stage3_total_time = 0
        stage4_total_time = 0
        stage5_total_time = 0

        for idx, face_info in enumerate(faces_data):
            box = face_info["facial_area"]
            landmarks = face_info["landmarks"]
            x1, y1, x2, y2 = [int(b) for b in box]

            w = x2 - x1
            h = y2 - y1

            # Thêm 20% padding phía trên để lấy trọn vẹn trán và tóc (tránh cắt mất phần đầu)
            top_pad = int(h * 0.20)
            y1_new = max(0, y1 - top_pad)
            h_new = y2 - y1_new

            # Tọa độ tâm số thực
            center_x = x1 + w / 2.0
            center_y = y1_new + h_new / 2.0

            # Tính góc nghiêng từ 2 mắt để xoay thẳng mặt (Chuẩn hóa hình học)
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]

            if left_eye[0] > right_eye[0]:
                left_eye, right_eye = right_eye, left_eye

            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = math.degrees(math.atan2(dY, dX))

            # Xoay ảnh về 0 độ bằng warpAffine
            M = cv2.getRotationMatrix2D((center_x, center_y), angle, 1.0)
            rotated_img = cv2.warpAffine(
                img, M, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR
            )

            # ÉP KHUNG VUÔNG (Đảm bảo tính đồng nhất tỷ lệ pixel, tránh méo hình học)
            pad_w = w * 0.15
            pad_h = h_new * 0.15
            new_w = w + 2 * pad_w
            new_h = h_new + 2 * pad_h
            square_size = int(max(new_w, new_h))

            # Trích xuất vùng mặt hình vuông bằng getRectSubPix
            face_img = cv2.getRectSubPix(
                rotated_img, (square_size, square_size), (center_x, center_y)
            )

            if face_img is None or face_img.size == 0:
                continue

            # XỬ LÝ THÍCH ỨNG THEO ĐỘ PHÂN GIẢI (Thuần túy xử lý tín hiệu ảnh)
            h_crop, w_crop = face_img.shape[:2]

            if h_crop < 100 or w_crop < 100:
                # ẢNH NHỎ/MỜ: Phóng to bằng Lanczos4 để cố định cấu trúc cạnh thô tốt nhất
                face_img_processed = cv2.resize(
                    face_img, (224, 224), interpolation=cv2.INTER_LANCZOS4
                )

                # Dùng CLAHE lưới nhỏ 4x4 để kích nét cục bộ mà không sinh nhiễu biên lớn
                ycrcb = cv2.cvtColor(face_img_processed, cv2.COLOR_BGR2YCrCb)
                clahe = cv2.createCLAHE(clipLimit=1.2, tileGridSize=(4, 4))
                ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
                final_face_img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
            else:
                # ẢNH LỚN/NÉT: Lọc giảm nhiễu hạt bề mặt da
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

                final_face_img = cv2.resize(
                    face_img_processed, (224, 224), interpolation=cv2.INTER_LANCZOS4
                )

            # Base64 encode
            _, buffer = cv2.imencode(".jpg", final_face_img)
            face_base64 = base64.b64encode(buffer).decode("utf-8")

            # 3. CLIVP-FER Extraction
            t3_start = time.time()
            feature_vector = self.clivp_model.extract(final_face_img).numpy()
            stage3_total_time += time.time() - t3_start

            # 4. Feature Selection (EPSO)
            t4_start = time.time()
            reduced_vector = feature_vector[self.epso_mask]
            stage4_total_time += time.time() - t4_start

            # 5. Emotion Classification (MLP)
            t5_start = time.time()
            with torch.no_grad():
                tensor_input = (
                    torch.tensor(reduced_vector, dtype=torch.float32)
                    .unsqueeze(0)
                    .to(self.device)
                )
                outputs = self.mlp(tensor_input)
                probs = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)

            latency_mlp = round((time.time() - t5_start) * 1000, 2)
            stage5_total_time += time.time() - t5_start

            # Trả về nhãn thuần túy từ mô hình học máy, không qua bộ lọc can thiệp nhãn
            emotion_str = self.idx_to_label[predicted.item()]
            stage4_total_time += latency_mlp / 1000.0

            # Tính toán phân phối của tất cả các cảm xúc
            probs_list = probs.squeeze(0).cpu().numpy().tolist()
            emotions_distribution = sorted(
                [
                    {"label": self.idx_to_label[i].capitalize(), "score": round(prob * 100, 1)}
                    for i, prob in enumerate(probs_list)
                ],
                key=lambda x: x["score"],
                reverse=True
            )

            # Đánh giá mức độ tập trung
            concentration_score = 0
            if emotion_str.lower() in ["happy", "neutral", "surprise"]:
                concentration_score = 70 + int(confidence.item() * 30)
            else:
                concentration_score = 20 + int(confidence.item() * 30)

            faces_list.append(
                {
                    "id": idx,
                    "box": [x1, y1, x2, y2],
                    "image": face_base64,
                    "emotion": emotion_str.capitalize(),
                    "confidence": round(confidence.item() * 100, 1),
                    "emotions": emotions_distribution,
                    "latency_ms": latency_mlp,
                    "concentration": concentration_score,
                }
            )

        if len(faces_list) > 0:
            avg_concentration = int(np.mean([f["concentration"] for f in faces_list]))
            emotions = [f["emotion"] for f in faces_list]
            dominant_emotion = max(set(emotions), key=emotions.count)
        else:
            avg_concentration = 0
            dominant_emotion = "No Face"

        logs.append(
            {
                "stage": "Feature Extraction (CLIVP-FER)",
                "latency_ms": round(stage3_total_time * 1000, 2),
                "message": f"Đã trích xuất đặc trưng cho {len(faces_list)} khuôn mặt",
            }
        )
        logs.append(
            {
                "stage": "Feature Selection (EPSO)",
                "latency_ms": round(stage4_total_time * 1000, 2),
                "message": f"Hoàn tất lọc đặc trưng (EPSO {len(self.epso_mask)} chiều)",
            }
        )
        logs.append(
            {
                "stage": "Emotion Classification (MLP)",
                "latency_ms": round(stage5_total_time * 1000, 2),
                "message": "Hoàn tất phân loại cảm xúc",
            }
        )

        return {
            "emotion": dominant_emotion,
            "concentration": avg_concentration,
            "faces": faces_list,
            "logs": logs,
        }
