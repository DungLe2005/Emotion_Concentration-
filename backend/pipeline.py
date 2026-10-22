import time
from typing import Dict, Any, List, Optional
import numpy as np
import cv2
import torch
import pandas as pd
import base64
import sys
import os
import math

# Tắt log telemetry ngầm MediaPipe clearcut uploader
os.environ["GLOG_minloglevel"] = "2"
os.environ["MEDIAPIPE_DISABLE_CLEARCUT"] = "1"
from retinaface import RetinaFace
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)
from dataset.preprocess_features import RealCLIVP_FER
from models.mlp import EmotionMLP
from features.landmark_features import LandmarkFeatureExtractor
from session_tracker import SessionEmotionTracker, calculate_concentration_score


class EmotionPipeline:
    """
    Class quản lý toàn bộ luồng xử lý AI trong thời gian thực.
    Được thiết kế để khởi tạo 1 lần (Singleton) nhằm tiết kiệm RAM và thời gian.
    """

    def __init__(self):
        print("\n[AI] Đang khởi tạo toàn bộ Pipeline Models...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        # Khởi tạo Session Emotion Tracker
        self.session_tracker = SessionEmotionTracker()

        # Khởi tạo MediaPipe Detector siêu tốc (12ms) cho phát hiện nhiều mặt
        task_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
        options_full = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=task_path),
            running_mode=RunningMode.IMAGE,
            num_faces=20,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
        )
        self.full_landmarker = FaceLandmarker.create_from_options(options_full)

        # 2. Khởi tạo CLIVP-FER
        self.clivp_model = RealCLIVP_FER(self.device)

        # 2.0 Khởi tạo CLIP Text Prompts cho 7 lớp cảm xúc
        import clip

        self.EMOTION_PROMPTS = [
            "a face with an angry expression, furrowed eyebrows and intense glare",
            "a face showing disgust, wrinkled nose and raised upper lip",
            "a fearful frightened face with wide eyes and open mouth",
            "a happy smiling cheerful face with open smile",
            "a neutral calm relaxed face with a plain expression",
            "a sad sorrowful face with downturned lips and gloomy eyes",
            "a shocked surprised face with raised eyebrows and gasping mouth",
        ]
        text_tokens = clip.tokenize(self.EMOTION_PROMPTS).to(self.device)
        with torch.no_grad():
            text_feats = self.clivp_model.clip_model.encode_text(text_tokens)
            # Ép kiểu rõ ràng sang torch.float (float32) để tránh xung đột dtype với Half (float16) của CLIP
            self.text_features = (
                text_feats / text_feats.norm(dim=-1, keepdim=True)
            ).float()  # (7, 512) float32

        # 2.1 Khởi tạo Landmark Feature Extractor (MediaPipe, CPU)
        self.landmark_extractor = LandmarkFeatureExtractor()
        self.last_valid_landmark = (
            None  # Lưu trữ cho Thuật toán Nội suy thời gian (Temporal Interpolation)
        )

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

        # 4. Tìm nhãn (labels mapping 7 lớp chuẩn từ new_data)
        emotions = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
        self.idx_to_label = {idx: label for idx, label in enumerate(emotions)}

        # 5. Load Soft-Voting Ensemble Head (ResMLP + XGBoost + MLP)
        from models.ensemble_head import EmotionEnsembleHead

        self.ensemble_head = EmotionEnsembleHead(
            input_dim=reduced_dim, num_classes=len(emotions), device=self.device
        )

        # 6. Khởi tạo Temporal Face Tracker (IoU Tracking + EMA Smoothing + Hysteresis Thresholding)
        from temporal_tracker import TemporalFaceTracker

        self.temporal_tracker = TemporalFaceTracker(
            idx_to_label=self.idx_to_label,
            iou_threshold=0.25,
            max_missed_frames=15,
            alpha=0.35,
            window_size=7,
            min_consecutive_to_switch=3,
            instant_switch_threshold=0.65,
            hysteresis_margin=0.12,
        )

        print(f"[AI] Đã nạp thành công Soft-Voting Ensemble Head (ResMLP + XGBoost + MLP) với đầu vào {len(self.epso_mask)} chiều.")
        print("[AI] Đã khởi tạo TemporalFaceTracker (IoU + EMA Smoothing + Hysteresis).")
        print("[AI] Pipeline đã sẵn sàng nhận ảnh!\n")

    def reset_temporal_tracker(self):
        """Reset trạng thái Face Tracker & bộ đệm Temporal Smoothing khi bắt đầu buổi học mới."""
        self.temporal_tracker.reset()

    def capture_negative_evidence_scene(
        self, full_img: np.ndarray, face_item: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Chụp lại toàn cảnh xung quanh học sinh khi cảm xúc tiêu cực xuất hiện liên tục >= 6 chu kỳ.
        Lưu giữ 100% không gian hiện trường lớp học xung quanh để làm bằng chứng khách quan,
        đồng thời làm nổi bật vị trí học sinh kèm thẻ thông tin và góc phóng to cận cảnh khuôn mặt.
        """
        evidence_dir = os.path.join(os.path.dirname(__file__), "reports", "evidence_images")
        os.makedirs(evidence_dir, exist_ok=True)

        now = time.time()
        timestamp_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(now))
        time_display = time.strftime("%H:%M:%S", time.localtime(now))

        start_ts = self.session_tracker.start_time if self.session_tracker.start_time else now
        elapsed_sec = max(0, int(now - start_ts))
        elapsed_fmt = f"{elapsed_sec // 60}m {elapsed_sec % 60}s"

        track_id = face_item.get("track_id", 1)
        emotion = face_item.get("emotion", "Sad")
        conf = face_item.get("confidence", 0.0)
        cycles = face_item.get("consecutive_negative_cycles", 6)
        is_speaking = face_item.get("is_speaking", False)
        box = face_item.get("box", [0, 0, 100, 100])

        x1, y1, x2, y2 = [int(b) for b in box]
        h_img, w_img = full_img.shape[:2]

        # Bản sao toàn cảnh hiện trường
        evidence_img = full_img.copy()

        # 1. Vẽ Bounding Box quanh học sinh bị gắn cờ tiêu cực (màu đỏ cam nổi bật)
        cv2.rectangle(evidence_img, (x1, y1), (x2, y2), (0, 69, 255), 3)

        # 2. Vẽ bảng nhãn thông tin phía trên bounding box
        label_text = f"Hoc sinh #{track_id} | {emotion} ({conf}%) | {cycles} chu ky tieu cuc"
        (lbl_w, lbl_h), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        lbl_y1 = max(0, y1 - lbl_h - 10)
        lbl_y2 = y1
        cv2.rectangle(evidence_img, (x1, lbl_y1), (min(w_img, x1 + lbl_w + 10), lbl_y2), (0, 69, 255), -1)
        cv2.putText(
            evidence_img,
            label_text,
            (x1 + 5, max(lbl_h + 2, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # 3. Chèn góc phóng to cận cảnh khuôn mặt (Picture-in-Picture) ở góc trên
        crop_y1 = max(0, y1)
        crop_y2 = min(h_img, y2)
        crop_x1 = max(0, x1)
        crop_x2 = min(w_img, x2)
        if (crop_y2 - crop_y1) > 10 and (crop_x2 - crop_x1) > 10:
            try:
                face_crop = full_img[crop_y1:crop_y2, crop_x1:crop_x2]
                pip_size = 140
                face_pip = cv2.resize(face_crop, (pip_size, pip_size), interpolation=cv2.INTER_LINEAR)

                # Đặt ở góc trên bên phải (hoặc bên trái nếu mặt học sinh ở góc phải)
                pip_x1 = w_img - pip_size - 15 if (x2 < w_img - 180) else 15
                pip_y1 = 15
                pip_x2 = pip_x1 + pip_size
                pip_y2 = pip_y1 + pip_size

                # Gán vào ảnh toàn cảnh
                evidence_img[pip_y1:pip_y2, pip_x1:pip_x2] = face_pip
                cv2.rectangle(evidence_img, (pip_x1, pip_y1), (pip_x2, pip_y2), (0, 69, 255), 2)
                cv2.putText(
                    evidence_img,
                    f"Can canh #{track_id}",
                    (pip_x1 + 4, pip_y1 + 16),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
            except Exception as pip_err:
                print(f"[Pipeline Pip Warning] {pip_err}")

        filename = f"evidence_student_{track_id}_{timestamp_str}.jpg"
        filepath = os.path.join(evidence_dir, filename)

        cv2.imwrite(filepath, evidence_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

        from session_tracker import EMOTION_LABELS_VI
        emotion_key = emotion.lower()
        emotion_vi = EMOTION_LABELS_VI.get(emotion_key, emotion)

        return {
            "track_id": track_id,
            "student_label": f"Học sinh #{track_id}",
            "timestamp": time_display,
            "elapsed_seconds": elapsed_sec,
            "elapsed_formatted": elapsed_fmt,
            "emotion": emotion,
            "emotion_vi": emotion_vi,
            "confidence": conf,
            "consecutive_cycles": cycles,
            "is_speaking": is_speaking,
            "image_filename": filename,
            "image_path": filepath,
            "context_note": "Cần đối chiếu toàn cảnh hiện trường: học sinh mệt mỏi/áp lực nội dung hoặc bị xao nhãng bởi môi trường xung quanh.",
        }

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

        # 2. Face Detection với RetinaFace làm bộ phát hiện CHÍNH (SOTA Accuracy & Robustness)
        stage2_start = time.time()
        faces_data = []

        h_orig, w_orig = img.shape[:2]
        target_w = 640
        if w_orig > target_w:
            scale_ratio = target_w / float(w_orig)
            target_h = max(1, int(h_orig * scale_ratio))
            img_det = cv2.resize(
                img, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
            scale_x = w_orig / float(target_w)
            scale_y = h_orig / float(target_h)
        else:
            img_det = img
            scale_x = 1.0
            scale_y = 1.0

        try:
            detections = RetinaFace.detect_faces(
                img_det, model=self.retinaface_model
            )
            if isinstance(detections, dict):
                for key in detections:
                    face = detections[key]
                    if "facial_area" in face and "landmarks" in face:
                        box = face["facial_area"]
                        landmarks = face["landmarks"]
                        orig_box = [
                            int(box[0] * scale_x),
                            int(box[1] * scale_y),
                            int(box[2] * scale_x),
                            int(box[3] * scale_y),
                        ]
                        orig_landmarks = {}
                        for lm_k, lm_v in landmarks.items():
                            orig_landmarks[lm_k] = [
                                lm_v[0] * scale_x,
                                lm_v[1] * scale_y,
                            ]
                        w_b = orig_box[2] - orig_box[0]
                        h_b = orig_box[3] - orig_box[1]
                        if w_b >= 30 and h_b >= 30:
                            faces_data.append(
                                {
                                    "facial_area": orig_box,
                                    "landmarks": orig_landmarks,
                                }
                            )
        except Exception as e:
            print(f"Lỗi RetinaFace: {e}")

        # Dự phòng MediaPipe nếu RetinaFace không bắt được mặt nào
        if len(faces_data) == 0:
            try:
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(
                    image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb_img)
                )
                det_result = self.full_landmarker.detect(mp_img)
                if det_result.face_landmarks and len(det_result.face_landmarks) > 0:
                    for face_lms in det_result.face_landmarks:
                        xs = [lm.x * w_orig for lm in face_lms]
                        ys = [lm.y * h_orig for lm in face_lms]
                        x1 = max(0, int(min(xs)))
                        y1 = max(0, int(min(ys)))
                        x2 = min(w_orig, int(max(xs)))
                        y2 = min(h_orig, int(max(ys)))
                        if (x2 - x1) >= 30 and (y2 - y1) >= 30:
                            left_eye = [face_lms[33].x * w_orig, face_lms[33].y * h_orig]
                            right_eye = [face_lms[263].x * w_orig, face_lms[263].y * h_orig]
                            faces_data.append(
                                {
                                    "facial_area": [x1, y1, x2, y2],
                                    "landmarks": {"left_eye": left_eye, "right_eye": right_eye},
                                }
                            )
            except Exception as ex:
                print(f"[MediaPipe Fallback Detection Error] {ex}")

        if len(faces_data) == 0:
            stage2_time = time.time()
            logs.append(
                {
                    "stage": "Face Detection (RetinaFace)",
                    "latency_ms": round((stage2_time - stage2_start) * 1000, 2),
                    "message": "Không tìm thấy khuôn mặt",
                }
            )
            # Cập nhật Temporal Tracker khi không có khuôn mặt (tăng missed_frames)
            self.temporal_tracker.process_frame_faces([])

            # Cập nhật tracker với danh sách mặt rỗng (tự động đếm rỗng và end session nếu quá giới hạn)
            tracker_status = self.session_tracker.update([])
            return {
                "emotion": "No Face",
                "concentration": 0,
                "faces": [],
                "logs": logs,
                "session": {
                    "is_active": self.session_tracker.is_active,
                    "total_faces": self.session_tracker.total_faces,
                    "current_percent": self.session_tracker.get_current_percent(),
                    "current_score": self.session_tracker.compute_score()["score"],
                    "tracker_status": tracker_status,
                },
            }

        stage2_time = time.time()
        logs.append(
            {
                "stage": "Face Detection (RetinaFace)",
                "latency_ms": round((stage2_time - stage2_start) * 1000, 2),
                "message": f"Tìm thấy {len(faces_data)} khuôn mặt",
            }
        )

        candidate_detections = []
        stage3_total_time = 0
        stage4_total_time = 0
        stage5_total_time = 0

        for idx, face_info in enumerate(faces_data):
            box = face_info["facial_area"]
            landmarks = face_info["landmarks"]
            x1, y1, x2, y2 = [int(b) for b in box]

            w = x2 - x1
            h = y2 - y1

            # Tọa độ tâm khuôn mặt chuẩn RetinaFace
            center_x = (x1 + x2) / 2.0
            center_y = (y1 + y2) / 2.0

            # Tính góc nghiêng từ 2 mắt để xoay thẳng mặt (Chuẩn hóa hình học)
            left_eye = landmarks["left_eye"]
            right_eye = landmarks["right_eye"]

            if left_eye[0] > right_eye[0]:
                left_eye, right_eye = right_eye, left_eye

            dY = right_eye[1] - left_eye[1]
            dX = right_eye[0] - left_eye[0]
            angle = math.degrees(math.atan2(dY, dX))

            # ÉP KHUNG VUÔNG VỪA KHÍT (Padding 5% - Khớp 100% tỷ lệ crop mặt của tập KMU-FED)
            pad = max(w, h) * 0.05
            square_size = int(max(w, h) + 2 * pad)

            # Tối ưu hóa siêu tốc độ: Cắt sub-patch cục bộ thay vì xoay toàn bộ ảnh phân giải cao
            if abs(angle) < 3.0:
                # Góc nghiêng nhỏ (< 3 độ): Cắt trực tiếp cực nhanh không qua warpAffine
                face_img = cv2.getRectSubPix(
                    img, (square_size, square_size), (center_x, center_y)
                )
            else:
                # Cắt patch nhỏ quanh tâm trước rồi mới xoay (Nhanh gấp 50 lần xoay toàn ảnh)
                margin = int(square_size * 1.4)
                x_start = max(0, int(center_x - margin / 2))
                y_start = max(0, int(center_y - margin / 2))
                x_end = min(img.shape[1], int(center_x + margin / 2))
                y_end = min(img.shape[0], int(center_y + margin / 2))

                patch = img[y_start:y_end, x_start:x_end]
                if patch is not None and patch.size > 0:
                    patch_center = (center_x - x_start, center_y - y_start)
                    M_patch = cv2.getRotationMatrix2D(patch_center, angle, 1.0)
                    rotated_patch = cv2.warpAffine(
                        patch,
                        M_patch,
                        (patch.shape[1], patch.shape[0]),
                        flags=cv2.INTER_LINEAR,
                    )
                    face_img = cv2.getRectSubPix(
                        rotated_patch, (square_size, square_size), patch_center
                    )
                else:
                    face_img = cv2.getRectSubPix(
                        img, (square_size, square_size), (center_x, center_y)
                    )

            if face_img is None or face_img.size == 0:
                continue

            # Chuẩn hóa 224x224 siêu tốc bằng INTER_LINEAR
            final_face_img = cv2.resize(
                face_img, (224, 224), interpolation=cv2.INTER_LINEAR
            )

            # Base64 encode siêu tốc với JPEG Quality 80
            _, buffer = cv2.imencode(
                ".jpg", final_face_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
            )
            face_base64 = base64.b64encode(buffer).decode("utf-8")

            # 3. CLIVP-FER Extraction (Image 512 + Prompt 7 + Landmark 25 = 544)
            t3_start = time.time()
            clip_tensor = self.clivp_model.extract(final_face_img)  # (512,)
            clip_norm = clip_tensor / clip_tensor.norm(dim=-1, keepdim=True)
            clip_vector = clip_norm.numpy()

            # Cosine similarity với 7 Prompt Text (7,) - Đảm bảo đồng bộ kiểu dữ liệu float32
            with torch.no_grad():
                clip_tensor_gpu = clip_norm.to(self.device).float()
                text_feats_gpu = self.text_features.to(self.device).float()
                prompt_sims = (
                    (clip_tensor_gpu.unsqueeze(0) @ text_feats_gpu.T)
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )

            # Trích xuất đặc trưng Landmark với Thuật toán Nội suy Thời gian (Temporal Interpolation)
            raw_landmark = self.landmark_extractor.extract(final_face_img)  # (25,)
            if np.any(raw_landmark != 0):
                landmark_vector = raw_landmark
                self.last_valid_landmark = raw_landmark  # Lưu vết làm cơ sở nội suy
            elif self.last_valid_landmark is not None:
                # NỘI SUY THỜI GIAN: Dùng đặc trưng của frame hợp lệ gần nhất (không dùng Mean giả)
                landmark_vector = self.last_valid_landmark
            else:
                landmark_vector = raw_landmark

            # Vector tổng hợp 544 chiều
            feature_vector = np.concatenate(
                [clip_vector, prompt_sims, landmark_vector]
            )  # (544,)
            stage3_total_time += time.time() - t3_start

            # 4. Feature Selection (EPSO)
            t4_start = time.time()
            reduced_vector = feature_vector[self.epso_mask]
            stage4_total_time += time.time() - t4_start

            # 5. Emotion Classification (Soft-Voting Ensemble Head)
            t5_start = time.time()
            probs_arr = self.ensemble_head.predict_probs(reduced_vector)

            latency_mlp = round((time.time() - t5_start) * 1000, 2)
            stage5_total_time += time.time() - t5_start
            stage4_total_time += latency_mlp / 1000.0

            candidate_detections.append(
                {
                    "box": [x1, y1, x2, y2],
                    "raw_probs": probs_arr[0],
                    "landmarks": landmark_vector,
                    "embedding": clip_vector,
                    "image": face_base64,
                    "latency_ms": latency_mlp,
                }
            )

        # 6. Temporal Face Tracking + EMA Smoothing + Upper-Face Gating + Lip Dynamics
        tracked_faces = self.temporal_tracker.process_frame_faces(candidate_detections)

        faces_list = []
        for item in tracked_faces:
            prob_dict = {
                p_item["label"].lower(): p_item["score"] / 100.0
                for p_item in item["emotions"]
            }
            concentration_score = calculate_concentration_score(
                prob_dict, self.session_tracker.weights
            )

            faces_list.append(
                {
                    "id": item["track_id"],
                    "track_id": item["track_id"],
                    "box": item["box"],
                    "image": item["image"],
                    "emotion": item["emotion"],
                    "confidence": item["confidence"],
                    "emotions": item["emotions"],
                    "latency_ms": item["latency_ms"],
                    "concentration": concentration_score,
                    "is_speaking": item.get("is_speaking", False),
                    "consecutive_negative_cycles": item.get("consecutive_negative_cycles", 0),
                    "is_negative_flagged": item.get("is_negative_flagged", False),
                }
            )

            # BẰNG CHỨNG KHÁCH QUAN: Tự động chụp lại toàn cảnh xung quanh học sinh khi cảm xúc tiêu cực kéo dài >= 6 chu kỳ
            if item.get("trigger_negative_evidence", False) and self.session_tracker.is_active:
                try:
                    evidence_record = self.capture_negative_evidence_scene(img, item)
                    self.session_tracker.record_negative_evidence(evidence_record)
                except Exception as ev_err:
                    print(f"[Pipeline Warning] Lỗi chụp lưu vết bằng chứng tiêu cực: {ev_err}")

        # Thu thập xác suất từng mặt để đưa vào session tracker
        frame_face_probs = []
        for face_item in faces_list:
            if "emotions" in face_item:
                prob_dict = {
                    item["label"].lower(): item["score"] / 100.0
                    for item in face_item["emotions"]
                }
                frame_face_probs.append(prob_dict)

        tracker_status = self.session_tracker.update(frame_face_probs)

        if len(faces_list) > 0:
            avg_concentration = int(np.mean([f["concentration"] for f in faces_list]))
            emotions = [f["emotion"] for f in faces_list]
            dominant_emotion = max(set(emotions), key=emotions.count)
        else:
            avg_concentration = 0
            dominant_emotion = "No Face"

        logs.append(
            {
                "stage": "Feature Extraction (544D Multimodal)",
                "latency_ms": round(stage3_total_time * 1000, 2),
                "message": f"Đã trích xuất 544D (CLIP + Prompt + MediaPipe AU) cho {len(faces_list)} khuôn mặt",
            }
        )
        logs.append(
            {
                "stage": "Feature Selection (EPSO 289D)",
                "latency_ms": round(stage4_total_time * 1000, 2),
                "message": f"Hoàn tất lọc đặc trưng (EPSO {len(self.epso_mask)} chiều)",
            }
        )
        logs.append(
            {
                "stage": "Emotion Classification (Ensemble Head)",
                "latency_ms": round(stage5_total_time * 1000, 2),
                "message": "Hoàn tất phân loại bằng Ensemble Classifier (ResMLP + XGBoost)",
            }
        )

        current_score_data = self.session_tracker.compute_score()

        return {
            "emotion": dominant_emotion,
            "concentration": avg_concentration,
            "faces": faces_list,
            "logs": logs,
            "session": {
                "is_active": self.session_tracker.is_active,
                "total_faces": self.session_tracker.total_faces,
                "current_percent": self.session_tracker.get_current_percent(),
                "current_score": current_score_data["score"],
                "raw_score": current_score_data["raw_score"],
                "history_length": len(self.session_tracker.history),
                "tracker_status": tracker_status,
            },
        }

    def run_single_frame(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Phương thức xử lý dedicated riêng cho tab 'Phân Tích Khung Hình Đơn'.
        Không ảnh hưởng tới Session Tracker tích lũy buổi học.
        """
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

        # 2. Face Detection với RetinaFace SOTA (Đảm bảo bắt đầy đủ mọi góc quay và khoảng cách xa)
        stage2_start = time.time()
        faces_data = []
        detector_name = "RetinaFace SOTA"

        h_orig, w_orig = img.shape[:2]
        target_w = 640
        if w_orig > target_w:
            scale_ratio = target_w / float(w_orig)
            target_h = max(1, int(h_orig * scale_ratio))
            img_det = cv2.resize(
                img, (target_w, target_h), interpolation=cv2.INTER_AREA
            )
            scale_x = w_orig / float(target_w)
            scale_y = h_orig / float(target_h)
        else:
            img_det = img
            scale_x = 1.0
            scale_y = 1.0

        try:
            detections = RetinaFace.detect_faces(
                img_det, model=self.retinaface_model
            )
            if isinstance(detections, dict):
                for key in detections:
                    face = detections[key]
                    if "facial_area" in face and "landmarks" in face:
                        box = face["facial_area"]
                        landmarks = face["landmarks"]
                        orig_box = [
                            int(box[0] * scale_x),
                            int(box[1] * scale_y),
                            int(box[2] * scale_x),
                            int(box[3] * scale_y),
                        ]
                        orig_landmarks = {}
                        for lm_k, lm_v in landmarks.items():
                            orig_landmarks[lm_k] = [
                                lm_v[0] * scale_x,
                                lm_v[1] * scale_y,
                            ]
                        faces_data.append(
                            {
                                "facial_area": orig_box,
                                "landmarks": orig_landmarks,
                            }
                        )
        except Exception as e:
            print(f"[Pipeline Error] Lỗi RetinaFace detect: {e}")

        stage2_time = time.time()
        det_ms = round((stage2_time - stage2_start) * 1000, 2)
        print(
            f"[Pipeline Debug] 🔍 Bước 2 - Face Detection ({detector_name}): Tìm thấy {len(faces_data)} khuôn mặt ({det_ms} ms)"
        )

        logs.append(
            {
                "stage": "Face Detection (RetinaFace)",
                "latency_ms": det_ms,
                "message": f"Tìm thấy {len(faces_data)} khuôn mặt qua {detector_name}",
            }
        )

        if len(faces_data) == 0:
            logs.append(
                {
                    "stage": "Feature Extraction (544D Multimodal)",
                    "latency_ms": 0,
                    "message": "Bỏ qua (Không phát hiện thấy khuôn mặt trong ảnh)",
                }
            )
            logs.append(
                {
                    "stage": "Feature Selection (EPSO 289D)",
                    "latency_ms": 0,
                    "message": "Bỏ qua (Không phát hiện thấy khuôn mặt trong ảnh)",
                }
            )
            logs.append(
                {
                    "stage": "Emotion Classification (Ensemble Head)",
                    "latency_ms": 0,
                    "message": "Bỏ qua (Không phát hiện thấy khuôn mặt trong ảnh)",
                }
            )
            print("[Pipeline Debug] ⚠️ KHÔNG TÌM THẤY KHUÔN MẶT NÀO TRONG ẢNH")
            return {"emotion": "No Face", "concentration": 0, "faces": [], "logs": logs}

        faces_list = []
        stage3_total_time = 0
        stage4_total_time = 0
        stage5_total_time = 0

        for idx, face_info in enumerate(faces_data):
            try:
                box = face_info["facial_area"]
                landmarks = face_info["landmarks"]
                x1, y1, x2, y2 = box

                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                center_x = (x1 + x2) / 2.0
                center_y = (y1 + y2) / 2.0

                left_eye = landmarks["left_eye"]
                right_eye = landmarks["right_eye"]
                if left_eye[0] > right_eye[0]:
                    left_eye, right_eye = right_eye, left_eye

                dY = right_eye[1] - left_eye[1]
                dX = right_eye[0] - left_eye[0]
                angle = math.degrees(math.atan2(dY, dX))

                pad = max(w, h) * 0.05
                square_size = int(max(w, h) + 2 * pad)

                print(
                    f"[Pipeline Debug] ✂️ Cắt mặt #{idx+1}: Box=[{x1},{y1},{x2},{y2}], Kích thước={w}x{h}px, Góc xoay={angle:.1f}°"
                )

                if abs(angle) < 3.0:
                    face_img = cv2.getRectSubPix(
                        img, (square_size, square_size), (center_x, center_y)
                    )
                else:
                    margin = int(square_size * 1.4)
                    x_start = max(0, int(center_x - margin / 2))
                    y_start = max(0, int(center_y - margin / 2))
                    x_end = min(img.shape[1], int(center_x + margin / 2))
                    y_end = min(img.shape[0], int(center_y + margin / 2))

                    patch = img[y_start:y_end, x_start:x_end]
                    if patch is not None and patch.size > 0:
                        patch_center = (center_x - x_start, center_y - y_start)
                        M_patch = cv2.getRotationMatrix2D(patch_center, angle, 1.0)
                        rotated_patch = cv2.warpAffine(
                            patch,
                            M_patch,
                            (patch.shape[1], patch.shape[0]),
                            flags=cv2.INTER_LINEAR,
                        )
                        face_img = cv2.getRectSubPix(
                            rotated_patch, (square_size, square_size), patch_center
                        )
                    else:
                        face_img = cv2.getRectSubPix(
                            img, (square_size, square_size), (center_x, center_y)
                        )

                if face_img is None or face_img.size == 0:
                    continue

                final_face_img = cv2.resize(
                    face_img, (224, 224), interpolation=cv2.INTER_LINEAR
                )
                _, buffer = cv2.imencode(
                    ".jpg", final_face_img, [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                )
                face_base64 = base64.b64encode(buffer).decode("utf-8")

                # 3. CLIVP-FER Extraction
                t3_start = time.time()
                clip_tensor = self.clivp_model.extract(final_face_img)
                clip_norm = clip_tensor / clip_tensor.norm(dim=-1, keepdim=True)
                clip_vector = clip_norm.numpy()

                with torch.no_grad():
                    clip_tensor_gpu = clip_norm.to(self.device).float()
                    text_feats_gpu = self.text_features.to(self.device).float()
                    prompt_sims = (
                        (clip_tensor_gpu.unsqueeze(0) @ text_feats_gpu.T)
                        .squeeze(0)
                        .cpu()
                        .numpy()
                    )

                raw_landmark = self.landmark_extractor.extract(final_face_img)
                landmark_vector = raw_landmark

                feature_vector = np.concatenate([clip_vector, prompt_sims, landmark_vector])
                stage3_total_time += time.time() - t3_start

                # 4. Feature Selection (EPSO)
                t4_start = time.time()
                reduced_vector = feature_vector[self.epso_mask]
                stage4_total_time += time.time() - t4_start

                # 5. Emotion Classification (Soft-Voting Ensemble Head)
                t5_start = time.time()
                probs_arr = self.ensemble_head.predict_probs(reduced_vector)

                # Áp dụng Upper-Face Gating cho ảnh đơn (tránh nhầm há miệng nói chuyện thành Surprise)
                from temporal_tracker import apply_upper_face_gating
                gated_probs_arr, _ = apply_upper_face_gating(probs_arr[0], landmark_vector, self.idx_to_label)
                probs = torch.tensor([gated_probs_arr])
                confidence, predicted = torch.max(probs, 1)

                latency_mlp = round((time.time() - t5_start) * 1000, 2)
                stage5_total_time += time.time() - t5_start

                emotion_str = self.idx_to_label[predicted.item()]
                probs_list = probs.squeeze(0).cpu().numpy().tolist()
                emotions_distribution = sorted(
                    [
                        {
                            "label": self.idx_to_label[i].capitalize(),
                            "score": round(prob * 100, 1),
                        }
                        for i, prob in enumerate(probs_list)
                    ],
                    key=lambda x: x["score"],
                    reverse=True,
                )

                # Đánh giá mức độ tập trung chính xác bằng công thức Softmax trọng số thống nhất
                prob_dict = {
                    self.idx_to_label[i].lower(): prob for i, prob in enumerate(probs_list)
                }
                concentration_score = calculate_concentration_score(
                    prob_dict, self.session_tracker.weights
                )

                print(
                    f"[Pipeline Debug] 🎯 Mặt #{idx+1} -> Cảm xúc: {emotion_str.upper()} ({confidence.item()*100:.1f}%), Điểm tập trung: {concentration_score}%"
                )

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
            except Exception as face_err:
                print(f"[Pipeline Warning] Error in run_single_frame face #{idx}: {face_err}")
                continue

        if len(faces_list) > 0:
            avg_concentration = int(np.mean([f["concentration"] for f in faces_list]))
            emotions = [f["emotion"] for f in faces_list]
            dominant_emotion = max(set(emotions), key=emotions.count)
        else:
            avg_concentration = 0
            dominant_emotion = "No Face"

        logs.append(
            {
                "stage": "Feature Extraction (544D Multimodal)",
                "latency_ms": round(stage3_total_time * 1000, 2),
                "message": f"Đã trích xuất 544D (CLIP + Prompt + MediaPipe AU) cho {len(faces_list)} khuôn mặt",
            }
        )
        logs.append(
            {
                "stage": "Feature Selection (EPSO 289D)",
                "latency_ms": round(stage4_total_time * 1000, 2),
                "message": f"Hoàn tất lọc đặc trưng (EPSO {len(self.epso_mask)} chiều)",
            }
        )
        logs.append(
            {
                "stage": "Emotion Classification (Ensemble Head)",
                "latency_ms": round(stage5_total_time * 1000, 2),
                "message": "Hoàn tất phân loại bằng Ensemble Classifier (ResMLP + XGBoost)",
            }
        )

        total_ms = round((time.time() - start_time) * 1000, 2)
        print(
            f"[Pipeline Debug] 🏁 HOÀN TẤT PHÂN TÍCH ({len(faces_list)} mặt) | Cảm xúc chính = {dominant_emotion.upper()} ({avg_concentration}%) | Tổng thời gian = {total_ms} ms\n"
        )

        return {
            "emotion": dominant_emotion,
            "concentration": avg_concentration,
            "faces": faces_list,
            "logs": logs,
        }
