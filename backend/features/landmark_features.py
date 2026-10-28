"""
Trích xuất 25 đặc trưng hình học AU/Landmark từ MediaPipe FaceLandmarker (Tasks API).

Mục đích: Bổ sung tín hiệu hình học tường minh (Action Unit proxies) mà CLIP thiếu,
đặc biệt để phân biệt Fear↔Surprise, Disgust↔Neutral, Angry↔Fear.

Tất cả khoảng cách được chuẩn hóa theo chiều rộng khuôn mặt (face width)
để bất biến tỷ lệ (scale-invariant).
"""

import sys
import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

# Thêm sys.path gốc backend
BACKEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BACKEND_DIR not in sys.path:
    sys.path.insert(0, BACKEND_DIR)

# Tổng số đặc trưng hình học
NUM_LANDMARK_FEATURES = 25

# Đường dẫn tới file model .task
_MODEL_PATH_ROOT = os.path.join(BACKEND_DIR, "face_landmarker.task")
_MODEL_PATH_LOCAL = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
_MODEL_PATH = (
    _MODEL_PATH_ROOT if os.path.exists(_MODEL_PATH_ROOT) else _MODEL_PATH_LOCAL
)


class LandmarkFeatureExtractor:
    """
    Trích xuất 25 đặc trưng hình học từ ảnh khuôn mặt BGR.
    Sử dụng MediaPipe FaceLandmarker Tasks API (mediapipe >= 0.10.x).
    Khởi tạo 1 lần (singleton), gọi extract() cho mỗi frame.
    """

    # Miệng ngoài (outer lips)
    MOUTH_LEFT = 61
    MOUTH_RIGHT = 291
    MOUTH_TOP = 13
    MOUTH_BOTTOM = 14

    # Miệng trong (inner lips)
    INNER_MOUTH_TOP = 82
    INNER_MOUTH_BOTTOM = 87
    INNER_MOUTH_LEFT = 78
    INNER_MOUTH_RIGHT = 308

    # Môi trên
    UPPER_LIP_TOP = 0

    # Mũi
    NOSE_TIP = 1
    NOSE_BRIDGE = 6
    NOSE_BOTTOM = 2

    # Mắt trái
    LEFT_EYE_INNER = 133
    LEFT_EYE_OUTER = 33
    LEFT_EYE_TOP = 159
    LEFT_EYE_BOTTOM = 145

    # Mắt phải
    RIGHT_EYE_INNER = 362
    RIGHT_EYE_OUTER = 263
    RIGHT_EYE_TOP = 386
    RIGHT_EYE_BOTTOM = 374

    # Lông mày trái
    LEFT_BROW_INNER = 107
    LEFT_BROW_OUTER = 70
    LEFT_BROW_MID = 105

    # Lông mày phải
    RIGHT_BROW_INNER = 336
    RIGHT_BROW_OUTER = 300
    RIGHT_BROW_MID = 334

    # Viền mặt
    FACE_LEFT = 234
    FACE_RIGHT = 454
    FACE_TOP = 10
    CHIN = 152

    # Vùng mũi (wrinkle proxy)
    NOSE_LEFT_SIDE = 49
    NOSE_RIGHT_SIDE = 279

    # Nasolabial fold proxies
    NASOLABIAL_LEFT = 205
    NASOLABIAL_RIGHT = 425

    def __init__(self, model_path=None):
        """Khởi tạo MediaPipe FaceLandmarker (IMAGE mode cho preprocessing)."""
        if model_path is None:
            model_path = _MODEL_PATH
        self._model_path = model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"[Landmark] Không tìm thấy model: {model_path}. "
                "Tải về từ: https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            )

        self._create_landmarker()
        self._call_count = 0
        self._reinit_every = 2000  # Khởi tạo lại sau mỗi 2000 ảnh để tránh WinError 6

    def _create_landmarker(self):
        """Tạo mới instance FaceLandmarker (dùng khi khởi tạo hoặc tái khởi tạo)."""
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=self._model_path),
            running_mode=RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.3,
            min_face_presence_confidence=0.3,
            min_tracking_confidence=0.3,
        )
        self.landmarker = FaceLandmarker.create_from_options(options)
        print("[Landmark] MediaPipe FaceLandmarker đã khởi tạo (IMAGE mode).")

    def _maybe_reinitialize(self):
        """Tái khởi tạo FaceLandmarker định kỳ để tránh lỗi [WinError 6] trên Windows."""
        self._call_count += 1
        if self._call_count >= self._reinit_every:
            self._call_count = 0
            try:
                self.landmarker.close()
            except Exception:
                pass
            self._create_landmarker()

    def _get_point(self, landmarks, idx, h, w):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    def _distance(self, p1, p2):
        return np.linalg.norm(p1 - p2)

    def _angle(self, p1, p2):
        delta = p2 - p1
        return np.arctan2(delta[1], delta[0])

    def extract(self, face_img_bgr):
        if face_img_bgr is None or face_img_bgr.size == 0:
            return np.zeros(NUM_LANDMARK_FEATURES, dtype=np.float32)

        self._maybe_reinitialize()

        h, w = face_img_bgr.shape[:2]
        rgb = cv2.cvtColor(face_img_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        try:
            result = self.landmarker.detect(mp_image)
        except Exception as e:
            # Bắt toàn bộ lỗi (WinError 6 / RuntimeError / MediaPipe C++ exception)
            try:
                self.landmarker.close()
            except Exception:
                pass
            try:
                self._create_landmarker()
                self._call_count = 0
                result = self.landmarker.detect(mp_image)
            except Exception:
                # Trả về 0 an toàn — Pipeline sẽ tự động dùng Temporal Interpolation (last_valid_landmark)
                return np.zeros(NUM_LANDMARK_FEATURES, dtype=np.float32)

        if not result.face_landmarks or len(result.face_landmarks) == 0:
            return np.zeros(NUM_LANDMARK_FEATURES, dtype=np.float32)

        lm_list = result.face_landmarks[0]
        p = lambda idx: self._get_point(lm_list, idx, h, w)

        face_width = self._distance(p(self.FACE_LEFT), p(self.FACE_RIGHT))
        if face_width < 1e-6:
            return np.zeros(NUM_LANDMARK_FEATURES, dtype=np.float32)

        nd = lambda a, b: self._distance(p(a), p(b)) / face_width
        features = np.zeros(NUM_LANDMARK_FEATURES, dtype=np.float32)

        mouth_width = self._distance(p(self.MOUTH_LEFT), p(self.MOUTH_RIGHT))
        mouth_height = self._distance(p(self.MOUTH_TOP), p(self.MOUTH_BOTTOM))
        inner_mouth_height = self._distance(
            p(self.INNER_MOUTH_TOP), p(self.INNER_MOUTH_BOTTOM)
        )

        features[0] = mouth_height / max(mouth_width, 1e-6)
        features[1] = inner_mouth_height / face_width

        mouth_center = (p(self.MOUTH_LEFT) + p(self.MOUTH_RIGHT)) / 2
        features[2] = self._angle(mouth_center, p(self.MOUTH_LEFT))
        features[3] = self._angle(mouth_center, p(self.MOUTH_RIGHT))

        features[4] = nd(self.NOSE_BOTTOM, self.MOUTH_TOP)
        features[5] = inner_mouth_height / max(mouth_height, 1e-6)
        features[6] = mouth_width / face_width
        features[7] = abs(features[2] - features[3])

        left_eye_h = self._distance(p(self.LEFT_EYE_TOP), p(self.LEFT_EYE_BOTTOM))
        left_eye_w = self._distance(p(self.LEFT_EYE_INNER), p(self.LEFT_EYE_OUTER))
        features[8] = left_eye_h / max(left_eye_w, 1e-6)

        right_eye_h = self._distance(p(self.RIGHT_EYE_TOP), p(self.RIGHT_EYE_BOTTOM))
        right_eye_w = self._distance(p(self.RIGHT_EYE_INNER), p(self.RIGHT_EYE_OUTER))
        features[9] = right_eye_h / max(right_eye_w, 1e-6)

        features[10] = abs(features[8] - features[9])
        features[11] = nd(self.LEFT_EYE_TOP, self.LEFT_BROW_MID)
        features[12] = nd(self.RIGHT_EYE_TOP, self.RIGHT_BROW_MID)

        avg_ear = (features[8] + features[9]) / 2
        features[13] = avg_ear

        features[14] = nd(self.LEFT_BROW_INNER, self.RIGHT_BROW_INNER)

        left_eye_center = (p(self.LEFT_EYE_TOP) + p(self.LEFT_EYE_BOTTOM)) / 2
        features[15] = (
            self._distance(left_eye_center, p(self.LEFT_BROW_MID)) / face_width
        )

        right_eye_center = (p(self.RIGHT_EYE_TOP) + p(self.RIGHT_EYE_BOTTOM)) / 2
        features[16] = (
            self._distance(right_eye_center, p(self.RIGHT_BROW_MID)) / face_width
        )

        features[17] = self._angle(p(self.LEFT_BROW_INNER), p(self.LEFT_BROW_OUTER))
        features[18] = self._angle(p(self.RIGHT_BROW_INNER), p(self.RIGHT_BROW_OUTER))

        features[19] = nd(self.NOSE_LEFT_SIDE, self.NOSE_RIGHT_SIDE)
        nasolabial_left_dist = nd(self.NOSE_TIP, self.NASOLABIAL_LEFT)
        nasolabial_right_dist = nd(self.NOSE_TIP, self.NASOLABIAL_RIGHT)
        features[20] = (nasolabial_left_dist + nasolabial_right_dist) / 2

        face_height = self._distance(p(self.FACE_TOP), p(self.CHIN))
        features[21] = face_width / max(face_height, 1e-6)
        features[22] = nd(self.NOSE_TIP, self.CHIN)

        nose_to_chin = self._distance(p(self.NOSE_TIP), p(self.CHIN))
        nose_to_forehead = self._distance(p(self.NOSE_TIP), p(self.FACE_TOP))
        features[23] = nose_to_chin / max(nose_to_forehead, 1e-6)

        key_distances = [
            features[0],
            features[1],
            features[8],
            features[9],
            features[14],
            features[15],
            features[16],
            features[22],
        ]
        features[24] = np.std(key_distances)

        return features

    def close(self):
        self.landmarker.close()


if __name__ == "__main__":
    extractor = LandmarkFeatureExtractor()
    print("LandmarkFeatureExtractor khởi tạo thành công.")
    extractor.close()
